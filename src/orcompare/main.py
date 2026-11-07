# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Main entry point for comparing OpenRouter models."""

from asyncio import run
from logging import INFO, WARNING, basicConfig, getLogger
from typing import Annotated
lazy from collections.abc import Sequence

from openrouter import OpenRouter
from pydantic_settings import BaseSettings, SettingsConfigDict
lazy from pydantic import Field, NonNegativeInt

import orcompare

from .data.columns import Column
from .data.filters import Filter
from .data.model_infos import ModelInfo
from .openrouter import fetch_endpoints, fetch_models, save_as_preset
from .storage import get_api_key, get_management_key


class Options(BaseSettings):  # ruff: ignore[undocumented-public-class]
    __doc__ = orcompare.__doc__

    model_config = SettingsConfigDict(
        cli_kebab_case=True, cli_parse_args=True, nested_model_default_partial_update=True
    )

    filter: Annotated[Filter, Field(description="filter criteria to apply")] = Filter.load_default()
    save_as_preset: Annotated[
        str | None,
        Field(
            description="if set, save the top results (number taken from --preset-top) as numerated presets with "
            "this name"
        ),
    ] = None
    preset_top: Annotated[
        NonNegativeInt,
        Field(description="number of top results to save as presets (only relevant if --save-as-preset is set)"),
    ] = 5


def main() -> None:
    """Run main entry point."""
    run(main_async())


async def main_async() -> None:
    """Run main entry point asynchronously."""
    basicConfig(format="[%(levelname)s] %(message)s", level=INFO)
    getLogger("httpx").setLevel(WARNING)
    options = Options()
    await ModelInfo.mixed_price_weights.from_openrouter(get_management_key())
    async with OpenRouter(api_key=get_api_key()) as client:
        model_infos = await fetch_models(client, options.filter)
        model_infos = await fetch_endpoints(client, model_infos, options.filter)
        model_infos = sort_and_select(model_infos, options.filter)
        print(format_model_infos(model_infos, options.filter.columns))  # ruff: ignore[print]
        if options.save_as_preset:
            for idx, model_info in enumerate(model_infos[: options.preset_top]):
                await save_as_preset(client, model_info, f"{options.save_as_preset}{idx + 1}")


def sort_and_select(model_infos: Sequence[ModelInfo], filter_: Filter) -> list[ModelInfo]:
    """Perform final selection of models and sort them.

    1. Select all non-rejected models, or, if ``filter_.invert`` is ``True``, select all rejected models which were not
       rejected due to model or provider exclusion.
    2. Sort the resulting subset by ``filter_.order_by``, taking ``filter_.descending`` into account.
    3. Return at most the first ``filter_.top`` models.

    :param model_infos: Sequence of model information.
    :param filter: Filter to apply to models.
    :return: Sorted subset of models.
    """
    if filter_.invert:
        model_infos = [
            m
            for m in model_infos
            if Column.MODEL_ID not in m.rejection_reasons
            and Column.PROVIDER_NAME not in m.rejection_reasons
            and m.rejection_reasons
        ]
    else:
        model_infos = [m for m in model_infos if not m.rejected]
    model_infos.sort(key=filter_.order_by.key_function, reverse=filter_.descending)
    return model_infos[: filter_.top]


def format_model_infos(model_infos: Sequence[ModelInfo], columns: Sequence[Column]) -> str:
    """Format the given sequence of model information as a text table.

    :param model_infos: Sequence of model information to format.
    :param columns: Columns to include.
    :return: Formatted text table.
    """
    left_aligned = [c.left_aligned for c in columns]
    rows = [[c.header for c in columns]] + [[c.format_value(m) for c in columns] for m in model_infos]
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
    rows.insert(1, [w * "-" for w in widths])
    lines = []
    for row in rows:
        cells = [v.ljust(w) if left else v.rjust(w) for left, v, w in zip(left_aligned, row, widths, strict=True)]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)

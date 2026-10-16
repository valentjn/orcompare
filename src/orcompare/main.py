# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Main entry point for comparing OpenRouter models."""

from asyncio import run
from copy import deepcopy
from logging import INFO, WARNING, basicConfig, getLogger
from typing import Annotated
lazy from collections.abc import Sequence

from openrouter import OpenRouter
from openrouter.components import ProviderPreferences, UnifiedBenchmarksAAItem
from pydantic_settings import BaseSettings, SettingsConfigDict
lazy from pydantic import Field, NonNegativeInt

import orcompare

from .data.columns import Column
from .data.filters import Filter
from .data.model_infos import ModelInfo
from .storage import get_api_key, get_management_key

_logger = getLogger(__name__)


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
        await fetch_benchmarks(client, model_infos, options.filter)
        model_infos = await fetch_endpoints(client, model_infos, options.filter)
        model_infos = sort_and_select(model_infos, options.filter)
        print(format_model_infos(model_infos, options.filter.columns))  # ruff: ignore[print]
        if options.save_as_preset:
            for idx, model_info in enumerate(model_infos[: options.preset_top]):
                await save_as_preset(client, model_info, f"{options.save_as_preset}{idx + 1}")


async def fetch_models(client: OpenRouter, filter_: Filter) -> list[ModelInfo]:
    """Fetch models from OpenRouter and apply model-level filters.

    :param client: OpenRouter client instance.
    :param filter_: Filter to apply to models.
    :return: List of model information with model-level rejection reasons applied.
    """
    _logger.info("fetching models")
    page = await client.models.list_async(
        context=None if filter_.min_context_size is None else filter_.min_context_size,
        zdr="true" if filter_.require_zdr else None,
    )
    model_infos = []
    while page:
        for model in page.result.data:
            model_info = ModelInfo(model)
            filter_.add_model_rejection_reasons(model_info)
            model_infos.append(model_info)
        page = await page.next()  # type: ignore[misc]  # ty: ignore[invalid-await]
    return sorted(model_infos, key=lambda m: m.model_id)


async def fetch_benchmarks(client: OpenRouter, model_infos: Sequence[ModelInfo], filter_: Filter) -> None:
    """Fetch models from OpenRouter and apply benchmark-level filters.

    :param client: OpenRouter client instance.
    :param model_infos: Sequence of model information. Modified in-place to apply benchmark-level rejection reasons.
    :param filter_: Filter to apply to models.
    """
    _logger.info("fetching benchmarks")
    response = await client.benchmarks.get_benchmarks_async(source="artificial-analysis")
    for benchmark in response.data:
        model_info = search_model_info(model_infos, benchmark.model_permaslug) or search_model_info(
            model_infos, benchmark.display_name
        )
        if model_info:
            if not isinstance(benchmark, UnifiedBenchmarksAAItem):
                msg = f"expected benchmark to be UnifiedBenchmarksAAItem, got {type(benchmark).__name__}"
                raise TypeError(msg)
            model_info.benchmark = benchmark
    for model_info in model_infos:
        filter_.add_benchmark_rejection_reasons(model_info)


async def fetch_endpoints(client: OpenRouter, model_infos: Sequence[ModelInfo], filter_: Filter) -> list[ModelInfo]:
    """Fetch endpoints from OpenRouter and apply benchmark-level filters.

    :param client: OpenRouter client instance.
    :param model_infos: Sequence of model information.
    :param filter_: Filter to apply to models.
    :return: List of model information with endpoint-level rejection reasons applied.
    """
    result = []
    if filter_.require_zdr:
        _logger.info("fetching ZDR endpoints")
        zdr_response = await client.endpoints.list_zdr_endpoints_async()
        for endpoint in zdr_response.data:
            if model_info := search_model_info(model_infos, endpoint.model_id):
                new_model_info = deepcopy(model_info)
                new_model_info.endpoint = endpoint
                filter_.add_endpoint_rejection_reasons(new_model_info)
                result.append(new_model_info)
    else:
        _logger.info("fetching endpoints")
        for model_info in model_infos:
            if model_info.rejected:
                result.append(model_info)
                continue
            author, slug = model_info.model_id.split("/", 1)
            response = await client.endpoints.list_async(author=author, slug=slug)
            for endpoint in response.data.endpoints:
                new_model_info = deepcopy(model_info)
                new_model_info.endpoint = endpoint
                filter_.add_endpoint_rejection_reasons(new_model_info)
                result.append(new_model_info)
    return result


def search_model_info(model_infos: Sequence[ModelInfo], string: str) -> ModelInfo | None:
    """Search a model in the given sequence of model information.

    Since OpenRouter doesn't seem to use model ID, slug, canonical slug, permanent slug, and model name consistently,
    this function returns the first model information in the given sequence for which the model ID, the canonical slug,
    or the model name equals the string. Model IDs starting with ``@`` are excluded (these are not real models, e.g.,
    presets start with ``@preset/``).

    :param model_infos: Model information to search.
    :param string: String to search for.
    :return: Found model information, or ``None`` if none matches.
    """
    return next(
        (
            m
            for m in model_infos
            if string in {m.model.id, m.model.canonical_slug, m.model.name} and not m.model.id.startswith("@")
        ),
        None,
    )


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


async def save_as_preset(client: OpenRouter, model_info: ModelInfo, name: str) -> None:
    """Save the given model information as an OpenRouter preset (updating it if it already exists).

    :param client: OpenRouter client instance.
    :param model_info: Model information to save as preset.
    :param name: Name of the preset. Must be a valid slug.
    """
    if not model_info.provider_id:
        msg = "provider ID not set"
        raise ValueError(msg)
    _logger.info("saving %s by %s as preset %s", model_info.model_id, model_info.provider_id, name)
    await client.presets.create_presets_chat_completions_async(
        slug=name,
        messages=[],
        model=model_info.model_id,
        provider=ProviderPreferences(allow_fallbacks=False, only=[model_info.provider_id]),
    )

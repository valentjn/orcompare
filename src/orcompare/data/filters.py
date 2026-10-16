# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Filters for OpenRouter model endpoints."""

from typing import TYPE_CHECKING, Annotated, Self
lazy import re
lazy from pathlib import Path

import yaml
from pydantic import Field, NegativeFloat, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator
from pydantic_settings import BaseSettings

try:
    from yaml import CDumper as Dumper
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import Dumper, SafeLoader  # type: ignore[assignment]

from orcompare.storage import get_storage_dir

from .columns import Column
lazy from .model_infos import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


type Percentage = Annotated[float, Field(ge=0, le=100)]
"""A percentage value constrained to the range [0, 100]."""


class ModelProviderFilter(BaseSettings):
    """Filter for model/provider pairs."""

    model: Annotated[re.Pattern[str], Field(description="regular expression for the model ID")]
    provider: Annotated[re.Pattern[str], Field(description="regular expression for the provider ID")]


class Filter(BaseSettings):
    """A filter for OpenRouter model endpoints."""

    min_age: Annotated[NegativeFloat | None, Field(description="minimum model age (days)")] = None
    max_age: Annotated[NonNegativeFloat | None, Field(description="maximum model age (days)")] = None
    min_context_size: Annotated[NonNegativeInt | None, Field(description="minimum context size (tokens)")] = None
    max_context_size: Annotated[NonNegativeInt | None, Field(description="maximum context size (tokens)")] = None
    min_quality: Annotated[
        Percentage | None, Field(description="minimum quality (Artificial Analysis Intelligence Index in percent)")
    ] = None
    max_quality: Annotated[
        Percentage | None, Field(description="maximum quality (Artificial Analysis Intelligence Index in percent)")
    ] = None

    min_input_price: Annotated[
        NonNegativeFloat | None, Field(description="minimum uncached input token price (dollars per million tokens)")
    ] = None
    max_input_price: Annotated[
        NonNegativeFloat | None, Field(description="maximum uncached input token price (dollars per million tokens)")
    ] = None
    min_output_price: Annotated[
        NonNegativeFloat | None, Field(description="minimum output token price (dollars per million tokens)")
    ] = None
    max_output_price: Annotated[
        NonNegativeFloat | None, Field(description="maximum output token price (dollars per million tokens)")
    ] = None
    min_cached_input_price: Annotated[
        NonNegativeFloat | None, Field(description="minimum cached input token price (dollars per million tokens)")
    ] = None
    max_cached_input_price: Annotated[
        NonNegativeFloat | None, Field(description="maximum cached input token price (dollars per million tokens)")
    ] = None
    min_mixed_price: Annotated[
        NonNegativeFloat | None,
        Field(
            description="minimum mixed price (weighted mean of token prices, in dollars per million uncached input "
            "tokens)"
        ),
    ] = None
    max_mixed_price: Annotated[
        NonNegativeFloat | None,
        Field(
            description="minimum mixed price (weighted mean of token prices, in dollars per million uncached input "
            "tokens)"
        ),
    ] = None
    min_discount: Annotated[Percentage | None, Field(description="minimum discount (percent)")] = None
    max_discount: Annotated[Percentage | None, Field(description="maximum discount (percent)")] = None
    min_latency: Annotated[
        NonNegativeFloat | None,
        Field(description="minimum median latency in past 30 minutes (time to first token in seconds)"),
    ] = None
    max_latency: Annotated[
        NonNegativeFloat | None,
        Field(description="maximum median latency in past 30 minutes (time to first token in seconds)"),
    ] = None
    min_throughput: Annotated[
        NonNegativeFloat | None, Field(description="minimum median throughput in past 30 minutes (tokens per second)")
    ] = None
    max_throughput: Annotated[
        NonNegativeFloat | None, Field(description="maximum median throughput in past 30 minutes (tokens per second)")
    ] = None
    min_uptime: Annotated[Percentage | None, Field(description="minimum uptime (percent)")] = None
    max_uptime: Annotated[Percentage | None, Field(description="maximum uptime (percent)")] = None

    include_models: Annotated[
        re.Pattern[str] | None, Field(description="the model ID must match this regular expression")
    ] = None
    exclude_models: Annotated[
        re.Pattern[str] | None, Field(description="the model ID must not match this regular expression")
    ] = None

    include_providers: Annotated[
        re.Pattern[str] | None, Field(description="the provider ID must match this regular expression")
    ] = None
    exclude_providers: Annotated[
        re.Pattern[str] | None, Field(description="the provider ID must not match this regular expression")
    ] = None

    include_model_providers: Annotated[
        list[ModelProviderFilter] | None,
        Field(description="the model ID/provider ID combination must match one of these"),
    ] = None
    exclude_model_providers: Annotated[
        list[ModelProviderFilter] | None,
        Field(description="the model ID/provider ID combination must not match one of these"),
    ] = None

    require_zdr: Annotated[bool, Field(description="only use providers with zero data retention (ZDR)")] = False

    invert: Annotated[bool, Field(description="only show results which do not match")] = False

    order_by: Annotated[Column, Field(description="order results by this key")] = Column.MIXED_PRICE
    descending: Annotated[bool, Field(description="order results in descending order")] = False

    top: Annotated[PositiveInt, Field(description="number of top results to show")] = 10

    columns: Annotated[list[Column], Field(description="show these columns")] = [
        Column.MODEL_ID,
        Column.CONTEXT_SIZE,
        Column.QUALITY,
        Column.PROVIDER_ID,
        Column.INPUT_PRICE,
        Column.OUTPUT_PRICE,
        Column.CACHED_INPUT_PRICE,
        Column.MIXED_PRICE,
        Column.DISCOUNT,
        Column.THROUGHPUT,
        Column.UPTIME,
    ]

    @model_validator(mode="after")
    def _validate_min_not_greater_than_max(self) -> Self:
        """Validate that all ``min_`` fields are not greater than the corresponding ``max_`` field.

        :return: The validated filter instance.
        """
        for min_name in Filter.model_fields:
            if min_name.startswith("min_"):
                min_value = getattr(self, min_name)
                max_name = f"max_{min_name.removeprefix('min_')}"
                max_value = getattr(self, max_name)
                if min_value is not None and max_value is not None and min_value > max_value:
                    msg = f"{min_name} = {min_value} must be lower than {max_name} = {max_value}"
                    raise ValueError(msg)
        return self

    @staticmethod
    def load_default() -> Filter:
        """Load the default filter from disk.

        :return: Loaded filter or a new default filter if none exists.
        """
        path = Filter._get_default_path()
        return Filter.model_validate(yaml.load(path.read_bytes(), SafeLoader)) if path.exists() else Filter()

    def save_default(self) -> None:
        """Save the current filter to disk as the default."""
        path = Filter._get_default_path()
        path.parent.mkdir(exist_ok=True)
        yaml.dump(self.model_dump(exclude_unset=True), Dumper=Dumper)

    @staticmethod
    def _get_default_path() -> Path:
        """Return the file path of the default filter.

        :return: Path to the default filter YAML file.
        """
        return get_storage_dir() / "default_filter.yaml"

    def add_model_rejection_reasons(self, model_info: ModelInfo) -> None:
        """Add rejection reasons based on model-level filters.

        :param model_info: Model information to annotate with rejection reasons.
        """
        if self.include_models and (not (m := model_info.model_id) or not self.include_models.search(m)):
            model_info.rejection_reasons.add(Column.MODEL_ID)
        if self.exclude_models and (not (m := model_info.model_id) or self.exclude_models.search(m)):
            model_info.rejection_reasons.add(Column.MODEL_ID)
        self._add_rejection_reasons(model_info, Column.AGE)
        self._add_rejection_reasons(model_info, Column.CONTEXT_SIZE)

    def add_endpoint_rejection_reasons(self, model_info: ModelInfo) -> None:
        """Add rejection reasons based on endpoint-level filters.

        :param model_info: Model information to annotate with rejection reasons.
        """
        if self.include_providers and (not (p := model_info.provider_id) or not self.include_providers.search(p)):
            model_info.rejection_reasons.add(Column.PROVIDER_ID)
        if self.exclude_providers and (not (p := model_info.provider_id) or self.exclude_providers.search(p)):
            model_info.rejection_reasons.add(Column.PROVIDER_ID)
        if self.include_model_providers is not None and (
            not (m := model_info.model_id)
            or not (p := model_info.provider_id)
            or not any(i.model.search(m) and i.provider.search(p) for i in self.include_model_providers)
        ):
            model_info.rejection_reasons |= {Column.MODEL_ID, Column.PROVIDER_ID}
        if self.exclude_model_providers is not None and (
            not (m := model_info.model_id)
            or not (p := model_info.provider_id)
            or any(i.model.search(m) and i.provider.search(p) for i in self.exclude_model_providers)
        ):
            model_info.rejection_reasons |= {Column.MODEL_ID, Column.PROVIDER_ID}
        self._add_rejection_reasons(model_info, Column.INPUT_PRICE)
        self._add_rejection_reasons(model_info, Column.OUTPUT_PRICE)
        self._add_rejection_reasons(model_info, Column.CACHED_INPUT_PRICE)
        self._add_rejection_reasons(model_info, Column.MIXED_PRICE)
        self._add_rejection_reasons(model_info, Column.DISCOUNT)
        self._add_rejection_reasons(model_info, Column.LATENCY)
        self._add_rejection_reasons(model_info, Column.THROUGHPUT)
        self._add_rejection_reasons(model_info, Column.UPTIME)

    def add_benchmark_rejection_reasons(self, model_info: ModelInfo) -> None:
        """Add rejection reasons based on benchmark-level filters.

        :param model_info: Model information to annotate with rejection reasons.
        """
        self._add_rejection_reasons(model_info, Column.QUALITY)

    def _add_rejection_reasons(self, model_info: ModelInfo, column: Column) -> None:
        """Add rejection reasons for a specific column based on min/max filters.

        :param model_info: Model information to annotate with rejection reasons.
        :param column: Column to check against min/max filter values.
        """
        value = column.get_value(model_info)
        minimum_name = f"min_{column.value}"
        maximum_name = f"max_{column.value}"
        if (m := getattr(self, minimum_name)) is not None and (value is None or value < m):
            model_info.rejection_reasons.add(minimum_name)
        if (m := getattr(self, maximum_name)) is not None and (value is None or value > m):
            model_info.rejection_reasons.add(maximum_name)

    def filter_and_sort[T](
        self, items: Mapping[str, T], check_function: Callable[[T], bool], key_function: Callable[[T], float]
    ) -> dict[str, T]:
        """Filter and sort items by a check function and sort key.

        :param items: Mapping of slugs to items.
        :param check_function: Predicate to determine if an item passes the filter.
        :param key_function: Function to extract the sort key from an item.
        :return: Filtered and sorted mapping of slugs to items.
        """
        slugs = sorted(
            (id_ for id_, i in items.items() if check_function(i)),
            key=lambda s: key_function(items[s]),
            reverse=self.descending,
        )
        return {s: items[s] for s in slugs}

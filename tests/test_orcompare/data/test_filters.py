# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from unittest.mock import patch
lazy from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orcompare.data import filters
from orcompare.data.columns import Column
from orcompare.data.filters import Filter, ModelProviderFilter
lazy from orcompare.data.model_infos import ModelInfo


class TestFilter:
    @staticmethod
    def test_validate_min_not_greater_than_max_valid() -> None:
        filter_ = Filter(min_context_size=10, max_context_size=10)
        assert (filter_.min_context_size, filter_.max_context_size) == (10, 10)

    @staticmethod
    def test_validate_min_not_greater_than_max_invalid() -> None:
        with pytest.raises(ValidationError, match=r"min_context_size = 20 must be lower than max_context_size = 10"):
            Filter(min_context_size=20, max_context_size=10)

    @staticmethod
    def test_load_default_missing_file(tmp_path: Path) -> None:
        path = tmp_path / "default_filter.yaml"
        with patch.object(Filter, "_get_default_path", return_value=path):
            assert Filter.load_default() == Filter()

    @staticmethod
    def test_load_default_existing_file(tmp_path: Path) -> None:
        path = tmp_path / "default_filter.yaml"
        path.write_text("min_context_size: 1000\ninclude_models: '^author/'\n")
        with patch.object(Filter, "_get_default_path", return_value=path):
            filter_ = Filter.load_default()
        assert filter_.min_context_size == 1000
        assert filter_.include_models
        assert filter_.include_models.pattern == r"^author/"

    @staticmethod
    def test_save_default(tmp_path: Path) -> None:
        path = tmp_path / "storage" / "default_filter.yaml"
        filter_ = Filter(min_context_size=1000, include_models=r"^author/")
        with patch.object(Filter, "_get_default_path", return_value=path):
            filter_.save_default()
        assert yaml.safe_load(path.read_text()) == {"include_models": "^author/", "min_context_size": 1000}

    @staticmethod
    def test_get_default_path(tmp_path: Path) -> None:
        with patch.object(filters, "get_storage_dir", return_value=tmp_path):
            assert Filter._get_default_path() == tmp_path / "default_filter.yaml"

    @staticmethod
    @pytest.mark.parametrize(
        ("filter_", "expected_reasons"),
        [
            (Filter(), set()),
            (Filter(include_models=r"^openai/"), set()),
            (Filter(include_models=r"^other/"), {Column.MODEL_ID}),
            (Filter(exclude_models=r"^openai/"), {Column.MODEL_ID}),
            (Filter(exclude_models=r"^other/"), set()),
            (Filter(max_age=50), {"max_age"}),
            (Filter(max_age=200), set()),
            (Filter(min_context_size=200_000), {"min_context_size"}),
            (Filter(max_context_size=50_000), {"max_context_size"}),
            (Filter(min_quality=50), {"min_quality"}),
            (Filter(max_quality=10), {"max_quality"}),
        ],
    )
    def test_add_model_rejection_reasons(filter_: Filter, expected_reasons: set[str], model_info: ModelInfo) -> None:
        filter_.add_model_rejection_reasons(model_info)
        assert model_info.rejection_reasons == expected_reasons

    @staticmethod
    @pytest.mark.parametrize(
        ("filter_", "with_endpoint", "expected_reasons"),
        [
            (Filter(), True, set()),
            (Filter(include_providers=r"^openai$"), True, set()),
            (Filter(include_providers=r"^other$"), True, {Column.PROVIDER_ID}),
            (Filter(exclude_providers=r"^openai$"), True, {Column.PROVIDER_ID}),
            (Filter(exclude_providers=r"^other$"), True, set()),
            (Filter(include_providers=r"^openai$"), False, {Column.PROVIDER_ID}),
            (
                Filter(include_model_providers=[ModelProviderFilter(model=r"^openai/", provider=r"^openai$")]),
                True,
                set(),
            ),
            (
                Filter(include_model_providers=[ModelProviderFilter(model=r"^other/", provider=r"^openai$")]),
                True,
                {Column.MODEL_ID, Column.PROVIDER_ID},
            ),
            (
                Filter(exclude_model_providers=[ModelProviderFilter(model=r"^openai/", provider=r"^openai$")]),
                True,
                {Column.MODEL_ID, Column.PROVIDER_ID},
            ),
            (
                Filter(exclude_model_providers=[ModelProviderFilter(model=r"^other/", provider=r"^openai$")]),
                True,
                set(),
            ),
            (Filter(min_input_price=3), True, {"min_input_price"}),
            (Filter(max_input_price=2), True, {"max_input_price"}),
            (Filter(min_output_price=11), True, {"min_output_price"}),
            (Filter(max_output_price=9), True, {"max_output_price"}),
            (Filter(min_cached_input_price=2), True, {"min_cached_input_price"}),
            (Filter(max_cached_input_price=1), True, {"max_cached_input_price"}),
            (Filter(min_mixed_price=30), True, {"min_mixed_price"}),
            (Filter(max_mixed_price=28), True, {"max_mixed_price"}),
            (Filter(min_discount=30), True, {"min_discount"}),
            (Filter(max_discount=20), True, {"max_discount"}),
            (Filter(min_latency=1), True, {"min_latency"}),
            (Filter(max_latency=0.1), True, {"max_latency"}),
            (Filter(min_throughput=200), True, {"min_throughput"}),
            (Filter(max_throughput=50), True, {"max_throughput"}),
            (Filter(min_uptime=100), True, {"min_uptime"}),
            (Filter(max_uptime=90), True, {"max_uptime"}),
        ],
    )
    def test_add_endpoint_rejection_reasons(
        filter_: Filter, *, with_endpoint: bool, expected_reasons: set[str], model_info: ModelInfo
    ) -> None:
        if not with_endpoint:
            model_info.endpoint = None
        filter_.add_endpoint_rejection_reasons(model_info)
        assert model_info.rejection_reasons == expected_reasons

    @staticmethod
    @pytest.mark.parametrize(
        ("filter_", "column", "expected_reasons"),
        [
            (Filter(), Column.CONTEXT_SIZE, set()),
            (Filter(min_context_size=200_000), Column.CONTEXT_SIZE, {"min_context_size"}),
            (Filter(max_context_size=50_000), Column.CONTEXT_SIZE, {"max_context_size"}),
            (Filter(min_quality=50), Column.QUALITY, {"min_quality"}),
            (Filter(max_quality=40), Column.QUALITY, {"max_quality"}),
        ],
    )
    def test_add_rejection_reasons(
        filter_: Filter, column: Column, expected_reasons: set[str], model_info: ModelInfo
    ) -> None:
        filter_._add_rejection_reasons(model_info, column)
        assert model_info.rejection_reasons == expected_reasons

    @staticmethod
    @pytest.mark.parametrize(("descending", "expected_keys"), [(False, ["a", "c"]), (True, ["c", "a"])])
    def test_filter_and_sort(*, descending: bool, expected_keys: list[str]) -> None:
        filter_ = Filter(descending=descending)
        items = {"a": 1, "b": 2, "c": 3}
        result = filter_.filter_and_sort(items, lambda x: x != 2, lambda x: x)
        assert list(result) == expected_keys

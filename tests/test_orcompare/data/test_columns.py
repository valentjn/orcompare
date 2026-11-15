# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

import orcompare.data.columns
from orcompare.data.columns import Column
lazy from orcompare.data.model_infos import ModelInfo


@pytest.mark.parametrize("column", list(Column))
def test_repr(column: Column) -> None:
    assert repr(column) == column.value


@pytest.mark.parametrize("column", list(Column))
def test_str(column: Column) -> None:
    assert str(column) == column.value


@pytest.mark.parametrize("column", list(Column))
def test_header(column: Column) -> None:
    assert (
        column.header
        == {
            Column.AGE: "Age",
            Column.CACHED_INPUT_PRICE: "CIP",
            Column.CONTEXT_SIZE: "CS",
            Column.DISCOUNT: "Ds",
            Column.INPUT_PRICE: "IP",
            Column.LATENCY: "Lat",
            Column.MIXED_PRICE: "Mx",
            Column.MODEL_ID: "Model ID",
            Column.MODEL_NAME: "Model",
            Column.OUTPUT_PRICE: "OP",
            Column.PROVIDER_ID: "Provider ID",
            Column.PROVIDER_NAME: "Provider",
            Column.QUALITY: "Q",
            Column.REJECTION_REASONS: "Rejected",
            Column.THROUGHPUT: "TP",
            Column.UPTIME: "UT",
        }[column]
    )


@pytest.mark.parametrize("column", list(Column))
def test_left_aligned(column: Column) -> None:
    left_aligned_columns = {
        Column.MODEL_ID,
        Column.MODEL_NAME,
        Column.PROVIDER_ID,
        Column.PROVIDER_NAME,
        Column.REJECTION_REASONS,
    }
    assert column.left_aligned == (column in left_aligned_columns)


def test_get_value_age(model_info: ModelInfo) -> None:
    assert Column.AGE.get_value(model_info) == pytest.approx(100, abs=0.01)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column.CACHED_INPUT_PRICE, 1.25),
        (Column.CONTEXT_SIZE, 128_000),
        (Column.DISCOUNT, 25.0),
        (Column.INPUT_PRICE, 2.5),
        (Column.LATENCY, 0.5),
        (Column.MIXED_PRICE, 29.0),
        (Column.MODEL_ID, "openai/gpt-4o"),
        (Column.MODEL_NAME, "GPT-4o"),
        (Column.OUTPUT_PRICE, 10.0),
        (Column.PROVIDER_ID, "openai"),
        (Column.PROVIDER_NAME, "OpenAI"),
        (Column.QUALITY, 42.0),
        (Column.THROUGHPUT, 100.0),
        (Column.UPTIME, 99.0),
    ],
)
def test_get_value(column: Column, expected: float | str, model_info: ModelInfo) -> None:
    value = column.get_value(model_info)
    if isinstance(expected, float):
        assert value == pytest.approx(expected)
    else:
        assert value == expected


def test_get_value_rejection_reasons(model_info: ModelInfo) -> None:
    model_info.rejection_reasons = {Column.UPTIME.value, Column.AGE.value}
    assert Column.REJECTION_REASONS.get_value(model_info) == "age,uptime"


def test_format_value_age(model_info: ModelInfo) -> None:
    assert Column.AGE.format_value(model_info) == "100"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column.CACHED_INPUT_PRICE, "125.00"),
        (Column.CONTEXT_SIZE, "128K"),
        (Column.DISCOUNT, "25"),
        (Column.INPUT_PRICE, "250"),
        (Column.LATENCY, "0.5"),
        (Column.MIXED_PRICE, "2900"),
        (Column.MODEL_ID, "openai/gpt-4o"),
        (Column.MODEL_NAME, "GPT-4o"),
        (Column.OUTPUT_PRICE, "1000"),
        (Column.PROVIDER_ID, "openai"),
        (Column.PROVIDER_NAME, "OpenAI"),
        (Column.QUALITY, "42"),
        (Column.REJECTION_REASONS, ""),
        (Column.THROUGHPUT, "100"),
        (Column.UPTIME, "99"),
    ],
)
def test_format_value(column: Column, expected: str, model_info: ModelInfo) -> None:
    assert column.format_value(model_info) == expected


@pytest.mark.parametrize(
    "column",
    [
        Column.CACHED_INPUT_PRICE,
        Column.CONTEXT_SIZE,
        Column.DISCOUNT,
        Column.INPUT_PRICE,
        Column.LATENCY,
        Column.MIXED_PRICE,
        Column.OUTPUT_PRICE,
        Column.PROVIDER_ID,
        Column.PROVIDER_NAME,
        Column.QUALITY,
        Column.THROUGHPUT,
        Column.UPTIME,
    ],
)
def test_format_value_none(column: Column, model_info: ModelInfo) -> None:
    model_info.model.context_length = None
    model_info.model.benchmarks = None
    model_info.endpoint = None
    assert column.format_value(model_info) == "N/A"


@pytest.mark.parametrize(("context_length", "expected"), [(1_000_000, "1.0M"), (2_500_000, "2.5M")])
def test_format_value_context_size_megatokens(context_length: int, expected: str, model_info: ModelInfo) -> None:
    model_info.model.context_length = context_length
    assert Column.CONTEXT_SIZE.format_value(model_info) == expected


def test_format_value_unsupported_column(model_info: ModelInfo) -> None:
    column = MagicMock()
    column.value = "unsupported"
    column.get_value.return_value = 1.0
    with pytest.raises(ValueError, match="unsupported column: unsupported"):
        Column.format_value(cast("Column", column), model_info)


@pytest.mark.parametrize(("column", "expected"), [(Column.INPUT_PRICE, 2.5), (Column.MODEL_ID, "openai/gpt-4o")])
def test_key_function(column: Column, expected: float | str, model_info: ModelInfo) -> None:
    value = column.key_function(model_info)
    if isinstance(expected, float):
        assert value == pytest.approx(expected)
    else:
        assert value == expected


def test_key_function_none_value_float(model_info: ModelInfo) -> None:
    model_info.endpoint = None
    assert Column.INPUT_PRICE.key_function(model_info) == 0


def test_key_function_none_value_str(model_info: ModelInfo) -> None:
    model_info.endpoint = None
    with patch.object(orcompare.data.columns, "signature", return_value=MagicMock(return_annotation=str)):
        assert not Column.INPUT_PRICE.key_function(model_info)


@pytest.mark.parametrize(
    ("return_annotation", "message"),
    [
        (float | str, "expected second Union argument in return type of ModelInfo.input_price to be None, got str"),
        (float | str | None, "expected two Union arguments in return type of ModelInfo.input_price, got 3"),
        (int, "expected return type of ModelInfo.input_price to be either float or str, got int"),
    ],
)
def test_key_function_none_value_invalid_return_annotation(
    return_annotation: Any,  # ruff: ignore[any-type]
    message: str,
    model_info: ModelInfo,
) -> None:
    model_info.endpoint = None
    with (
        patch.object(orcompare.data.columns, "signature", return_value=MagicMock(return_annotation=return_annotation)),
        pytest.raises(TypeError, match=re.escape(message)),
    ):
        Column.INPUT_PRICE.key_function(model_info)

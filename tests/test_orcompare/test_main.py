# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import builtins
import sys
from textwrap import dedent
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
lazy from openrouter.components import Model

import orcompare.main
from orcompare.data.columns import Column
from orcompare.data.filters import Filter
from orcompare.data.model_infos import ModelInfo
from orcompare.main import Options, format_model_infos, main, main_async, sort_and_select


def create_model_info(model: Model, model_id: str, rejection_reasons: set[str] | None = None) -> ModelInfo:
    return ModelInfo(
        model=model.model_copy(update={"id": model_id}),
        rejection_reasons=set() if rejection_reasons is None else rejection_reasons,
    )


def create_client_mock() -> MagicMock:
    client = MagicMock()
    client.__aenter__.return_value = client
    return client


class TestOptions:
    @staticmethod
    def test_defaults() -> None:
        with patch.object(sys, "argv", ["orcompare"]):
            options = Options()
        assert options.filter == Options.model_fields["filter"].default
        assert options.save_as_preset is None
        assert options.preset_top == 5

    @staticmethod
    def test_cli_arguments() -> None:
        argv = [
            "orcompare",
            "--filter.min-context-size",
            "1000",
            "--filter.order-by",
            "quality",
            "--save-as-preset",
            "fav",
            "--preset-top",
            "3",
        ]
        with patch.object(sys, "argv", argv):
            options = Options()
        assert options.filter.min_context_size == 1000
        assert options.filter.order_by == Column.QUALITY
        assert options.save_as_preset == "fav"
        assert options.preset_top == 3

    @staticmethod
    def test_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORCOMPARE_PRESET_TOP", "7")
        with patch.object(sys, "argv", ["orcompare"]):
            assert Options().preset_top == 7


def test_main() -> None:
    with (
        patch.object(orcompare.main, "run") as run_mock,
        patch.object(orcompare.main, "main_async", MagicMock(return_value=mock.sentinel.coroutine)) as main_async_mock,
    ):
        main()
    assert main_async_mock.call_args_list == [call()]
    assert run_mock.call_args_list == [call(mock.sentinel.coroutine)]


@pytest.mark.parametrize(
    ("save_as_preset", "preset_top", "answer", "expected_preset_names"),
    [
        (None, 5, None, []),
        ("fav", 2, "y", ["fav1", "fav2"]),
        ("fav", 2, " YES ", ["fav1", "fav2"]),
        ("fav", 2, "n", []),
        ("fav", 2, "", []),
    ],
)
async def test_main_async(
    save_as_preset: str | None,
    preset_top: int,
    answer: str | None,
    expected_preset_names: list[str],
    model: Model,
) -> None:
    filter_ = Filter()
    options = MagicMock(filter=filter_, save_as_preset=save_as_preset, preset_top=preset_top)
    model_infos = [create_model_info(model, f"author/model{idx}") for idx in range(3)]
    management_client = create_client_mock()
    api_client = create_client_mock()
    with (
        patch.object(orcompare.main, "Options", return_value=options),
        patch.object(orcompare.main, "get_management_key", return_value="management-key"),
        patch.object(orcompare.main, "get_api_key", return_value="api-key"),
        patch.object(orcompare.main, "OpenRouter", side_effect=[management_client, api_client]) as client_mock,
        patch.object(ModelInfo.mixed_price_weights, "from_openrouter", new_callable=AsyncMock) as weights_mock,
        patch.object(orcompare.main, "fetch_models", new_callable=AsyncMock) as fetch_models_mock,
        patch.object(orcompare.main, "fetch_endpoints", new_callable=AsyncMock) as fetch_endpoints_mock,
        patch.object(orcompare.main, "sort_and_select", return_value=model_infos) as sort_and_select_mock,
        patch.object(orcompare.main, "format_model_infos", return_value="table") as format_model_infos_mock,
        patch.object(orcompare.main, "save_as_preset", new_callable=AsyncMock) as save_as_preset_mock,
        patch.object(builtins, "input", return_value=answer) as input_mock,
        patch.object(builtins, "print") as print_mock,
    ):
        fetch_models_mock.return_value = model_infos
        fetch_endpoints_mock.return_value = model_infos
        await main_async()
    assert client_mock.call_args_list == [call(api_key="management-key"), call(api_key="api-key")]
    assert weights_mock.call_args_list == [call(management_client)]
    assert fetch_models_mock.call_args_list == [call(api_client, filter_)]
    assert fetch_endpoints_mock.call_args_list == [call(api_client, model_infos, filter_)]
    assert sort_and_select_mock.call_args_list == [call(model_infos, filter_)]
    assert format_model_infos_mock.call_args_list == [call(model_infos, filter_.columns)]
    assert print_mock.call_args_list == [call("table", flush=True)]
    assert input_mock.call_args_list == ([] if save_as_preset is None else [call("Apply these changes? [y/N]\n")])
    assert save_as_preset_mock.call_args_list == [
        call(management_client, model_infos[idx], name) for idx, name in enumerate(expected_preset_names)
    ]


@pytest.mark.parametrize(
    ("filter_", "expected_model_ids"),
    [
        (Filter(order_by=Column.MODEL_ID), ["a", "b", "c"]),
        (Filter(order_by=Column.MODEL_ID, descending=True), ["c", "b", "a"]),
        (Filter(order_by=Column.MODEL_ID, top=2), ["a", "b"]),
    ],
)
def test_sort_and_select_ordering(filter_: Filter, expected_model_ids: list[str], model: Model) -> None:
    model_infos = [create_model_info(model, model_id) for model_id in ["b", "c", "a"]]
    result = sort_and_select(model_infos, filter_)
    assert [m.model_id for m in result] == expected_model_ids


def test_sort_and_select_rejected(model: Model) -> None:
    model_infos = [
        create_model_info(model, "a"),
        create_model_info(model, "b", {Column.AGE.value}),
        create_model_info(model, "c"),
    ]
    result = sort_and_select(model_infos, Filter(order_by=Column.MODEL_ID))
    assert [m.model_id for m in result] == ["a", "c"]


def test_sort_and_select_invert(model: Model) -> None:
    model_infos = [
        create_model_info(model, "a"),
        create_model_info(model, "b", {Column.AGE.value}),
        create_model_info(model, "c", {Column.MODEL_ID.value}),
        create_model_info(model, "d", {Column.PROVIDER_NAME.value}),
    ]
    result = sort_and_select(model_infos, Filter(order_by=Column.MODEL_ID, invert=True))
    assert [m.model_id for m in result] == ["b"]


def test_format_model_infos_table(model: Model) -> None:
    model_infos = [ModelInfo(model=model)]
    columns = [Column.MODEL_ID, Column.MODEL_NAME, Column.CONTEXT_SIZE]
    assert format_model_infos(model_infos, columns) == dedent(
        """\
        Model ID       Model     CS
        -------------  ------  ----
        openai/gpt-4o  GPT-4o  128K
        """.rstrip()
    )


def test_format_model_infos_empty() -> None:
    columns = [Column.MODEL_ID, Column.CONTEXT_SIZE]
    assert format_model_infos([], columns) == dedent(
        """\
        Model ID  CS
        --------  --
        """.rstrip()
    )

# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from openrouter.components import (
    Architecture,
    Model,
    ModelsListResponse,
    ModelsListResponseLinks,
    PercentileStats,
    Pricing,
    ProviderPreferences,
    PublicEndpoint,
    PublicZdrEndpoint,
    ToolChoiceSupport,
)
from openrouter.components import ListEndpointsResponse as ListEndpointsResponseData
from openrouter.operations import GetModelsResponse, ListEndpointsResponse, ListEndpointsZdrResponse

from orcompare.data.columns import Column
from orcompare.data.filters import Filter
from orcompare.data.model_infos import ModelInfo
from orcompare.openrouter import _search_model_info, fetch_endpoints, fetch_models, save_as_preset


def create_models_response(models: list[Model], *, next_: GetModelsResponse | None = None) -> GetModelsResponse:
    return GetModelsResponse(
        next=AsyncMock(return_value=next_),
        result=ModelsListResponse(data=models, links=ModelsListResponseLinks(next=None), total_count=len(models)),
    )


def create_endpoints_response(endpoints: list[PublicEndpoint]) -> ListEndpointsResponse:
    return ListEndpointsResponse(
        data=ListEndpointsResponseData(
            architecture=Architecture(
                input_modalities=[], instruct_type=None, modality=None, output_modalities=[], tokenizer=None
            ),
            created=0,
            description="",
            endpoints=endpoints,
            id="openai/gpt-4o",
            name="GPT-4o",
        )
    )


@pytest.fixture
def models(model: Model) -> list[Model]:
    return [model, model.model_copy(update={"id": "anthropic/claude"})]


@pytest.fixture
def model_infos(model: Model) -> list[ModelInfo]:
    return [ModelInfo(model=model)]


@pytest.fixture
def endpoints(endpoint: PublicEndpoint) -> list[PublicEndpoint]:
    return [endpoint, endpoint.model_copy(update={"tag": "azure", "provider_name": "Azure"})]


@pytest.fixture
def zdr_endpoint() -> PublicZdrEndpoint:
    return PublicZdrEndpoint(
        context_length=128_000,
        latency_last_30m=PercentileStats(p50=0.5, p75=0.5, p90=0.5, p99=0.5),
        max_completion_tokens=None,
        max_prompt_tokens=None,
        model_id="openai/gpt-4o",
        model_name="GPT-4o",
        name="OpenAI",
        native_tools={},
        pricing=Pricing(prompt="0.0000025", completion="0.00001", input_cache_read="0.00000125", discount=0.25),
        provider_name="OpenAI",
        quantization=None,
        service_tier=None,
        supported_parameters=[],
        supports_diarization=False,
        supports_implicit_caching=False,
        supports_tool_choice=ToolChoiceSupport(auto=False, function=False, none=False, required=False),
        tag="openai",
        throughput_last_30m=PercentileStats(p50=100.0, p75=100.0, p90=100.0, p99=100.0),
        uptime_last_1d=None,
        uptime_last_30m=99.0,
        uptime_last_5m=None,
        safety_retention=False,
    )


@pytest.fixture
def zdr_endpoints(zdr_endpoint: PublicZdrEndpoint) -> list[PublicZdrEndpoint]:
    return [zdr_endpoint, zdr_endpoint.model_copy(update={"tag": "azure", "provider_name": "Azure"})]


async def test_fetch_models_default(client: MagicMock, models: list[Model]) -> None:
    with patch.object(client.models, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_models_response(models)
        model_infos = await fetch_models(client, Filter())
    assert list_mock.call_args_list == [call(context=None, zdr=None)]
    assert [m.model_id for m in model_infos] == ["anthropic/claude", "openai/gpt-4o"]
    assert all(m.rejection_reasons == set() for m in model_infos)


async def test_fetch_models_with_filter(client: MagicMock, model: Model) -> None:
    with patch.object(client.models, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_models_response([model])
        await fetch_models(client, Filter(min_context_size=1000, require_zdr=True))
    assert list_mock.call_args_list == [call(context=1000, zdr="true")]


async def test_fetch_models_model_filters(client: MagicMock, models: list[Model]) -> None:
    with patch.object(client.models, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_models_response(models)
        model_infos = await fetch_models(client, Filter(include_models=r"^openai/"))
    assert {m.model_id: m.rejection_reasons for m in model_infos} == {
        "anthropic/claude": {Column.MODEL_ID},
        "openai/gpt-4o": set(),
    }


async def test_fetch_models_pagination(client: MagicMock, model: Model) -> None:
    second_page = create_models_response([model])
    first_page = create_models_response([model.model_copy(update={"id": "anthropic/claude"})], next_=second_page)
    with patch.object(client.models, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = first_page
        model_infos = await fetch_models(client, Filter())
    assert list_mock.call_args_list == [call(context=None, zdr=None)]
    assert [m.model_id for m in model_infos] == ["anthropic/claude", "openai/gpt-4o"]


async def test_fetch_endpoints_default(
    client: MagicMock, model_infos: list[ModelInfo], endpoints: list[PublicEndpoint]
) -> None:
    with patch.object(client.endpoints, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_endpoints_response(endpoints)
        result = await fetch_endpoints(client, model_infos, Filter())
    assert list_mock.call_args_list == [call(author="openai", slug="gpt-4o")]
    assert [m.provider_id for m in result] == ["openai", "azure"]
    assert all(m.model_id == "openai/gpt-4o" for m in result)
    assert all(m.rejection_reasons == set() for m in result)


async def test_fetch_endpoints_input_unmodified(client: MagicMock, model: Model, endpoint: PublicEndpoint) -> None:
    model_info = ModelInfo(model=model)
    with patch.object(client.endpoints, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_endpoints_response([endpoint])
        result = await fetch_endpoints(client, [model_info], Filter())
    assert model_info.endpoint is None
    assert result[0] is not model_info


async def test_fetch_endpoints_rejected_models(client: MagicMock, model: Model) -> None:
    model_info = ModelInfo(model=model, rejection_reasons={Column.MODEL_ID})
    with patch.object(client.endpoints, "list_async", new_callable=AsyncMock) as list_mock:
        result = await fetch_endpoints(client, [model_info], Filter())
    assert list_mock.call_args_list == []
    assert result == [model_info]


async def test_fetch_endpoints_endpoint_filters(
    client: MagicMock, model_infos: list[ModelInfo], endpoints: list[PublicEndpoint]
) -> None:
    with patch.object(client.endpoints, "list_async", new_callable=AsyncMock) as list_mock:
        list_mock.return_value = create_endpoints_response(endpoints)
        result = await fetch_endpoints(client, model_infos, Filter(include_providers=r"^openai$"))
    assert {m.provider_id: m.rejection_reasons for m in result} == {
        "azure": {Column.PROVIDER_ID},
        "openai": set(),
    }


async def test_fetch_endpoints_zdr(
    client: MagicMock, model_infos: list[ModelInfo], zdr_endpoints: list[PublicZdrEndpoint]
) -> None:
    with (
        patch.object(client.endpoints, "list_zdr_endpoints_async", new_callable=AsyncMock) as zdr_mock,
        patch.object(client.endpoints, "list_async", new_callable=AsyncMock) as list_mock,
    ):
        zdr_mock.return_value = ListEndpointsZdrResponse(data=zdr_endpoints)
        result = await fetch_endpoints(client, model_infos, Filter(require_zdr=True))
    assert zdr_mock.call_args_list == [call()]
    assert list_mock.call_args_list == []
    assert [m.provider_id for m in result] == ["openai", "azure"]
    assert all(m.model_id == "openai/gpt-4o" for m in result)


async def test_fetch_endpoints_zdr_unknown_models(
    client: MagicMock, model_infos: list[ModelInfo], zdr_endpoint: PublicZdrEndpoint
) -> None:
    zdr_endpoints = [zdr_endpoint, zdr_endpoint.model_copy(update={"model_id": "unknown/model"})]
    with patch.object(client.endpoints, "list_zdr_endpoints_async", new_callable=AsyncMock) as zdr_mock:
        zdr_mock.return_value = ListEndpointsZdrResponse(data=zdr_endpoints)
        result = await fetch_endpoints(client, model_infos, Filter(require_zdr=True))
    assert [m.model_id for m in result] == ["openai/gpt-4o"]


async def test_fetch_endpoints_zdr_endpoint_filters(
    client: MagicMock, model_infos: list[ModelInfo], zdr_endpoints: list[PublicZdrEndpoint]
) -> None:
    with patch.object(client.endpoints, "list_zdr_endpoints_async", new_callable=AsyncMock) as zdr_mock:
        zdr_mock.return_value = ListEndpointsZdrResponse(data=zdr_endpoints)
        result = await fetch_endpoints(client, model_infos, Filter(require_zdr=True, include_providers=r"^openai$"))
    assert {m.provider_id: m.rejection_reasons for m in result} == {
        "azure": {Column.PROVIDER_ID},
        "openai": set(),
    }


@pytest.mark.parametrize(
    ("string", "expected_model_id"),
    [
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("openai/gpt-4o-canonical", "openai/gpt-4o"),
        ("GPT-4o", "openai/gpt-4o"),
        ("unknown", None),
    ],
)
def test_search_model_info_by_id_canonical_slug_or_name(
    model: Model, string: str, expected_model_id: str | None
) -> None:
    model_infos = [ModelInfo(model=model.model_copy(update={"canonical_slug": "openai/gpt-4o-canonical"}))]
    model_info = _search_model_info(model_infos, string)
    assert (model_info.model_id if model_info else None) == expected_model_id


def test_search_model_info_first_match(model: Model) -> None:
    model_infos = [ModelInfo(model=model), ModelInfo(model=model.model_copy(update={"id": "openai/gpt-4o-2"}))]
    model_info = _search_model_info(model_infos, "GPT-4o")
    assert model_info
    assert model_info.model_id == "openai/gpt-4o"


def test_search_model_info_presets_excluded(model: Model) -> None:
    model_infos = [ModelInfo(model=model.model_copy(update={"id": "@preset/fav", "name": "@preset/fav"}))]
    assert _search_model_info(model_infos, "@preset/fav") is None


async def test_save_as_preset_with_provider_id(client: MagicMock, model_info: ModelInfo) -> None:
    with patch.object(client.presets, "create_presets_chat_completions_async", new_callable=AsyncMock) as create_mock:
        await save_as_preset(client, model_info, "fav1")
    assert create_mock.call_args_list == [
        call(
            slug="fav1",
            messages=[],
            model="openai/gpt-4o",
            provider=ProviderPreferences(allow_fallbacks=False, only=["openai"]),
        )
    ]


async def test_save_as_preset_without_provider_id(client: MagicMock, model: Model) -> None:
    with (
        patch.object(client.presets, "create_presets_chat_completions_async", new_callable=AsyncMock) as create_mock,
        pytest.raises(ValueError, match=r"^provider ID not set$"),
    ):
        await save_as_preset(client, ModelInfo(model=model), "fav1")
    assert create_mock.call_args_list == []

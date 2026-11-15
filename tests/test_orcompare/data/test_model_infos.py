# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, call, patch

import pytest
from openrouter.components import ActivityItem, ActivityResponse, ModelBenchmarks

from orcompare.data.model_infos import MixedPriceWeights, ModelInfo

if TYPE_CHECKING:
    from openrouter import OpenRouter


def create_activity_item(*, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> ActivityItem:
    return ActivityItem(
        byok_usage_inference=0,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        date_="2026-01-01",
        endpoint_id="endpoint",
        model="openai/gpt-4o",
        model_permaslug="openai/gpt-4o",
        prompt_tokens=prompt_tokens,
        provider_name="OpenAI",
        reasoning_tokens=0,
        requests=1,
        usage=0,
    )


class TestMixedPriceWeights:
    @staticmethod
    async def test_from_openrouter_with_activity(client: OpenRouter) -> None:
        weights = MixedPriceWeights()
        response = ActivityResponse(
            data=[
                create_activity_item(prompt_tokens=1000, cached_tokens=100, completion_tokens=50),
                create_activity_item(prompt_tokens=500, cached_tokens=0, completion_tokens=25),
            ]
        )
        with patch.object(
            client.analytics, "get_user_activity_async", new_callable=AsyncMock
        ) as get_user_activity_mock:
            get_user_activity_mock.return_value = response
            await weights.from_openrouter(client)
        assert get_user_activity_mock.call_args_list == [call()]
        assert weights.input_tokens == 1400
        assert weights.output_tokens == 75
        assert weights.cached_input_tokens == 100

    @staticmethod
    async def test_from_openrouter_no_activity(client: OpenRouter) -> None:
        weights = MixedPriceWeights()
        with patch.object(
            client.analytics, "get_user_activity_async", new_callable=AsyncMock
        ) as get_user_activity_mock:
            get_user_activity_mock.return_value = ActivityResponse(data=[])
            await weights.from_openrouter(client)
        assert get_user_activity_mock.call_args_list == [call()]
        assert weights.input_tokens == 1
        assert weights.output_tokens == pytest.approx(0.15)
        assert weights.cached_input_tokens == 20


class TestModelInfo:
    @staticmethod
    def test_age(model_info: ModelInfo) -> None:
        assert model_info.age == pytest.approx(100, abs=0.01)

    @staticmethod
    def test_cached_input_price(model_info: ModelInfo) -> None:
        assert model_info.cached_input_price == pytest.approx(1.25)

    @staticmethod
    def test_cached_input_price_without_cache_read(model_info: ModelInfo) -> None:
        assert model_info.endpoint
        model_info.endpoint.pricing.input_cache_read = None
        assert model_info.cached_input_price == pytest.approx(2.5)

    @staticmethod
    def test_cached_input_price_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.cached_input_price is None

    @staticmethod
    def test_context_size(model_info: ModelInfo) -> None:
        assert model_info.context_size == 128_000

    @staticmethod
    def test_context_size_none(model_info: ModelInfo) -> None:
        model_info.model.context_length = None
        assert model_info.context_size is None

    @staticmethod
    def test_discount(model_info: ModelInfo) -> None:
        assert model_info.discount == pytest.approx(25.0)

    @staticmethod
    def test_discount_none(model_info: ModelInfo) -> None:
        assert model_info.endpoint
        model_info.endpoint.pricing.discount = None
        assert model_info.discount is None

    @staticmethod
    def test_discount_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.discount is None

    @staticmethod
    def test_input_price(model_info: ModelInfo) -> None:
        assert model_info.input_price == pytest.approx(2.5)

    @staticmethod
    def test_input_price_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.input_price is None

    @staticmethod
    def test_latency(model_info: ModelInfo) -> None:
        assert model_info.latency == pytest.approx(0.5)

    @staticmethod
    def test_latency_none(model_info: ModelInfo) -> None:
        assert model_info.endpoint
        model_info.endpoint.latency_last_30m = None
        assert model_info.latency is None

    @staticmethod
    def test_latency_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.latency is None

    @staticmethod
    def test_mixed_price(model_info: ModelInfo) -> None:
        assert model_info.mixed_price == pytest.approx(29.0)

    @staticmethod
    def test_mixed_price_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.mixed_price is None

    @staticmethod
    def test_model_id(model_info: ModelInfo) -> None:
        assert model_info.model_id == "openai/gpt-4o"

    @staticmethod
    def test_model_name(model_info: ModelInfo) -> None:
        assert model_info.model_name == "GPT-4o"

    @staticmethod
    def test_output_price(model_info: ModelInfo) -> None:
        assert model_info.output_price == pytest.approx(10.0)

    @staticmethod
    def test_output_price_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.output_price is None

    @staticmethod
    def test_pricing(model_info: ModelInfo) -> None:
        assert model_info.endpoint
        assert model_info.pricing == model_info.endpoint.pricing

    @staticmethod
    def test_pricing_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.pricing is None

    @staticmethod
    def test_provider_id(model_info: ModelInfo) -> None:
        assert model_info.provider_id == "openai"

    @staticmethod
    def test_provider_id_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.provider_id is None

    @staticmethod
    def test_provider_name(model_info: ModelInfo) -> None:
        assert model_info.provider_name == "OpenAI"

    @staticmethod
    def test_provider_name_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.provider_name is None

    @staticmethod
    def test_quality(model_info: ModelInfo) -> None:
        assert model_info.quality == pytest.approx(42.0)

    @staticmethod
    def test_quality_without_artificial_analysis(model_info: ModelInfo) -> None:
        model_info.model.benchmarks = ModelBenchmarks(design_arena=[])
        assert model_info.quality is None

    @staticmethod
    def test_quality_without_benchmarks(model_info: ModelInfo) -> None:
        model_info.model.benchmarks = None
        assert model_info.quality is None

    @staticmethod
    def test_rejected(model_info: ModelInfo) -> None:
        model_info.rejection_reasons = {"age"}
        assert model_info.rejected

    @staticmethod
    def test_not_rejected(model_info: ModelInfo) -> None:
        assert not model_info.rejected

    @staticmethod
    def test_throughput(model_info: ModelInfo) -> None:
        assert model_info.throughput == pytest.approx(100.0)

    @staticmethod
    def test_throughput_none(model_info: ModelInfo) -> None:
        assert model_info.endpoint
        model_info.endpoint.throughput_last_30m = None
        assert model_info.throughput is None

    @staticmethod
    def test_throughput_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.throughput is None

    @staticmethod
    def test_uptime(model_info: ModelInfo) -> None:
        assert model_info.uptime == pytest.approx(99.0)

    @staticmethod
    def test_uptime_without_endpoint(model_info: ModelInfo) -> None:
        model_info.endpoint = None
        assert model_info.uptime is None

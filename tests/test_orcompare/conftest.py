# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from openrouter.components import (
    AABenchmarkEntry,
    Model,
    ModelArchitecture,
    ModelBenchmarks,
    ModelLinks,
    PercentileStats,
    Pricing,
    PublicEndpoint,
    PublicPricing,
    ToolChoiceSupport,
    TopProviderInfo,
)

from orcompare.data.model_infos import ModelInfo


def pytest_report_teststatus(report: pytest.TestReport) -> tuple[str, str, str] | None:
    # omit dots for passed tests
    return (report.outcome, "", report.outcome.upper()) if report.when == "call" and report.passed else None


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def model() -> Model:
    return Model(
        architecture=ModelArchitecture(input_modalities=[], modality=None, output_modalities=[]),
        canonical_slug="openai/gpt-4o",
        context_length=128_000,
        created=int((datetime.now(tz=UTC) - timedelta(days=100)).timestamp()),
        default_parameters=None,
        endpoints=[],
        id="openai/gpt-4o",
        links=ModelLinks(details=""),
        name="GPT-4o",
        per_request_limits=None,
        pricing=PublicPricing(prompt="0", completion="0"),
        supported_parameters=[],
        supported_voices=None,
        top_provider=TopProviderInfo(is_moderated=False),
        benchmarks=ModelBenchmarks(
            design_arena=[],
            artificial_analysis=AABenchmarkEntry(agentic_index=None, coding_index=None, intelligence_index=42.0),
        ),
    )


@pytest.fixture
def endpoint() -> PublicEndpoint:
    return PublicEndpoint(
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
    )


@pytest.fixture
def model_info(model: Model, endpoint: PublicEndpoint) -> ModelInfo:
    return ModelInfo(model=model, endpoint=endpoint)

# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of available information of models."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

from openrouter import OpenRouter

if TYPE_CHECKING:
    from openrouter.components import Model, Pricing, PublicEndpoint, UnifiedBenchmarksAAItem

_logger = getLogger(__name__)


@dataclass(kw_only=True)
class MixedPriceWeights:
    """Weights to compute mixed prices of models."""

    input_tokens: float = 1
    """Number of uncached input tokens a task typically consumes."""
    output_tokens: float = 0.15
    """Number of output tokens a task typically consumes."""
    cached_input_tokens: float = 20
    """Number of cached input tokens a task typically consumes."""

    async def from_openrouter(self, management_key: str) -> None:
        """Set the values by using the last 30 days of usage data from OpenRouter, if any."""
        _logger.info("getting user activity to estimate mixed prices")
        async with OpenRouter(api_key=management_key) as client:
            response = await client.analytics.get_user_activity_async()
        if not response.data:
            return
        self.input_tokens, self.output_tokens, self.cached_input_tokens = 0, 0, 0
        for activity_item in response.data:
            self.input_tokens += activity_item.prompt_tokens - activity_item.cached_tokens
            self.output_tokens += activity_item.completion_tokens
            self.cached_input_tokens += activity_item.cached_tokens
        _logger.info(
            "user activity: %.1fM input, %.1fM output, %.1fM cached input",
            self.input_tokens / 1e6,
            self.output_tokens / 1e6,
            self.cached_input_tokens / 1e6,
        )


@dataclass
class ModelInfo:
    """Container for model, benchmark, endpoint, and rejection information."""

    model: Model
    """The OpenRouter model."""
    benchmark: UnifiedBenchmarksAAItem | None = None
    """Benchmark data for the model, if available."""
    endpoint: PublicEndpoint | None = None
    """Endpoint data for the model, if available."""
    rejection_reasons: set[str] = field(default_factory=set)
    """Set of rejection reason strings."""

    mixed_price_weights: ClassVar[MixedPriceWeights] = MixedPriceWeights()
    """Weights to compute the mixed price (input tokens will be normalized to 1)."""

    @property
    def age(self) -> float:
        """Model age in days."""
        return (datetime.now(tz=UTC) - datetime.fromtimestamp(self.model.created, tz=UTC)) / timedelta(days=1)

    @property
    def cached_input_price(self) -> float | None:
        """Price in dollars per million cached input tokens."""
        return 1e6 * float(x) if (p := self.pricing) and (x := p.input_cache_read) is not None else self.input_price

    @property
    def context_size(self) -> float | None:
        """Context size in tokens."""
        return self.model.context_length

    @property
    def discount(self) -> float | None:
        """Discount percentage."""
        return 100 * x if (p := self.pricing) and (x := p.discount) is not None else None

    @property
    def input_price(self) -> float | None:
        """Price in dollars per million uncached input tokens."""
        return 1e6 * float(p.prompt) if (p := self.pricing) else None

    @property
    def latency(self) -> float | None:
        """Median latency in seconds."""
        return la.p50 if (e := self.endpoint) and (la := e.latency_last_30m) is not None else None

    @property
    def mixed_price(self) -> float | None:
        """Mixed price (weighted mean of token prices) in dollars per million input tokens."""
        weights = ModelInfo.mixed_price_weights
        return (
            (
                i
                + weights.output_tokens / weights.input_tokens * o
                + weights.cached_input_tokens / weights.input_tokens * c
            )
            if (i := self.input_price) is not None
            and (o := self.output_price) is not None
            and (c := self.cached_input_price) is not None
            else None
        )

    @property
    def model_id(self) -> str:
        """Model identifier string."""
        return self.model.id

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return self.model.name

    @property
    def output_price(self) -> float | None:
        """Price in dollars per million output tokens."""
        return 1e6 * float(p.completion) if (p := self.pricing) else None

    @property
    def pricing(self) -> Pricing | None:
        """Pricing information from the endpoint, if available."""
        return e.pricing if (e := self.endpoint) else None

    @property
    def provider_id(self) -> str | None:
        """Provider identifier string."""
        return e.tag if (e := self.endpoint) else None

    @property
    def provider_name(self) -> str | None:
        """Provider name string."""
        return e.provider_name if (e := self.endpoint) else None

    @property
    def quality(self) -> float | None:
        """Quality score in percent."""
        return b.intelligence_index if (b := self.benchmark) else None

    @property
    def rejected(self) -> bool:
        """Whether the model has any rejection reasons."""
        return bool(self.rejection_reasons)

    @property
    def throughput(self) -> float | None:
        """Median throughput in tokens per second."""
        return t.p50 if (e := self.endpoint) and (t := e.throughput_last_30m) is not None else None

    @property
    def uptime(self) -> float | None:
        """Uptime percentage."""
        return e.uptime_last_30m if (e := self.endpoint) else None

# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Column definitions for comparing OpenRouter models."""

import builtins
from enum import StrEnum, auto
from inspect import signature
from types import NoneType, UnionType
from typing import override

from .model_infos import ModelInfo


class Column(StrEnum):
    """Enumeration of table columns available for display and sorting."""

    AGE = auto()
    """Model age in days."""
    CACHED_INPUT_PRICE = auto()
    """Cached input price in dollars per million tokens."""
    CONTEXT_SIZE = auto()
    """Context size in tokens."""
    DISCOUNT = auto()
    """Discount percentage."""
    INPUT_PRICE = auto()
    """Input price in dollars per million tokens."""
    LATENCY = auto()
    """Median latency in seconds."""
    MIXED_PRICE = auto()
    """Mixed price in dollars per million tokens."""
    MODEL_ID = auto()
    """Model identifier string."""
    MODEL_NAME = auto()
    """Human-readable model name."""
    OUTPUT_PRICE = auto()
    """Output price in dollars per million tokens."""
    PROVIDER_ID = auto()
    """Provider identifier string."""
    PROVIDER_NAME = auto()
    """Provider name string."""
    QUALITY = auto()
    """Quality score in percent."""
    REJECTION_REASONS = auto()
    """Comma-separated rejection reasons."""
    THROUGHPUT = auto()
    """Median throughput in tokens per second."""
    UPTIME = auto()
    """Uptime percentage."""

    @override
    def __repr__(self) -> str:
        return self.value

    @override
    def __str__(self) -> str:
        return self.value

    @property
    def header(self) -> str:
        """Short header label for the column."""
        return {
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
        }[self]

    @property
    def left_aligned(self) -> bool:
        """Whether the column is left-aligned in the table."""
        return self in {
            Column.MODEL_ID,
            Column.MODEL_NAME,
            Column.PROVIDER_ID,
            Column.PROVIDER_NAME,
            Column.REJECTION_REASONS,
        }

    def get_value(self, model_info: ModelInfo) -> float | str | None:
        """Get the value of this specific column.

        :param model_info: Model information to read the value from.
        :return: Column value, or ``None`` if not available.
        """
        if self == Column.REJECTION_REASONS:
            return ",".join(sorted(model_info.rejection_reasons))
        return getattr(model_info, self.value)

    def format_value(self, model_info: ModelInfo) -> str:
        """Format the value of this column for display.

        :param model_info: Model information to read the value from.
        :return: Formatted string representation.
        """
        value = self.get_value(model_info)
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        match self:
            case Column.AGE | Column.DISCOUNT | Column.QUALITY | Column.THROUGHPUT | Column.UPTIME:
                result = f"{value:.0f}"
            case Column.CACHED_INPUT_PRICE:
                result = f"{100 * value:.2f}"
            case Column.CONTEXT_SIZE:
                # ruff: ignore[magic-value-comparison]
                result = f"{value / 1e3:.0f}K" if value < 1e6 else f"{value / 1e6:.1f}M"
            case Column.INPUT_PRICE | Column.OUTPUT_PRICE | Column.MIXED_PRICE:
                result = f"{100 * value:.0f}"
            case Column.LATENCY:
                result = f"{value:.1f}"
            case _:
                msg = f"unsupported column: {self.value}"
                raise ValueError(msg)
        return result

    def key_function(self, model_info: ModelInfo) -> float | str:
        """Return the sort key for this column.

        :param model_info: Model information to extract the key from.
        :return: Sort key value.
        """
        value = self.get_value(model_info)
        if value is None:
            signature_ = signature(getattr(ModelInfo, self.value).fget)
            value_type = signature_.return_annotation
            if isinstance(value_type, UnionType):
                number_of_expected_union_arguments = 2
                if len(value_type.__args__) != number_of_expected_union_arguments:
                    msg = (
                        f"expected two Union arguments in return type of {ModelInfo.__name__}.{self.value}, got "
                        f"{len(value_type.__args__)}"
                    )
                    raise TypeError(msg)
                if value_type.__args__[1] is not NoneType:
                    msg = (
                        f"expected second Union argument in return type of {ModelInfo.__name__}.{self.value} to be "
                        f"None, got {value_type.__args__[1].__name__}"
                    )
                    raise TypeError(msg)
                value_type = value_type.__args__[0]
            match value_type:
                case builtins.float:
                    value = 0
                case builtins.str:
                    value = ""
                case _:
                    msg = (
                        f"expected return type of {ModelInfo.__name__}.{self.value} to be either float or str, got "
                        f"{value_type.__name__}"
                    )
                    raise TypeError(msg)
        return value

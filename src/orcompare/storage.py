# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Storage utilities for API keys and persistent data."""

from getpass import getpass
from logging import getLogger
from os import environ
from pathlib import Path
from typing import Literal

import orcompare

_logger = getLogger(__name__)


def get_storage_dir() -> Path:
    """Return the storage directory path.

    :return: Path to the storage directory.
    """
    return Path(orcompare.__file__).with_name("storage")


def get_api_key() -> str:
    """Read or prompt for the OpenRouter API key.

    Prefers the ``OPENROUTER_API_KEY`` environment variable if set and not empty. Otherwise, looks for a file named
    ``api_key`` in the storage directory. If it doesn't exist, prompts for the API key and saves it in the file.

    :return: The API key string.
    """
    return _get_key("API")


def get_management_key() -> str:
    """Read or prompt for the OpenRouter management key.

    Prefers the ``OPENROUTER_MANAGEMENT_KEY`` environment variable if set and not empty. Otherwise, looks for a file
    named ``management_key`` in the storage directory. If it doesn't exist, prompts for the management key and saves it
    in the file.

    :return: The management key string.
    """
    return _get_key("management")


def _get_key(name: Literal["API", "management"]) -> str:
    """Read or prompt for an OpenRouter key.

    :return: The key string.
    """
    if key := environ.get(f"OPENROUTER_{name.upper()}_KEY"):
        return key
    path = get_storage_dir() / f"{name.lower()}_key"
    if path.exists():
        key = path.read_text().strip()
        if key:
            _logger.info("using OpenRouter %s key from %s", name, path)
            return key
    path.parent.mkdir(exist_ok=True)
    path.touch()
    path.chmod(0o600)
    key = getpass(f"Enter OpenRouter {name} key: ")
    path.write_text(key)
    _logger.info("OpenRouter %s key saved to %s", name, path)
    return key

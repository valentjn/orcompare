# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path
from typing import Literal
from unittest.mock import call, patch

import pytest

import orcompare.storage
from orcompare.storage import _get_key, get_api_key, get_management_key, get_storage_dir

type KeyName = Literal["API", "management"]


@pytest.fixture(params=["API", "management"])
def key_name(request: pytest.FixtureRequest) -> KeyName:
    return request.param


@pytest.fixture
def key_path(key_name: KeyName, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(f"OPENROUTER_{key_name.upper()}_KEY", raising=False)
    return tmp_path / f"{key_name.lower()}_key"


def test_get_storage_dir() -> None:
    storage_dir = get_storage_dir()
    assert storage_dir.name == "storage"
    assert storage_dir.parent == Path(orcompare.storage.__file__).parent


def test_get_api_key() -> None:
    with patch.object(orcompare.storage, "_get_key", return_value="key") as get_key_mock:
        assert get_api_key() == "key"
    assert get_key_mock.call_args_list == [call("API")]


def test_get_management_key() -> None:
    with patch.object(orcompare.storage, "_get_key", return_value="key") as get_key_mock:
        assert get_management_key() == "key"
    assert get_key_mock.call_args_list == [call("management")]


def test_get_key_environment_variable(key_name: KeyName, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"OPENROUTER_{key_name.upper()}_KEY", "env-key")
    with patch.object(orcompare.storage, "get_storage_dir") as get_storage_dir_mock:
        assert _get_key(key_name) == "env-key"
    assert get_storage_dir_mock.call_args_list == []


def test_get_key_empty_environment_variable(key_name: KeyName, key_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"OPENROUTER_{key_name.upper()}_KEY", "")
    key_path.write_text("file-key")
    with patch.object(orcompare.storage, "get_storage_dir", return_value=key_path.parent):
        assert _get_key(key_name) == "file-key"


def test_get_key_file(key_name: KeyName, key_path: Path) -> None:
    key_path.write_text("file-key\n")
    with (
        patch.object(orcompare.storage, "get_storage_dir", return_value=key_path.parent),
        patch.object(orcompare.storage, "getpass") as getpass_mock,
    ):
        assert _get_key(key_name) == "file-key"
    assert getpass_mock.call_args_list == []


def test_get_key_empty_file(key_name: KeyName, key_path: Path) -> None:
    key_path.write_text("")
    with (
        patch.object(orcompare.storage, "get_storage_dir", return_value=key_path.parent),
        patch.object(orcompare.storage, "getpass", return_value="prompted-key") as getpass_mock,
    ):
        assert _get_key(key_name) == "prompted-key"
    assert getpass_mock.call_args_list == [call(f"Enter OpenRouter {key_name} key: ")]
    assert key_path.read_text() == "prompted-key"


def test_get_key_missing_file(key_name: KeyName, key_path: Path) -> None:
    storage_dir = key_path.parent / "storage"
    with (
        patch.object(orcompare.storage, "get_storage_dir", return_value=storage_dir),
        patch.object(orcompare.storage, "getpass", return_value="prompted-key") as getpass_mock,
    ):
        assert _get_key(key_name) == "prompted-key"
    assert getpass_mock.call_args_list == [call(f"Enter OpenRouter {key_name} key: ")]
    assert (storage_dir / key_path.name).read_text() == "prompted-key"

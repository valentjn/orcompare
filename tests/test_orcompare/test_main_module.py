# Copyright (C) 2026 Julian Valentin
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from runpy import run_module
from unittest.mock import patch

import orcompare
import orcompare.main


def test_main_module() -> None:
    with patch.object(orcompare.main, "main") as main_mock:
        run_module(orcompare.__name__, run_name="__main__")
    main_mock.assert_called_once()

# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Mocker:

    MagicMock = mock.MagicMock

    def __init__(self) -> None:
        self._patches: list[mock._patch] = []  

    def patch(self, *args, **kwargs): 
        patcher = mock.patch(*args, **kwargs)
        obj = patcher.start()
        self._patches.append(patcher)
        return obj

    def stopall(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()
        self._patches.clear()


@pytest.fixture
def mocker() -> Generator[_Mocker, None, None]:
    m = _Mocker()
    try:
        yield m
    finally:
        m.stopall()


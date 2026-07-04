import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.tools import BUILTIN_TOOLS
from src.agent.tools.service import menu_tool


def test_menu_tool_registered():
    assert menu_tool in BUILTIN_TOOLS

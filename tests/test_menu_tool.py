import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.tools import BUILTIN_TOOLS
from src.agent.tools.service import menu_tool


def test_menu_tool_registered():
    assert menu_tool in BUILTIN_TOOLS


def _stub_browser(script_result):
    from src.agent.browser.service import Browser
    browser = Browser.__new__(Browser)
    browser.execute_script = AsyncMock(return_value=script_result)
    return browser


def test_select_option_embeds_labels_via_json_dumps():
    browser = _stub_browser({'selected': [], 'notFound': ["Mechanic's Lien"], 'available': ['Deed of Trust']})
    asyncio.run(browser.select_option("//select[1]", ["Mechanic's Lien"]))
    script = browser.execute_script.call_args.args[0]
    assert json.dumps(["Mechanic's Lien"]) in script


def test_select_option_returns_structured_result():
    expected = {'selected': ['BMW'], 'notFound': [], 'available': ['BMW', 'Audi']}
    browser = _stub_browser(expected)
    result = asyncio.run(browser.select_option("//select[1]", ['BMW']))
    assert result == expected


def test_select_option_script_handles_error_shapes():
    browser = _stub_browser({'error': 'not_found'})
    asyncio.run(browser.select_option("//select[1]", ['BMW']))
    script = browser.execute_script.call_args.args[0]
    assert 'not_found' in script
    assert 'not_select' in script
    assert 'available' in script

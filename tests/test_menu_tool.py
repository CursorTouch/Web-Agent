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


def _stub_session(select_result):
    element = MagicMock()
    element.xpath = {'element': '//select[1]'}
    session = MagicMock()
    session.get_element_by_index = AsyncMock(return_value=element)
    session.select_option = AsyncMock(return_value=select_result)
    return session


def test_menu_tool_success_message():
    session = _stub_session({'selected': ["Mechanic's Lien"], 'notFound': [], 'available': ["Mechanic's Lien", 'Deed of Trust']})
    message = asyncio.run(menu_tool.ainvoke(index=3, labels=["Mechanic's Lien"], session=session))
    assert "Mechanic's Lien" in message
    assert 'label 3' in message


def test_menu_tool_raises_with_available_options_on_mismatch():
    session = _stub_session({'selected': [], 'notFound': ["Mechanic's Lien"], 'available': ['Mechanic’s Lien', 'Deed of Trust']})
    with pytest.raises(Exception) as exc_info:
        asyncio.run(menu_tool.ainvoke(index=3, labels=["Mechanic's Lien"], session=session))
    assert 'Mechanic’s Lien' in str(exc_info.value)
    assert 'Available options' in str(exc_info.value)


def test_menu_tool_raises_on_non_select_element():
    session = _stub_session({'error': 'not_select', 'tag': 'div'})
    with pytest.raises(Exception) as exc_info:
        asyncio.run(menu_tool.ainvoke(index=5, labels=['BMW'], session=session))
    assert 'not a <select>' in str(exc_info.value)
    assert 'click_tool' in str(exc_info.value)


def test_menu_tool_raises_on_element_not_found():
    session = _stub_session({'error': 'not_found'})
    with pytest.raises(Exception) as exc_info:
        asyncio.run(menu_tool.ainvoke(index=7, labels=['BMW'], session=session))
    assert 'label 7' in str(exc_info.value)

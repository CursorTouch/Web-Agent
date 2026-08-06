from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.browser.config import BROWSER_ARGS, BrowserConfig, detect_installed_browser
from src.agent.browser.service import Browser


def test_force_device_scale_factor_removed():
    assert '--force-device-scale-factor=1' not in BROWSER_ARGS


def test_disable_sync_present():
    assert '--disable-sync' in BROWSER_ARGS


def test_attach_to_existing_default_false():
    assert BrowserConfig().attach_to_existing is False


def test_attach_to_existing_can_be_enabled():
    assert BrowserConfig(attach_to_existing=True).attach_to_existing is True


def test_resolved_browser_explicit():
    assert BrowserConfig(browser='chrome').resolved_browser() == 'chrome'
    assert BrowserConfig(browser='edge').resolved_browser() == 'edge'


def test_detect_linux_uses_shutil_which():
    with patch('src.agent.browser.config.platform.system', return_value='Linux'), \
         patch('shutil.which', side_effect=lambda cmd: '/usr/bin/google-chrome' if cmd == 'google-chrome' else None):
        assert detect_installed_browser() == 'chrome'


def test_detect_linux_falls_back_to_edge():
    with patch('src.agent.browser.config.platform.system', return_value='Linux'), \
         patch('shutil.which', side_effect=lambda cmd: '/usr/bin/msedge' if cmd == 'microsoft-edge' else None):
        assert detect_installed_browser() == 'edge'


def test_read_devtools_active_port_valid(tmp_path):
    port_file = tmp_path / 'DevToolsActivePort'
    port_file.write_text('9222\n/devtools/browser/abc-123\n')

    browser = Browser(BrowserConfig(user_data_dir=str(tmp_path)))

    assert browser._read_devtools_active_port() == 'ws://127.0.0.1:9222/devtools/browser/abc-123'


def test_read_devtools_active_port_no_user_data_dir():
    browser = Browser(BrowserConfig(user_data_dir=None))
    assert browser._read_devtools_active_port() is None


def test_read_devtools_active_port_malformed(tmp_path):
    port_file = tmp_path / 'DevToolsActivePort'
    port_file.write_text('notaport\n')

    browser = Browser(BrowserConfig(user_data_dir=str(tmp_path)))

    assert browser._read_devtools_active_port() is None


@pytest.mark.asyncio
async def test_resolve_ws_url_attach_uses_devtools_active_port(tmp_path):
    port_file = tmp_path / 'DevToolsActivePort'
    port_file.write_text('9222\n/devtools/browser/abc\n')

    browser = Browser(BrowserConfig(attach_to_existing=True, user_data_dir=str(tmp_path)))
    with patch.object(browser, '_is_port_responsive', new=AsyncMock()) as mock_poll:
        await browser._resolve_ws_url()

    assert browser._resolved_attach_ws_url == 'ws://127.0.0.1:9222/devtools/browser/abc'
    mock_poll.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_ws_url_attach_fallback_to_port_polling(tmp_path):
    browser = Browser(BrowserConfig(attach_to_existing=True, user_data_dir=str(tmp_path)))

    with patch.object(browser, '_is_port_responsive', new=AsyncMock(return_value=True)):
        await browser._resolve_ws_url()

    assert browser._resolved_attach_ws_url is None


@pytest.mark.asyncio
async def test_resolve_ws_url_attach_raises_when_port_dead(tmp_path):
    browser = Browser(BrowserConfig(attach_to_existing=True, user_data_dir=str(tmp_path)))

    with patch.object(browser, '_is_port_responsive', new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match='attach_to_existing'):
            await browser._resolve_ws_url()


@pytest.mark.asyncio
async def test_init_browser_attach_uses_resolved_url(tmp_path):
    port_file = tmp_path / 'DevToolsActivePort'
    port_file.write_text('9222\n/devtools/browser/abc\n')

    browser = Browser(BrowserConfig(attach_to_existing=True, user_data_dir=str(tmp_path)))

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)

    with patch('src.agent.browser.service.Client', return_value=mock_client), \
         patch.object(browser, '_launch_process') as mock_launch:
        await browser.init_browser()

    mock_launch.assert_not_called()
    assert browser._client is mock_client


@pytest.mark.asyncio
async def test_init_browser_attach_never_kills(tmp_path):
    port_file = tmp_path / 'DevToolsActivePort'
    port_file.write_text('9222\n/devtools/browser/abc\n')

    browser = Browser(BrowserConfig(attach_to_existing=True, user_data_dir=str(tmp_path)))

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)

    with patch('src.agent.browser.service.Client', return_value=mock_client), \
         patch.object(browser, '_kill_on_port') as mock_kill:
        await browser.init_browser()

    mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_close_browser_attach_does_not_terminate_process():
    browser = Browser(BrowserConfig(attach_to_existing=True))
    browser._client = MagicMock()
    browser._client.__aexit__ = AsyncMock(return_value=None)
    process = MagicMock()
    browser._process = process

    await browser.close_browser()

    process.terminate.assert_not_called()
    process.kill.assert_not_called()

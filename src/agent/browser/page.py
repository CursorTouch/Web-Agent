from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from pathlib import Path
from typing import Any

from src.agent.browser.events import StateInvalidatedEvent

logger = logging.getLogger(__name__)


class Page:
    """Page-level operations for the current browser target/session."""

    def __init__(self, browser) -> None:
        self.browser = browser

    async def execute_script(self, script: str, truncate: bool = False, repair: bool = False) -> Any:
        sid = self.browser._get_current_session_id()
        if repair:
            script = self.browser._repair_js(script)
        try:
            result = await self.browser.send('Runtime.evaluate', {
                'expression': script, 'returnByValue': True, 'awaitPromise': True,
            }, session_id=sid)
            if result and 'result' in result:
                value = result['result'].get('value')
                if truncate and isinstance(value, str) and len(value) > 20_000:
                    value = value[:20_000] + f'\n... [truncated, total length: {len(value)}]'
                return value
        except Exception as e:
            logger.warning('execute_script error: %s', e)
        return None

    async def get_screenshot(self, full_page: bool = False, save_screenshot: bool = False) -> bytes | None:
        sid = self.browser._get_current_session_id()
        await asyncio.sleep(0.3)
        try:
            result = await self.browser.send('Page.captureScreenshot', {
                'format': 'jpeg', 'quality': 80, 'captureBeyondViewport': full_page,
            }, session_id=sid)
            data = base64.b64decode(result['data'])
        except Exception as e:
            logger.warning('Screenshot failed: %s', e)
            return None

        if save_screenshot:
            from datetime import datetime

            folder_path = Path('./screenshots')
            folder_path.mkdir(parents=True, exist_ok=True)
            path = folder_path / f'screenshot_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.jpeg'
            with open(path, 'wb') as f:
                f.write(data)

        if self.browser.hooks and data:
            await self.browser.hooks.on_screenshot(data=data, browser=self.browser)

        return data

    async def get_page_content(self) -> str:
        return await self.execute_script('document.documentElement.outerHTML') or ''

    async def get_viewport(self) -> tuple[int, int]:
        result = await self.execute_script('({width: window.innerWidth, height: window.innerHeight})')
        if isinstance(result, dict):
            return result.get('width', 1280), result.get('height', 720)
        return 1280, 720

    async def click_at(self, x: int, y: int) -> None:
        sid = self.browser._get_current_session_id()
        self.browser.emit_browser_event(StateInvalidatedEvent(session_id=sid, reason='click'))
        jx = x + random.randint(-3, 3)
        jy = y + random.randint(-3, 3)
        await self.browser._move_mouse(jx, jy)
        await asyncio.sleep(random.uniform(0.02, 0.06))
        await self.browser.send('Input.dispatchMouseEvent', {
            'type': 'mousePressed', 'x': jx, 'y': jy, 'button': 'left', 'clickCount': 1,
        }, session_id=sid)
        try:
            await self.execute_script('window.__wu_click_cursor__ && window.__wu_click_cursor__()')
        except Exception:
            pass
        await asyncio.sleep(random.uniform(0.06, 0.18))
        await self.browser.send('Input.dispatchMouseEvent', {
            'type': 'mouseReleased', 'x': jx, 'y': jy, 'button': 'left', 'clickCount': 1,
        }, session_id=sid)
        if self.browser.hooks:
            await self.browser.hooks.on_click(x=jx, y=jy, browser=self.browser)

    async def type_text(self, text: str, delay_ms: int = 50) -> None:
        sid = self.browser._get_current_session_id()
        self.browser.emit_browser_event(StateInvalidatedEvent(session_id=sid, reason='type'))
        for char in text:
            key_code = ord(char.upper()) if char.isascii() and char.isprintable() else 0
            code = f'Key{char.upper()}' if char.isalpha() else (
                'Space' if char == ' ' else
                'Enter' if char == '\n' else
                'Period' if char == '.' else
                'Comma' if char == ',' else ''
            )
            key = char if char != '\n' else 'Enter'
            # keyDown/keyUp carry the key identity only — no text field, so no character insertion
            # the char event is the sole source of character input
            await self.browser.send('Input.dispatchKeyEvent', {
                'type': 'keyDown', 'key': key, 'code': code,
                'windowsVirtualKeyCode': key_code, 'nativeVirtualKeyCode': key_code,
            }, session_id=sid)
            await self.browser.send('Input.dispatchKeyEvent', {
                'type': 'char', 'text': char,
            }, session_id=sid)
            await self.browser.send('Input.dispatchKeyEvent', {
                'type': 'keyUp', 'key': key, 'code': code,
                'windowsVirtualKeyCode': key_code, 'nativeVirtualKeyCode': key_code,
            }, session_id=sid)

            if char == ' ':
                delay = random.uniform(0.04, 0.09)
            elif char in '.,!?;:\n':
                delay = random.uniform(0.06, 0.14)
            elif char.isupper():
                delay = random.uniform(0.03, 0.07)
            else:
                delay = random.uniform(0.02, 0.06)
            # occasional burst pause (simulates thinking mid-word)
            if random.random() < 0.04:
                delay += random.uniform(0.1, 0.3)
            await asyncio.sleep(delay)
        if self.browser.hooks:
            await self.browser.hooks.on_type(text=text, browser=self.browser)

    async def key_press(self, keys: str) -> None:
        sid = self.browser._get_current_session_id()
        self.browser.emit_browser_event(StateInvalidatedEvent(session_id=sid, reason='key_press'))
        mods, key_name = self.browser._parse_key_combo_impl(keys)
        combined = sum(m['bit'] for m in mods)

        key_def = self.browser._special_keys.get(key_name)
        if key_def is None:
            if len(key_name) == 1:
                key_def = {'key': key_name, 'code': f'Key{key_name.upper()}', 'keyCode': ord(key_name.upper())}
            else:
                key_def = {'key': key_name, 'code': key_name, 'keyCode': 0}

        for mod in mods:
            await self.browser.send('Input.dispatchKeyEvent', {
                'type': 'rawKeyDown', 'key': mod['key'], 'code': mod['code'],
                'windowsVirtualKeyCode': mod['keyCode'], 'modifiers': combined,
            }, session_id=sid)

        await self.browser.send('Input.dispatchKeyEvent', {
            'type': 'rawKeyDown', 'key': key_def['key'], 'code': key_def['code'],
            'windowsVirtualKeyCode': key_def.get('keyCode', 0), 'modifiers': combined,
        }, session_id=sid)
        await self.browser.send('Input.dispatchKeyEvent', {
            'type': 'keyUp', 'key': key_def['key'], 'code': key_def['code'],
            'windowsVirtualKeyCode': key_def.get('keyCode', 0), 'modifiers': combined,
        }, session_id=sid)

        for mod in reversed(mods):
            await self.browser.send('Input.dispatchKeyEvent', {
                'type': 'keyUp', 'key': mod['key'], 'code': mod['code'],
                'windowsVirtualKeyCode': mod['keyCode'], 'modifiers': 0,
            }, session_id=sid)

        if self.browser.hooks:
            await self.browser.hooks.on_key_press(keys=keys, browser=self.browser)

    async def scroll_page(self, direction: str, amount: int = 500) -> None:
        sid = self.browser._get_current_session_id()
        self.browser.emit_browser_event(StateInvalidatedEvent(session_id=sid, reason='scroll_page'))
        viewport = await self.get_viewport()
        cx = viewport[0] // 2
        cy = viewport[1] // 2
        delta = -amount if direction == 'up' else amount
        steps = random.randint(4, 7)
        for i in range(steps):
            # ease-in-out acceleration: slow at edges, fast in middle
            t = (i + 0.5) / steps
            weight = t * t * (3 - 2 * t)
            # distribute delta with a bell curve across steps
            step_frac = (1 / steps) * (0.6 + 0.8 * (1 - abs(weight - 0.5) * 2))
            step_delta = delta * step_frac + random.uniform(-8, 8)
            await self.browser.send('Input.dispatchMouseEvent', {
                'type': 'mouseWheel', 'x': cx, 'y': cy, 'deltaX': 0, 'deltaY': step_delta,
            }, session_id=sid)
            await asyncio.sleep(random.uniform(0.03, 0.09))
        if self.browser.hooks:
            await self.browser.hooks.on_scroll(direction=direction, amount=amount, browser=self.browser)

    async def scroll_at(self, x: int, y: int, direction: str, amount: int = 500) -> None:
        sid = self.browser._get_current_session_id()
        self.browser.emit_browser_event(StateInvalidatedEvent(session_id=sid, reason='scroll_at'))
        delta = -amount if direction == 'up' else amount
        steps = random.randint(4, 7)
        for i in range(steps):
            t = (i + 0.5) / steps
            weight = t * t * (3 - 2 * t)
            step_frac = (1 / steps) * (0.6 + 0.8 * (1 - abs(weight - 0.5) * 2))
            step_delta = delta * step_frac + random.uniform(-8, 8)
            await self.browser.send('Input.dispatchMouseEvent', {
                'type': 'mouseWheel', 'x': x, 'y': y, 'deltaX': 0, 'deltaY': step_delta,
            }, session_id=sid)
            await asyncio.sleep(random.uniform(0.03, 0.09))
        if self.browser.hooks:
            await self.browser.hooks.on_scroll(direction=direction, amount=amount, browser=self.browser)

    async def get_scroll_position(self) -> dict:
        result = await self.execute_script(
            '({scrollY: window.scrollY, scrollHeight: document.documentElement.scrollHeight, innerHeight: window.innerHeight})'
        )
        return result or {'scrollY': 0, 'scrollHeight': 0, 'innerHeight': 0}

    async def set_file_input_at(self, x: int, y: int, files: list[str]) -> None:
        sid = self.browser._get_current_session_id()
        await self.browser.send('DOM.enable', {}, session_id=sid)
        result = await self.browser.send('DOM.getNodeForLocation', {'x': x, 'y': y}, session_id=sid)
        backend_node_id = result.get('backendNodeId')
        if not backend_node_id:
            raise Exception(f'No element found at ({x}, {y})')
        await self.browser.send('DOM.setFileInputFiles', {
            'files': files, 'backendNodeId': backend_node_id,
        }, session_id=sid)

    async def select_option_at(self, x: int, y: int, labels: list[str]) -> dict:
        labels_json = json.dumps(labels)
        return await self.execute_script(
            f'(function(){{'
            f'  var start = document.elementFromPoint({x}, {y});'
            f'  if (!start) return {{error: "not_found"}};'
            f'  var el = start;'
            f'  while (el && el.tagName !== "SELECT") el = el.parentElement;'
            f'  if (!el) return {{error: "not_select", tag: start.tagName.toLowerCase()}};'
            f'  var labels = {labels_json};'
            f'  var texts = Array.from(el.options).map(function(o){{ return o.text.trim(); }});'
            f'  var selected = labels.filter(function(l){{ return texts.indexOf(l) >= 0; }});'
            f'  var notFound = labels.filter(function(l){{ return texts.indexOf(l) < 0; }});'
            f'  if (selected.length) {{'
            f'    if (el.multiple) {{'
            f'      for (var i = 0; i < el.options.length; i++) el.options[i].selected = labels.indexOf(texts[i]) >= 0;'
            f'    }} else {{'
            f'      el.selectedIndex = texts.indexOf(selected[0]);'
            f'    }}'
            f'    el.dispatchEvent(new Event("input", {{bubbles: true}}));'
            f'    el.dispatchEvent(new Event("change", {{bubbles: true}}));'
            f'  }}'
            f'  return {{selected: selected, notFound: notFound, available: texts.slice(0, 30)}};'
            f'}})()'
        )

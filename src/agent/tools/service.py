from src.agent.tools.views import Click, Type, Wait, Scroll, GoTo, Back, Key, Download, Scrape, Tab, Upload, Menu, Done, Forward, HumanInput, Script
from src.agent.browser import Browser
from src.providers.events import LLMEventType
from src.messages import HumanMessage, SystemMessage, ImageMessage
from markdownify import markdownify
from typing import Literal, Optional
from termcolor import colored
from src.tools import Tool
from asyncio import sleep
from pathlib import Path
from os import getcwd
import httpx


@Tool('done_tool', model=Done)
async def done_tool(content: str, session: Browser = None):
    '''Signals task completion. The content must be a comprehensive markdown report — include all findings, extracted data, tables, prices, URLs, and any limitations in full. Never condense or abbreviate results.'''
    return content


@Tool('click_tool', model=Click)
async def click_tool(index: int, session: Browser = None):
    '''Clicks on interactive elements like buttons, links, checkboxes, radio buttons, tabs, or any clickable UI component. Automatically scrolls the element into view if needed.'''
    element = await session.get_element_by_index(index=index)
    print(f'[CLICK] label={index} | tag=<{element.tag}> | role={element.role} | name={element.name!r} | attrs={element.attributes} | xpath={element.xpath.get("element", "")}')
    xpath   = element.xpath.get('element', '')
    if xpath:
        await session.scroll_into_view(xpath)
    await session.click_at(element.center.x, element.center.y)
    await session._wait_for_page(timeout=8.0)
    return f'Clicked on the element at label {index}'


@Tool('type_tool', model=Type)
async def type_tool(index: int, text: str, clear: Literal['True', 'False'] = 'False', press_enter: Literal['True', 'False'] = 'False', session: Browser = None):
    '''Types text into input fields, text areas, search boxes, or any editable element. Can optionally clear existing content before typing.'''
    element = await session.get_element_by_index(index=index)
    print(f'[TYPE] label={index} | tag=<{element.tag}> | role={element.role} | name={element.name!r} | attrs={element.attributes} | xpath={element.xpath.get("element", "")} | text={text!r}')
    xpath   = element.xpath.get('element', '')
    if xpath:
        await session.scroll_into_view(xpath)
    await session.click_at(element.center.x, element.center.y)
    if clear == 'True':
        await session.key_press('Control+a')
        await session.key_press('Backspace')
    await session.type_text(text, delay_ms=50)
    if press_enter == 'True':
        await session.key_press('Enter')
        await session._wait_for_page(timeout=8.0)
    return f'Typed {text} in element at label {index}'


@Tool('wait_tool', model=Wait)
async def wait_tool(time: int, session: Browser = None):
    '''Pauses execution for a specified number of seconds. Use this to wait for page loading, animations to complete, or content to appear after an action.'''
    await sleep(time)
    return f'Waited for {time}s'


@Tool('scroll_tool', model=Scroll)
async def scroll_tool(direction: Literal['up', 'down'] = 'down', index: int = None,amount: int = 500, session: Browser = None):
    '''Scrolls either the webpage or a specific scrollable container. If index is provided, scrolls that element; otherwise scrolls the page.'''
    if index is not None:
        element = await session.get_element_by_index(index=index)
        xpath   = element.xpath.get('element', '')
        await session.scroll_element(xpath, direction, amount)
        return f'Scrolled {direction} inside element at label {index} by {amount}px'

    await session.scroll_page(direction, amount)
    pos        = await session.get_scroll_position()
    scroll_y   = pos.get('scrollY', 0)
    max_scroll = max(pos.get('scrollHeight', 0) - pos.get('innerHeight', 0), 0)
    if max_scroll > 0:
        hint = f' (position: {int(scroll_y)}/{int(max_scroll)}px'
        if scroll_y >= max_scroll:
            hint += ', at bottom)'
        elif scroll_y <= 0:
            hint += ', at top)'
        else:
            hint += ')'
    else:
        hint = ''
    return f'Scrolled {direction} by {amount}px{hint}.'


@Tool('goto_tool', model=GoTo)
async def goto_tool(url: str, session: Browser = None):
    '''Navigates directly to a specified URL in the current tab. Waits for the page to load before proceeding.'''
    await session.navigate(url)
    return f'Navigated to {url}'


@Tool('back_tool', model=Back)
async def back_tool(session: Browser = None):
    '''Navigates to the previous page in browser history, equivalent to clicking the browser Back button.'''
    await session.go_back()
    return 'Navigated to previous page'


@Tool('forward_tool', model=Forward)
async def forward_tool(session: Browser = None):
    '''Navigates to the next page in browser history, equivalent to clicking the browser Forward button.'''
    await session.go_forward()
    return 'Navigated to next page'


@Tool('key_tool', model=Key)
async def key_tool(keys: str, times: int = 1, session: Browser = None):
    '''Performs keyboard shortcuts and key combinations (e.g. "Control+C", "Enter", "Escape", "PageDown"). Can repeat the key press multiple times.'''
    for _ in range(times):
        await session.key_press(keys)
    return f'Pressed {keys}'


@Tool('download_tool', model=Download)
async def download_tool(url: str = None, filename: str = None, session: Browser = None):
    '''Downloads a file from a URL and saves it to the downloads directory.'''
    folder_path = Path(session.config.downloads_dir)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    path = folder_path / filename
    with open(path, 'wb') as f:
        async for chunk in response.aiter_bytes():
            f.write(chunk)
    return f'Downloaded {filename} from {url} and saved to {path}'


async def _extract_pdf(url: str, pages: list[int] = [1]) -> tuple[str, list]:
    import fitz
    from io import BytesIO
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ct = resp.headers.get('content-type', '')
        if 'pdf' not in ct.lower() and not url.lower().endswith('.pdf'):
            return '(Response is not a PDF)', []
    doc = fitz.open(stream=resp.content, filetype='pdf')
    total = len(doc)
    requested = sorted(set(max(1, min(p, total)) for p in pages))
    sections = []
    pil_images = []
    for p in requested:
        html = doc[p - 1].get_text('html')
        text = markdownify(html).strip()
        if text:
            sections.append(f'--- Page {p} of {total} ---\n{text}')
        else:
            try:
                from PIL import Image as PILImage
                pixmap = doc[p - 1].get_pixmap(dpi=150)
                pil_images.append(PILImage.open(BytesIO(pixmap.tobytes('png'))))
                sections.append(f'--- Page {p} of {total} ---\n(Image page)')
            except Exception:
                sections.append(f'--- Page {p} of {total} ---\n(No extractable text on this page)')
    doc.close()
    last = requested[-1]
    remaining = total - last
    footer = f'{remaining} page{"s" if remaining != 1 else ""} remaining after page {last}' if remaining > 0 else 'Last page reached'
    return '\n\n'.join(sections) + f'\n\n{footer}', pil_images


@Tool('scrape_tool', model=Scrape)
async def scrape_tool(prompt: str = None, pages: list[int] = [1], session: Browser = None, llm=None):
    '''Extracts content from the current webpage or PDF. Captures the entire page DOM — all content regardless of scroll position, not just the visible viewport.
    If prompt is given, uses the LLM to extract only the requested information.
    If prompt is omitted, returns the full content as markdown (HTML) or plain text (PDF).
    For PDFs, use pages=[1,5,10] to read specific pages together. If a prompt is given, it is applied across all requested pages combined.'''
    tab = await session.get_current_tab()
    url = tab.url if tab else ''

    is_pdf = url.lower().endswith('.pdf')
    if not is_pdf and url.startswith('http'):
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=5.0) as client:
                head = await client.head(url)
            is_pdf = 'pdf' in head.headers.get('content-type', '').lower()
        except Exception:
            try:
                ct = await session.execute_script(
                    "(function(){try{return document.contentType}catch(e){return ''}})()"
                )
                is_pdf = 'pdf' in str(ct).lower()
            except Exception:
                pass

    if is_pdf:
        content, images = await _extract_pdf(url, pages=pages)
        if images and llm:
            system = SystemMessage(content=(
                'You are analyzing pages from a PDF document that contains images. '
                'Describe what you see in each image in detail.'
            ))
            content = prompt if prompt else 'Describe the content of these PDF pages.'
            msg = ImageMessage(content=content, images=images)
            event = await llm.ainvoke(messages=[system, msg])
            if event.type == LLMEventType.TEXT:
                return f'PDF image content:\n{event.content}'
    else:
        html    = await session.get_page_content()
        content = markdownify(html)

    if prompt and llm:
        system = SystemMessage(content=(
            'You are a precise information extractor. '
            'The user will give you a webpage content and a specific extraction request. '
            'Extract only the requested information — nothing more. '
            'Be concise and structured. Use markdown lists or tables where appropriate. '
            'If the requested information is not present on the page, say so clearly.'
        ))
        msg = HumanMessage(content=(
            f'Extraction request: {prompt}\n\n'
            f'Webpage content:\n{content}'
        ))
        event = await llm.ainvoke(messages=[system, msg])
        if event.type == LLMEventType.TEXT:
            return f'Extracted information:\n{event.content}'

    return f'Scraped the contents of the webpage:\n{content}'


@Tool('tab_tool', model=Tab)
async def tab_tool(mode: Literal['open', 'close', 'switch'], tab_index: Optional[int] = None, session: Browser = None):
    '''Manages browser tabs: opens new blank tabs, closes the current tab, or switches between existing tabs by index.'''
    match mode:
        case 'open':
            await session.new_tab()
            await session._wait_for_page(timeout=5.0)
            return 'Opened a new blank tab and switched to it.'
        case 'close':
            if len(session._sessions) <= 1:
                return 'Cannot close the last remaining tab.'
            await session.close_tab()
            return 'Closed current tab and switched to the last remaining tab.'
        case 'switch':
            tabs = await session.get_all_tabs()
            if tab_index is None or tab_index < 0 or tab_index >= len(tabs):
                raise IndexError(f'Tab index {tab_index} out of range. Available: {len(tabs)}')
            await session.switch_tab(tab_index)
            await session._wait_for_page(timeout=5.0)
            return f'Switched to tab {tab_index} (Total tabs: {len(tabs)}).'
        case _:
            raise ValueError("Invalid mode. Use 'open', 'close', or 'switch'.")


@Tool('upload_tool', model=Upload)
async def upload_tool(index: int, filenames: list[str], session: Browser = None):
    '''Uploads one or more files to a file input element. Files must be present in the ./uploads directory.'''
    element = await session.get_element_by_index(index=index)
    xpath   = element.xpath.get('element', '')
    files   = [str(Path(getcwd()) / 'uploads' / fn) for fn in filenames]
    await session.set_file_input(xpath, files)
    return f'Uploaded {filenames} to element at label {index}'


@Tool('menu_tool', model=Menu)
async def menu_tool(index: int, labels: list[str], session: Browser = None):
    '''Selects one or more options in a <select> dropdown by their visible label text.'''
    element = await session.get_element_by_index(index=index)
    xpath   = element.xpath.get('element', '')
    result  = await session.select_option(xpath, labels)
    if not isinstance(result, dict) or result.get('error') == 'not_found':
        raise Exception(f'Could not resolve the dropdown element at label {index}')
    if result.get('error') == 'not_select':
        tag = result.get('tag', 'unknown')
        raise Exception(f'Element at label {index} is a <{tag}>, not a <select> dropdown — use click_tool to open custom dropdowns')
    not_found = result.get('notFound', [])
    if not_found:
        raise Exception(f"Could not find option(s) {not_found} in dropdown at label {index}. Available options: {result.get('available', [])}")
    return f"Selected {', '.join(result.get('selected', []))} in element at label {index}"


@Tool('script_tool', model=Script)
async def script_tool(script: str, session: Browser = None):
    '''Executes JavaScript on the current page and returns the result. Has full access to the entire DOM — all elements on the page regardless of scroll position or visibility.
    Always wrap in an IIFE with try-catch:
    (function(){ try { /* code */ } catch(e) { return 'Error: '+e.message } })()
    Use only browser APIs (document, window, DOM). Keep return values small.
    Only for elements without an index label — use click_tool for indexed elements.'''
    result = await session.execute_script(script, truncate=True, repair=True)
    return f'Script result: {result}'


@Tool('human_tool', model=HumanInput)
async def human_tool(prompt: str, session: Browser = None):
    '''Requests human assistance when encountering CAPTCHAs, OTP codes, or other challenges that require a human.'''
    print(colored(f'Agent: {prompt}', color='cyan', attrs=['bold']))
    human_response = input('User: ')
    return f"User provided: '{human_response}'"

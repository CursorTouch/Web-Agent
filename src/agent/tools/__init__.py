from src.agent.tools.service import (
    done_tool,
    click_tool,
    type_tool,
    wait_tool,
    scroll_tool,
    goto_tool,
    back_tool,
    forward_tool,
    key_tool,
    download_tool,
    scrape_tool,
    tab_tool,
    upload_tool,
    menu_tool,
    script_tool,
    human_tool,
)

BUILTIN_TOOLS = [
    click_tool, goto_tool, key_tool, scrape_tool,
    type_tool, scroll_tool, wait_tool, back_tool,
    tab_tool, done_tool, forward_tool, download_tool,
    script_tool, menu_tool,
]

__all__ = ['BUILTIN_TOOLS']

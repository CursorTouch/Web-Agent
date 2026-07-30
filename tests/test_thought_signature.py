"""Regression: thought_signature must survive LLMEvent -> ToolMessage -> provider history.

Gemini 3.x rejects a replayed function call whose thought_signature is missing:

    400 INVALID_ARGUMENT: Function call is missing a thought_signature in
    functionCall parts ... function call `default_api:goto_tool`, position 2

Both ends of the round-trip already worked: the Google provider extracts the
signature from a response and replays it onto the function-call Part when
rebuilding history. The gap was in the middle — the agent recorded its
ToolMessage without thinking/thinking_signature, so the value was dropped
between the two. That broke every multi-step task on its second turn.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from src.agent.registry.views import ToolResult
from src.agent.service import Agent
from src.messages import HumanMessage, ToolMessage
from src.providers.events import LLMEvent, LLMEventType, Thinking, ToolCall
from src.providers.google import ChatGoogle

SIGNATURE = b"test-thought-signature-bytes"
THINKING = "I should navigate to the page first."


def _tool_call_event(thinking: Thinking | None) -> LLMEvent:
    return LLMEvent(
        type=LLMEventType.TOOL_CALL,
        tool_call=ToolCall(
            id="call_1", name="goto_tool", params={"url": "https://example.com"}
        ),
        thinking=thinking,
    )


def _stubbed_agent(event: LLMEvent) -> Agent:
    """An Agent whose collaborators are stubbed, so aloop() runs one offline step."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=event)

    with patch("src.agent.service.Browser"), patch("src.agent.service.Context"), \
         patch("src.agent.service.Registry"), patch("src.agent.service.Hook"):
        agent = Agent(llm=llm, max_steps=1, log_to_console=False)

    agent.browser = MagicMock()
    agent.browser.crashed = False
    agent.browser._browser_state = None
    agent.context = MagicMock()
    agent.context.task = MagicMock(return_value=HumanMessage(content="task"))
    agent.context.state = AsyncMock(return_value=HumanMessage(content="state"))
    agent.registry = MagicMock()
    agent.registry.NON_TOOL_PARAMS = set()
    agent.registry.aexecute = AsyncMock(
        return_value=ToolResult(is_success=True, content="ok")
    )
    agent.hook_runner = AsyncMock()
    agent.state.task = "test task"
    return agent


def _run_one_step(event: LLMEvent) -> list[ToolMessage]:
    """Drive one aloop() iteration and return the ToolMessages it recorded."""
    agent = _stubbed_agent(event)
    with patch.object(Agent, "system_message", new_callable=PropertyMock) as system:
        system.return_value = HumanMessage(content="system")
        asyncio.run(agent.aloop())
    return [m for m in agent.state.messages if isinstance(m, ToolMessage)]


def _signatures_on_function_calls(contents) -> list:
    return [
        getattr(part, "thought_signature", None)
        for content in contents
        for part in (content.parts or [])
        if getattr(part, "function_call", None) is not None
    ]


def test_agent_records_thought_signature_on_tool_message():
    """The regression itself: the agent must not drop the signature."""
    messages = _run_one_step(
        _tool_call_event(Thinking(content=THINKING, signature=SIGNATURE))
    )
    assert messages, "agent recorded no ToolMessage"
    assert messages[0].thinking_signature == SIGNATURE
    assert messages[0].thinking == THINKING


def test_agent_tolerates_absent_thinking():
    """Providers that return no Thinking must still produce a usable ToolMessage."""
    messages = _run_one_step(_tool_call_event(None))
    assert messages, "agent recorded no ToolMessage"
    assert messages[0].thinking_signature is None
    assert messages[0].thinking is None


def test_provider_replays_signature_onto_function_call_part():
    """The provider half: a signed ToolMessage rebuilds into a signed Part."""
    llm = ChatGoogle(model="gemini-flash-latest", api_key="test-key")
    message = ToolMessage(
        id="call_1",
        name="goto_tool",
        params={"url": "https://example.com"},
        content="ok",
        thinking=THINKING,
        thinking_signature=SIGNATURE,
    )
    _, contents = llm._convert_messages([message])
    assert SIGNATURE in _signatures_on_function_calls(contents)


def test_signature_survives_agent_to_provider_round_trip():
    """End to end: what the agent records is what the provider replays."""
    messages = _run_one_step(
        _tool_call_event(Thinking(content=THINKING, signature=SIGNATURE))
    )
    llm = ChatGoogle(model="gemini-flash-latest", api_key="test-key")
    _, contents = llm._convert_messages(messages)
    assert SIGNATURE in _signatures_on_function_calls(contents), (
        "signature lost between the agent's ToolMessage and the provider's history"
    )

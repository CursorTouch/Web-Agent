from src.agent.events import AgentEvent, Event, EventType, ConsoleEventSubscriber, FileEventSubscriber
from src.messages import AIMessage, HumanMessage
from src.agent.tools import BUILTIN_TOOLS, human_tool
from src.agent.registry import Registry
from src.agent.views import AgentResult, AgentState
from src.agent.browser import Browser, BrowserConfig
from src.agent.context import Context
from src.agent.loop import LoopGuard
from src.providers.events import LLMEventType
from src.messages import ToolMessage
from src.agent.base import BaseAgent
from src.hooks import BaseHook, Hook, GlowHook
from src.tools import Tool
from rich.markdown import Markdown
from rich.console import Console
from itertools import chain
from typing import Callable, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from src.providers.base import BaseChatLLM

DONE_TOOL_NAME = "done_tool"


class Agent(BaseAgent):
    def __init__(
        self,
        config: BrowserConfig = None,
        additional_tools: list[Tool] = [],
        instructions: list[str] = [],
        llm: 'BaseChatLLM' = None,
        max_steps: int = 25,
        max_consecutive_failures: int = 3,
        use_vision: bool = False,
        use_web_mcp: bool = False,
        enable_loop_detection: bool = False,
        include_human_in_loop: bool = False,
        log_to_file: bool = False,
        log_to_console: bool = True,
        event_subscriber: Callable[[AgentEvent], None] | None = None,
        sensitive_data: dict[str, str] = {},
        keep_alive: bool = False,
        hooks: list[BaseHook] | None = None,
    ) -> None:
        self.browser = Browser(config=config)
        self.hook_runner = Hook([GlowHook()] + (hooks or []))
        self.browser.hooks = self.hook_runner
        self.context = Context(session=self.browser)
        self.use_web_mcp = use_web_mcp
        self.registry = Registry(
            tools=BUILTIN_TOOLS + additional_tools + ([human_tool] if include_human_in_loop else []),
            browser=self.browser if use_web_mcp else None
        )
        self.state = AgentState(
            max_steps=max_steps,
            max_consecutive_failures=max_consecutive_failures,
        )
        self.instructions = instructions
        self.use_vision = use_vision
        self.llm = llm
        self.sensitive_data = sensitive_data
        self.keep_alive = keep_alive
        self._cached_system_message = None
        self._loop_guard = LoopGuard() if enable_loop_detection else None

        self.event = Event()
        if event_subscriber is not None:
            self.event.add_subscriber(event_subscriber)
        if log_to_console:
            self.event.add_subscriber(ConsoleEventSubscriber())
        if log_to_file:
            self.event.add_subscriber(FileEventSubscriber())

    @property
    def system_message(self):
        if self._cached_system_message is None:
            self._cached_system_message = self.context.system(
                instructions=self.instructions,
                max_steps=self.state.max_steps,
            )
        return self._cached_system_message

    @property
    def tools(self):
        return self.registry.get_tools()

    def _scrub_sensitive(self, text: str) -> str:
        """Replace any real credential values in text with their placeholder names."""
        for placeholder, real_value in self.sensitive_data.items():
            if real_value and real_value in text:
                text = text.replace(real_value, f'<{placeholder}>')
        return text

    def _scrub_messages(self):
        """Scan all messages in state and redact any leaked credential values."""
        if not self.sensitive_data:
            return
        for msg in self.state.messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                msg.content = self._scrub_sensitive(msg.content)

    def _resolve_sensitive(self, tool_params: dict) -> dict:
        """Replace placeholder keys with real credential values.

        Any param whose string value exactly matches a key in sensitive_data is
        substituted with the real value.  Keys ending with '_2fa_code' are treated
        as TOTP secrets — a fresh code is generated at call time via pyotp.
        """
        if not self.sensitive_data:
            return tool_params
        import pyotp
        resolved = {}
        for k, v in tool_params.items():
            if isinstance(v, str) and v in self.sensitive_data:
                secret = self.sensitive_data[v]
                resolved[k] = pyotp.TOTP(secret).now() if v.endswith('_2fa_code') else secret
            else:
                resolved[k] = v
        return resolved

    async def aloop(self) -> AgentResult:
        self.state.messages.insert(0, self.system_message)
        self.state.messages.append(self.context.task(self.state.task))
        consecutive_failures = 0
        tool_result = "No previous action."
        if self._loop_guard:
            self._loop_guard.reset()

        for step in range(self.state.max_steps):
            self.state.step = step

            if self.browser.crashed:
                error = 'Browser crashed — all tabs are gone. Aborting.'
                self.event.emit(AgentEvent(type=EventType.ERROR, data={'step': step, 'error': error}))
                return AgentResult(is_done=False, error=error)

            nudge=None
            if self._loop_guard:
                nudge = self._loop_guard.check()
            state_msg = await self.context.state(
                query=self.state.task,
                step=step,
                max_steps=self.state.max_steps,
                tool_result=tool_result,
                use_vision=self.use_vision,
                nudge=nudge or '',
            )
            if nudge:
                self.event.emit(AgentEvent(type=EventType.ERROR, data={'step': step, 'error': f'Loop detected: {nudge}'}))
            self.state.messages.append(state_msg)

            # Record page fingerprint (browser state is freshly populated by context.state)
            bs = self.browser._browser_state
            if bs and bs.current_tab:
                dom_text = bs.dom_state.semantic_tree_to_string()
                if self._loop_guard:
                    self._loop_guard.record_page(bs.current_tab.url, dom_text)

            # Reason: call LLM with tools
            message: ToolMessage | None = None
            last_error: Exception | None = None
            
            if self.use_web_mcp:
                await self.registry.refresh_dynamic_tools()
            
            for attempt in range(self.state.max_consecutive_failures):
                try:
                    messages = list(chain(self.state.messages, self.state.error_messages))
                    llm_event = await self.llm.ainvoke(messages=messages, tools=self.tools)
                    match llm_event.type:
                        case LLMEventType.TOOL_CALL:
                            message = ToolMessage(
                                id=llm_event.tool_call.id,
                                name=llm_event.tool_call.name,
                                params=llm_event.tool_call.params,
                                thinking=llm_event.thinking.content if llm_event.thinking else None,
                                thinking_signature=llm_event.thinking.signature if llm_event.thinking else None,
                            )
                            break
                        case LLMEventType.TEXT:
                            ai_msg = AIMessage(content=llm_event.content)
                            human_msg = HumanMessage(
                                content="Response rejected. You must call a tool. Use `done_tool` to complete the task."
                            )
                            self.state.error_messages.extend([ai_msg, human_msg])
                            continue
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    is_rate_limit = '429' in error_str or 'rate_limit' in error_str.lower() or 'rate limit' in error_str.lower()
                    if attempt < self.state.max_consecutive_failures - 1:
                        wait_time = 30 if is_rate_limit else 2 ** (attempt + 1)
                        self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": step, "error": f"LLM call failed, retrying ({attempt + 1}/{self.state.max_consecutive_failures}): {e}"}))
                        await asyncio.sleep(wait_time)
                    else:
                        self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": step, "error": f"All {self.state.max_consecutive_failures} LLM attempts exhausted: {e}"}))

            if message is None:
                error = f"Agent failed after exhausting retries: {last_error}"
                self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": step, "error": error}))
                return AgentResult(is_done=False, error=error)

            self.state.messages.pop()  # Remove state message

            tool_name   = message.name
            tool_params = message.params

            thought = tool_params.get("thought", "")
            self.event.emit(AgentEvent(type=EventType.THOUGHT, data={"step": step, "thought": thought}))
            await self.hook_runner.on_thought(step=step, thought=thought, browser=self.browser)

            filtered_params = {k: v for k, v in tool_params.items() if k not in self.registry.NON_TOOL_PARAMS}
            if tool_name != DONE_TOOL_NAME:
                self.event.emit(AgentEvent(
                    type=EventType.TOOL_CALL,
                    data={"step": step, "tool_name": tool_name, "tool_params": filtered_params},
                ))
                await self.hook_runner.on_tool_call(step=step, tool_name=tool_name, params=filtered_params, browser=self.browser)

            # Act: execute tool (resolve sensitive placeholders just before execution)
            exec_params = self._resolve_sensitive(tool_params)
            tool_result_obj = await self.registry.aexecute(
                tool_name=tool_name,
                tool_params=exec_params,
            )

            if self._loop_guard:
                self._loop_guard.record_action(tool_name, exec_params, is_success=tool_result_obj.is_success)

            if tool_result_obj.is_success:
                content = tool_result_obj.content
                tool_result = content
                message.content = content
                self.state.messages.append(message)
                self._scrub_messages()
                self.state.error_messages.clear()
                consecutive_failures = 0
            else:
                content = tool_result_obj.error
                tool_result = f"Tool failed: {content}"
                message.content = content
                self.state.error_messages.append(message)
                consecutive_failures += 1

            if tool_name != DONE_TOOL_NAME:
                self.event.emit(AgentEvent(
                    type=EventType.TOOL_RESULT,
                    data={"step": step, "tool_name": tool_name, "is_success": tool_result_obj.is_success, "content": content},
                ))
                await self.hook_runner.on_tool_result(step=step, tool_name=tool_name, success=tool_result_obj.is_success, content=content, browser=self.browser)

            if not tool_result_obj.is_success:
                if consecutive_failures >= self.state.max_consecutive_failures:
                    error = (
                        f"Agent aborted after {self.state.max_consecutive_failures} "
                        f"consecutive failures. Last: {content}"
                    )
                    self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": step, "error": error}))
                    await self.hook_runner.on_agent_error(error=error, step=step, browser=self.browser)
                    return AgentResult(is_done=False, error=error)

            if tool_name == DONE_TOOL_NAME:
                content = tool_params.get("content", content)
                self.event.emit(AgentEvent(type=EventType.DONE, data={"step": step, "content": content}))
                return AgentResult(is_done=True, content=content)

        error = f"Agent reached max steps ({self.state.max_steps}) without completing."
        self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": self.state.max_steps, "error": error}))
        await self.hook_runner.on_agent_error(error=error, step=self.state.max_steps, browser=self.browser)
        return AgentResult(is_done=False, error=error)

    async def ainvoke(self, task: str) -> AgentResult:
        self.state.reset()
        self.state.task = task
        await self.browser.ensure_open()
        self.registry.add_extension('session', self.browser)
        self.registry.add_extension('llm', self.llm)
        await self.hook_runner.on_agent_start(task=task, browser=self.browser)
        try:
            result = await self.aloop()
            await self.hook_runner.on_agent_done(result=result, browser=self.browser)
            return result
        except Exception as e:
            error = str(e)
            self.event.emit(AgentEvent(type=EventType.ERROR, data={"step": self.state.step, "error": error}))
            await self.hook_runner.on_agent_error(error=error, step=self.state.step, browser=self.browser)
            return AgentResult(is_done=False, error=error)
        finally:
            await self.close()

    def invoke(self, task: str) -> AgentResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainvoke(task))

    def print_response(self, task: str):
        console = Console()
        result = self.invoke(task)
        if result.is_done and result.content:
            console.print(Markdown(result.content))
        elif result.error:
            console.print(f"[red]Error:[/red] {result.error}")

    async def close(self):
        self.event.close()
        try:
            if self.keep_alive:
                await self.browser.disconnect()
            else:
                await self.browser.close_browser()
        except Exception:
            pass
        finally:
            pass  # browser closed
            self.browser = None

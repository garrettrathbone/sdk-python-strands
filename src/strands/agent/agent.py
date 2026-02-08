"""Agent Interface.

This module implements the core Agent class that serves as the primary entry point for interacting with foundation
models and tools in the SDK.

The Agent interface supports two complementary interaction patterns:

1. Natural language for conversation: `agent("Analyze this data")`
2. Method-style for direct tool access: `agent.tool.tool_name(param1="value")`
"""

import logging
import threading
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
)

from opentelemetry import trace as trace_api
from pydantic import BaseModel

from .. import _identifier
from .._async import run_async
from ..event_loop._retry import ModelRetryStrategy
from ..event_loop.event_loop import INITIAL_DELAY, MAX_ATTEMPTS, MAX_DELAY, event_loop_cycle
from ..tools._tool_helpers import generate_missing_tool_result_content

if TYPE_CHECKING:
    from ..tools import ToolProvider
from ..handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from ..hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from ..interrupt import _InterruptState
from ..models.bedrock import BedrockModel
from ..models.model import Model
from ..session.session_manager import SessionManager
from ..telemetry.metrics import EventLoopMetrics
from ..telemetry.tracer import get_tracer, serialize
from ..tools._caller import _ToolCaller
from ..tools.executors import ConcurrentToolExecutor
from ..tools.executors._executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.structured_output._structured_output_context import StructuredOutputContext
from ..tools.watcher import ToolWatcher
from ..types._events import AgentResultEvent, EventLoopStopEvent, InitEventLoopEvent, ModelStreamChunkEvent, TypedEvent
from ..types.agent import AgentInput
from ..types.content import ContentBlock, Message, Messages, SystemContentBlock
from ..types.exceptions import AgentDelegationException, ConcurrencyException, ContextWindowOverflowException
from ..types.tools import ToolResult, ToolUse
from ..types.traces import AttributeValue
from .agent_result import AgentResult
from .base import AgentBase
from .conversation_manager import (
    ConversationManager,
    SlidingWindowConversationManager,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# TypeVar for generic structured output
T = TypeVar("T", bound=BaseModel)


# Sentinel class and object to distinguish between explicit None and default parameter value
class _DefaultCallbackHandlerSentinel:
    """Sentinel class to distinguish between explicit None and default parameter value."""

    pass


class _DefaultRetryStrategySentinel:
    """Sentinel class to distinguish between explicit None and default parameter value for retry_strategy."""

    pass


_DEFAULT_CALLBACK_HANDLER = _DefaultCallbackHandlerSentinel()
_DEFAULT_RETRY_STRATEGY = _DefaultRetryStrategySentinel()
_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class Agent(AgentBase):
    """Core Agent implementation.

    An agent orchestrates the following workflow:

    1. Receives user input
    2. Processes the input using a language model
    3. Decides whether to use tools to gather information or perform actions
    4. Executes those tools and receives results
    5. Continues reasoning with the new information
    6. Produces a final response
    """

    # For backwards compatibility
    ToolCaller = _ToolCaller

    def __init__(
        self,
        model: Model | str | None = None,
        messages: Messages | None = None,
        tools: list[Union[str, dict[str, str], "ToolProvider", Any]] | None = None,
        system_prompt: str | list[SystemContentBlock] | None = None,
        structured_output_model: type[BaseModel] | None = None,
        callback_handler: Callable[..., Any] | _DefaultCallbackHandlerSentinel | None = _DEFAULT_CALLBACK_HANDLER,
        conversation_manager: ConversationManager | None = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        trace_attributes: Mapping[str, AttributeValue] | None = None,
        *,
        agent_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        state: AgentState | dict | None = None,
        hooks: list[HookProvider] | None = None,
        session_manager: SessionManager | None = None,
        structured_output_prompt: str | None = None,
        tool_executor: ToolExecutor | None = None,
        retry_strategy: ModelRetryStrategy | _DefaultRetryStrategySentinel | None = _DEFAULT_RETRY_STRATEGY,
        sub_agents: Optional[list["Agent"]] = None,
        delegation_timeout: Optional[float] = 300.0,
        delegation_state_transfer: bool = True,
        delegation_message_transfer: bool = True,
        delegation_state_serializer: Optional[Callable[[Any], Any]] = None,
        max_delegation_depth: int = 10,
        delegation_streaming_proxy: bool = True,
    ):
        """Initialize the Agent with the specified configuration.

        Args:
            model: Provider for running inference or a string representing the model-id for Bedrock to use.
                Defaults to strands.models.BedrockModel if None.
            messages: List of initial messages to pre-load into the conversation.
                Defaults to an empty list if None.
            tools: List of tools to make available to the agent.
                Can be specified as:

                - String tool names (e.g., "retrieve")
                - File paths (e.g., "/path/to/tool.py")
                - Imported Python modules (e.g., from strands_tools import current_time)
                - Dictionaries with name/path keys (e.g., {"name": "tool_name", "path": "/path/to/tool.py"})
                - ToolProvider instances for managed tool collections
                - Functions decorated with `@strands.tool` decorator.

                If provided, only these tools will be available. If None, all tools will be available.
            system_prompt: System prompt to guide model behavior.
                Can be a string or a list of SystemContentBlock objects for advanced features like caching.
                If None, the model will behave according to its default settings.
            structured_output_model: Pydantic model type(s) for structured output.
                When specified, all agent calls will attempt to return structured output of this type.
                This can be overridden on the agent invocation.
                Defaults to None (no structured output).
            callback_handler: Callback for processing events as they happen during agent execution.
                If not provided (using the default), a new PrintingCallbackHandler instance is created.
                If explicitly set to None, null_callback_handler is used.
            conversation_manager: Manager for conversation history and context window.
                Defaults to strands.agent.conversation_manager.SlidingWindowConversationManager if None.
            record_direct_tool_call: Whether to record direct tool calls in message history.
                Defaults to True.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
                Defaults to False.
            trace_attributes: Custom trace attributes to apply to the agent's trace span.
            agent_id: Optional ID for the agent, useful for session management and multi-agent scenarios.
                Defaults to "default".
            name: name of the Agent
                Defaults to "Strands Agents".
            description: description of what the Agent does
                Defaults to None.
            state: stateful information for the agent. Can be either an AgentState object, or a json serializable dict.
                Defaults to an empty AgentState object.
            hooks: hooks to be added to the agent hook registry
                Defaults to None.
            session_manager: Manager for handling agent sessions including conversation history and state.
                If provided, enables session-based persistence and state management.
            sub_agents: List of sub-agents available for delegation.
                Each sub-agent will have a corresponding handoff_to_{name} tool
                auto-generated for complete delegation.
            delegation_timeout: Timeout in seconds for delegation operations.
                Defaults to 300 seconds (5 minutes). Set to None for no timeout.
            delegation_state_transfer: Whether to transfer agent.state to sub-agents.
                Defaults to True. When True, sub-agents receive a deep copy of the
                orchestrator's state. When False, sub-agents use their own state.
            delegation_message_transfer: Whether to transfer conversation history.
                Defaults to True. Controls whether messages are copied to sub-agent.
            max_delegation_depth: Maximum allowed depth for nested delegation.
                Prevents infinite delegation chains. Defaults to 10.
            delegation_state_serializer: Optional custom serializer for state transfer.
                When provided, this callable will be used to serialize state instead of
                deepcopy. Useful for large or complex states where deepcopy is inefficient.
                Should return a serialized copy of the state.
            delegation_streaming_proxy: Whether to proxy streaming events from sub-agents.
                Defaults to True. When True, streaming events from sub-agents are
                proxied back to the original caller for real-time visibility.

            structured_output_prompt: Custom prompt message used when forcing structured output.
                When using structured output, if the model doesn't automatically use the output tool,
                the agent sends a follow-up message to request structured formatting. This parameter
                allows customizing that message.
                Defaults to "You must format the previous response as structured output."
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
            retry_strategy: Strategy for retrying model calls on throttling or other transient errors.
                Defaults to ModelRetryStrategy with max_attempts=6, initial_delay=4s, max_delay=240s.
                Implement a custom HookProvider for custom retry logic, or pass None to disable retries.

        Raises:
            ValueError: If agent id contains path separators.
        """
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model
        self.messages = messages if messages is not None else []
        # initializing self._system_prompt for backwards compatibility
        self._system_prompt, self._system_prompt_content = self._initialize_system_prompt(system_prompt)
        self._default_structured_output_model = structured_output_model
        self._structured_output_prompt = structured_output_prompt
        self.agent_id = _identifier.validate(agent_id or _DEFAULT_AGENT_ID, _identifier.Identifier.AGENT)
        self.name = name or _DEFAULT_AGENT_NAME
        self.description = description

        # If not provided, create a new PrintingCallbackHandler instance
        # If explicitly set to None, use null_callback_handler
        # Otherwise use the passed callback_handler
        self.callback_handler: Callable[..., Any] | PrintingCallbackHandler
        if isinstance(callback_handler, _DefaultCallbackHandlerSentinel):
            self.callback_handler = PrintingCallbackHandler()
        elif callback_handler is None:
            self.callback_handler = null_callback_handler
        else:
            self.callback_handler = callback_handler

        self.conversation_manager = conversation_manager if conversation_manager else SlidingWindowConversationManager()

        # Process trace attributes to ensure they're of compatible types
        self.trace_attributes: dict[str, AttributeValue] = {}
        if trace_attributes:
            for k, v in trace_attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    self.trace_attributes[k] = v

        self.record_direct_tool_call = record_direct_tool_call
        self.load_tools_from_directory = load_tools_from_directory

        self.tool_registry = ToolRegistry()

        # Process tool list if provided
        if tools is not None:
            self.tool_registry.process_tools(tools)

        # Initialize tools and configuration
        self.tool_registry.initialize_tools(self.load_tools_from_directory)
        if load_tools_from_directory:
            self.tool_watcher = ToolWatcher(tool_registry=self.tool_registry)

        self.event_loop_metrics = EventLoopMetrics()

        # Initialize tracer instance (no-op if not configured)
        self.tracer = get_tracer()
        self.trace_span: trace_api.Span | None = None

        # Initialize agent state management
        if state is not None:
            if isinstance(state, dict):
                self.state = AgentState(state)
            elif isinstance(state, AgentState):
                self.state = state
            else:
                raise ValueError("state must be an AgentState object or a dict")
        else:
            self.state = AgentState()

        self.tool_caller = _ToolCaller(self)

        self.hooks = HookRegistry()

        self._interrupt_state = _InterruptState()

        # Initialize lock for guarding concurrent invocations
        # Using threading.Lock instead of asyncio.Lock because run_async() creates
        # separate event loops in different threads, so asyncio.Lock wouldn't work
        self._invocation_lock = threading.Lock()

        # In the future, we'll have a RetryStrategy base class but until
        # that API is determined we only allow ModelRetryStrategy
        if (
            retry_strategy is not None
            and not isinstance(retry_strategy, _DefaultRetryStrategySentinel)
            and type(retry_strategy) is not ModelRetryStrategy
        ):
            raise ValueError("retry_strategy must be an instance of ModelRetryStrategy")

        # If not provided (using the default), create a new ModelRetryStrategy instance
        # If explicitly set to None, disable retries (max_attempts=1 means no retries)
        # Otherwise use the passed retry_strategy
        if isinstance(retry_strategy, _DefaultRetryStrategySentinel):
            self._retry_strategy = ModelRetryStrategy(
                max_attempts=MAX_ATTEMPTS, max_delay=MAX_DELAY, initial_delay=INITIAL_DELAY
            )
        elif retry_strategy is None:
            # If no retry strategy is passed in, then we turn retries off
            self._retry_strategy = ModelRetryStrategy(max_attempts=1)
        else:
            self._retry_strategy = retry_strategy

        # Initialize session management functionality
        self._session_manager = session_manager
        if self._session_manager:
            self.hooks.add_hook(self._session_manager)

        # Allow conversation_managers to subscribe to hooks
        self.hooks.add_hook(self.conversation_manager)

        # Register retry strategy as a hook
        self.hooks.add_hook(self._retry_strategy)

        self.tool_executor = tool_executor or ConcurrentToolExecutor()

        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)
        self.hooks.invoke_callbacks(AgentInitializedEvent(agent=self))

        # Initialization of the sub-agents and delegation configuration

        self._sub_agents: dict[str, "Agent"] = {}
        self.delegation_timeout = delegation_timeout
        self.delegation_state_transfer = delegation_state_transfer
        self.delegation_message_transfer = delegation_message_transfer
        self.delegation_state_serializer = delegation_state_serializer
        self.max_delegation_depth = max_delegation_depth
        self.delegation_streaming_proxy = delegation_streaming_proxy

        if sub_agents:
            self._validate_sub_agents(sub_agents)
            for sub_agent in sub_agents:
                self._sub_agents[sub_agent.name] = sub_agent
            self._generate_delegation_tools(list(self._sub_agents.values()))

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt as a string for backwards compatibility.

        Returns the system prompt as a concatenated string when it contains text content,
        or None if no text content is present. This maintains backwards compatibility
        with existing code that expects system_prompt to be a string.

        Returns:
            The system prompt as a string, or None if no text content exists.
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | list[SystemContentBlock] | None) -> None:
        """Set the system prompt and update internal content representation.

        Accepts either a string or list of SystemContentBlock objects.
        When set, both the backwards-compatible string representation and the internal
        content block representation are updated to maintain consistency.

        Args:
            value: System prompt as string, list of SystemContentBlock objects, or None.
                  - str: Simple text prompt (most common use case)
                  - list[SystemContentBlock]: Content blocks with features like caching
                  - None: Clear the system prompt
        """
        self._system_prompt, self._system_prompt_content = self._initialize_system_prompt(value)

    @property
    def tool(self) -> _ToolCaller:
        """Call tool as a function.

        Returns:
            Tool caller through which user can invoke tool as a function.

        Example:
            ```
            agent = Agent(tools=[calculator])
            agent.tool.calculator(...)
            ```
        """
        return self.tool_caller

    @property
    def tool_names(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            Names of all tools available to this agent.
        """
        all_tools = self.tool_registry.get_all_tools_config()
        return list(all_tools.keys())

    def __call__(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[BaseModel] | None = None,
        structured_output_prompt: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface with multiple input patterns:
        - String input: `agent("hello!")`
        - ContentBlock list: `agent([{"text": "hello"}, {"image": {...}}])`
        - Message list: `agent([{"role": "user", "content": [{"text": "hello"}]}])`
        - No input: `agent()` - uses existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            structured_output_prompt: Custom prompt for forcing structured output (overrides agent default).
            **kwargs: Additional parameters to pass through the event loop.[Deprecating]

        Returns:
            Result object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop
                - structured_output: Parsed structured output when structured_output_model was specified
        """
        return run_async(
            lambda: self.invoke_async(
                prompt,
                invocation_state=invocation_state,
                structured_output_model=structured_output_model,
                structured_output_prompt=structured_output_prompt,
                **kwargs,
            )
        )

    async def invoke_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[BaseModel] | None = None,
        structured_output_prompt: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface with multiple input patterns:
        - String input: Simple text input
        - ContentBlock list: Multi-modal content blocks
        - Message list: Complete messages with roles
        - No input: Use existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            structured_output_prompt: Custom prompt for forcing structured output (overrides agent default).
            **kwargs: Additional parameters to pass through the event loop.[Deprecating]

        Returns:
            Result: object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop
        """
        events = self.stream_async(
            prompt,
            invocation_state=invocation_state,
            structured_output_model=structured_output_model,
            structured_output_prompt=structured_output_prompt,
            **kwargs,
        )
        async for event in events:
            _ = event

        return cast(AgentResult, event["result"])

    def structured_output(self, output_model: type[T], prompt: AgentInput = None) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history

        Raises:
            ValueError: If no conversation history or prompt is provided.
        """
        warnings.warn(
            "Agent.structured_output method is deprecated."
            " You should pass in `structured_output_model` directly into the agent invocation."
            " see: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/structured-output/",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return run_async(lambda: self.structured_output_async(output_model, prompt))

    async def structured_output_async(self, output_model: type[T], prompt: AgentInput = None) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent (will not be added to conversation history).

        Raises:
            ValueError: If no conversation history or prompt is provided.
        -
        """
        if self._interrupt_state.activated:
            raise RuntimeError("cannot call structured output during interrupt")

        warnings.warn(
            "Agent.structured_output_async method is deprecated."
            " You should pass in `structured_output_model` directly into the agent invocation."
            " see: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/structured-output/",
            category=DeprecationWarning,
            stacklevel=2,
        )
        await self.hooks.invoke_callbacks_async(BeforeInvocationEvent(agent=self, invocation_state={}))
        with self.tracer.tracer.start_as_current_span(
            "execute_structured_output", kind=trace_api.SpanKind.CLIENT
        ) as structured_output_span:
            try:
                if not self.messages and not prompt:
                    raise ValueError("No conversation history or prompt provided")

                temp_messages: Messages = self.messages + await self._convert_prompt_to_messages(prompt)

                structured_output_span.set_attributes(
                    {
                        "gen_ai.system": "strands-agents",
                        "gen_ai.agent.name": self.name,
                        "gen_ai.agent.id": self.agent_id,
                        "gen_ai.operation.name": "execute_structured_output",
                    }
                )
                if self.system_prompt:
                    structured_output_span.add_event(
                        "gen_ai.system.message",
                        attributes={"role": "system", "content": serialize([{"text": self.system_prompt}])},
                    )
                for message in temp_messages:
                    structured_output_span.add_event(
                        f"gen_ai.{message['role']}.message",
                        attributes={"role": message["role"], "content": serialize(message["content"])},
                    )
                events = self.model.structured_output(output_model, temp_messages, system_prompt=self.system_prompt)
                async for event in events:
                    if isinstance(event, TypedEvent):
                        event.prepare(invocation_state={})
                        if event.is_callback_event:
                            self.callback_handler(**event.as_dict())

                structured_output_span.add_event(
                    "gen_ai.choice", attributes={"message": serialize(event["output"].model_dump())}
                )
                return event["output"]

            finally:
                await self.hooks.invoke_callbacks_async(AfterInvocationEvent(agent=self, invocation_state={}))

    def cleanup(self) -> None:
        """Clean up resources used by the agent.

        This method cleans up all tool providers that require explicit cleanup,
        such as MCP clients. It should be called when the agent is no longer needed
        to ensure proper resource cleanup.

        Note: This method uses a "belt and braces" approach with automatic cleanup
        through finalizers as a fallback, but explicit cleanup is recommended.
        """
        self.tool_registry.cleanup()

    def __del__(self) -> None:
        """Clean up resources when agent is garbage collected."""
        # __del__ is called even when an exception is thrown in the constructor,
        # so there is no guarantee tool_registry was set..
        if hasattr(self, "tool_registry"):
            self.tool_registry.cleanup()

    async def stream_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[BaseModel] | None = None,
        structured_output_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Process a natural language prompt and yield events as an async iterator.

        This method provides an asynchronous interface for streaming agent events with multiple input patterns:
        - String input: Simple text input
        - ContentBlock list: Multi-modal content blocks
        - Message list: Complete messages with roles
        - No input: Use existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            structured_output_prompt: Custom prompt for forcing structured output (overrides agent default).
            **kwargs: Additional parameters to pass to the event loop.[Deprecating]

        Yields:
            An async iterator that yields events. Each event is a dictionary containing
                information about the current state of processing, such as:

                - data: Text content being generated
                - complete: Whether this is the final chunk
                - current_tool_use: Information about tools being executed
                - And other event data provided by the callback handler

        Raises:
            ConcurrencyException: If another invocation is already in progress on this agent instance.
            Exception: Any exceptions from the agent invocation will be propagated to the caller.

        Example:
            ```python
            async for event in agent.stream_async("Analyze this data"):
                if "data" in event:
                    yield event["data"]
            ```
        """
        # Acquire lock to prevent concurrent invocations
        # Using threading.Lock instead of asyncio.Lock because run_async() creates
        # separate event loops in different threads
        acquired = self._invocation_lock.acquire(blocking=False)
        if not acquired:
            raise ConcurrencyException(
                "Agent is already processing a request. Concurrent invocations are not supported."
            )

        try:
            self._interrupt_state.resume(prompt)

            self.event_loop_metrics.reset_usage_metrics()

            merged_state = {}
            if kwargs:
                warnings.warn("`**kwargs` parameter is deprecating, use `invocation_state` instead.", stacklevel=2)
                merged_state.update(kwargs)
                if invocation_state is not None:
                    merged_state["invocation_state"] = invocation_state
            else:
                if invocation_state is not None:
                    merged_state = invocation_state

            callback_handler = self.callback_handler
            if kwargs:
                callback_handler = kwargs.get("callback_handler", self.callback_handler)

            # Process input and get message to add (if any)
            messages = await self._convert_prompt_to_messages(prompt)

            self.trace_span = self._start_agent_trace_span(messages)

            with trace_api.use_span(self.trace_span):
                try:
                    events = self._run_loop(messages, merged_state, structured_output_model, structured_output_prompt)

                    async for event in events:
                        event.prepare(invocation_state=merged_state)

                        if event.is_callback_event:
                            as_dict = event.as_dict()
                            callback_handler(**as_dict)
                            yield as_dict

                    result = AgentResult(*event["stop"])
                    callback_handler(result=result)
                    yield AgentResultEvent(result=result).as_dict()

                    self._end_agent_trace_span(response=result)

                except Exception as e:
                    self._end_agent_trace_span(error=e)
                    raise

        finally:
            self._invocation_lock.release()

    async def _run_loop(
        self,
        messages: Messages,
        invocation_state: dict[str, Any],
        structured_output_model: type[BaseModel] | None = None,
        structured_output_prompt: str | None = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the agent's event loop with the given message and parameters.

        Args:
            messages: The input messages to add to the conversation.
            invocation_state: Additional parameters to pass to the event loop.
            structured_output_model: Optional Pydantic model type for structured output.
            structured_output_prompt: Optional custom prompt for forcing structured output.

        Yields:
            Events from the event loop cycle.
        """
        before_invocation_event, _interrupts = await self.hooks.invoke_callbacks_async(
            BeforeInvocationEvent(agent=self, invocation_state=invocation_state, messages=messages)
        )
        messages = before_invocation_event.messages if before_invocation_event.messages is not None else messages

        agent_result: AgentResult | None = None
        try:
            yield InitEventLoopEvent()

            await self._append_messages(*messages)

            structured_output_context = StructuredOutputContext(
                structured_output_model or self._default_structured_output_model,
                structured_output_prompt=structured_output_prompt or self._structured_output_prompt,
            )

            # Execute the event loop cycle with retry logic for context limits
            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                # Signal from the model provider that the message sent by the user should be redacted,
                # likely due to a guardrail.
                if (
                    isinstance(event, ModelStreamChunkEvent)
                    and event.chunk
                    and event.chunk.get("redactContent")
                    and event.chunk["redactContent"].get("redactUserContentMessage")
                ):
                    self.messages[-1]["content"] = self._redact_user_content(
                        self.messages[-1]["content"], str(event.chunk["redactContent"]["redactUserContentMessage"])
                    )
                    if self._session_manager:
                        self._session_manager.redact_latest_message(self.messages[-1], self)
                yield event

            # Capture the result from the final event if available
            if isinstance(event, EventLoopStopEvent):
                agent_result = AgentResult(*event["stop"])

        finally:
            self.conversation_manager.apply_management(self)
            await self.hooks.invoke_callbacks_async(
                AfterInvocationEvent(agent=self, invocation_state=invocation_state, result=agent_result)
            )

    async def _execute_event_loop_cycle(
        self, invocation_state: dict[str, Any], structured_output_context: StructuredOutputContext | None = None
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the event loop cycle with retry logic for context window limits.

        This internal method handles the execution of the event loop cycle and implements
        retry logic for handling context window overflow exceptions by reducing the
        conversation context and retrying.

        Args:
            invocation_state: Additional parameters to pass to the event loop.
            structured_output_context: Optional structured output context for this invocation.

        Yields:
            Events of the loop cycle.
        """
        # Add `Agent` to invocation_state to keep backwards-compatibility
        invocation_state["agent"] = self

        if structured_output_context:
            structured_output_context.register_tool(self.tool_registry)

        try:
            events = event_loop_cycle(
                agent=self,
                invocation_state=invocation_state,
                structured_output_context=structured_output_context,
            )
            async for event in events:
                yield event

        except ContextWindowOverflowException as e:
            # Try reducing the context size and retrying
            self.conversation_manager.reduce_context(self, e=e)

            # Sync agent after reduce_context to keep conversation_manager_state up to date in the session
            if self._session_manager:
                self._session_manager.sync_agent(self)

            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                yield event

        finally:
            if structured_output_context:
                structured_output_context.cleanup(self.tool_registry)

    async def _convert_prompt_to_messages(self, prompt: AgentInput) -> Messages:
        if self._interrupt_state.activated:
            return []

        messages: Messages | None = None
        if prompt is not None:
            # Check if the latest message is toolUse
            if len(self.messages) > 0 and any("toolUse" in content for content in self.messages[-1]["content"]):
                # Add toolResult message after to have a valid conversation
                logger.info(
                    "Agents latest message is toolUse, appending a toolResult message to have valid conversation."
                )
                tool_use_ids = [
                    content["toolUse"]["toolUseId"] for content in self.messages[-1]["content"] if "toolUse" in content
                ]
                await self._append_messages(
                    {
                        "role": "user",
                        "content": generate_missing_tool_result_content(tool_use_ids),
                    }
                )
            if isinstance(prompt, str):
                # String input - convert to user message
                messages = [{"role": "user", "content": [{"text": prompt}]}]
            elif isinstance(prompt, list):
                if len(prompt) == 0:
                    # Empty list
                    messages = []
                # Check if all item in input list are dictionaries
                elif all(isinstance(item, dict) for item in prompt):
                    # Check if all items are messages
                    if all(all(key in item for key in Message.__annotations__.keys()) for item in prompt):
                        # Messages input - add all messages to conversation
                        messages = cast(Messages, prompt)

                    # Check if all items are content blocks
                    elif all(any(key in ContentBlock.__annotations__.keys() for key in item) for item in prompt):
                        # Treat as List[ContentBlock] input - convert to user message
                        # This allows invalid structures to be passed through to the model
                        messages = [{"role": "user", "content": cast(list[ContentBlock], prompt)}]
        else:
            messages = []
        if messages is None:
            raise ValueError("Input prompt must be of type: `str | list[Contentblock] | Messages | None`.")
        return messages

    def _start_agent_trace_span(self, messages: Messages) -> trace_api.Span:
        """Starts a trace span for the agent.

        Args:
            messages: The input messages.
        """
        model_id = self.model.config.get("model_id") if hasattr(self.model, "config") else None
        return self.tracer.start_agent_span(
            messages=messages,
            agent_name=self.name,
            model_id=model_id,
            tools=self.tool_names,
            system_prompt=self.system_prompt,
            custom_trace_attributes=self.trace_attributes,
            tools_config=self.tool_registry.get_all_tools_config(),
        )

    def _end_agent_trace_span(
        self,
        response: AgentResult | None = None,
        error: Exception | None = None,
    ) -> None:
        """Ends a trace span for the agent.

        Args:
            span: The span to end.
            response: Response to record as a trace attribute.
            error: Error to record as a trace attribute.
        """
        if self.trace_span:
            trace_attributes: dict[str, Any] = {
                "span": self.trace_span,
            }

            if response:
                trace_attributes["response"] = response
            if error:
                trace_attributes["error"] = error

            self.tracer.end_agent_span(**trace_attributes)

    def _initialize_system_prompt(
        self, system_prompt: str | list[SystemContentBlock] | None
    ) -> tuple[str | None, list[SystemContentBlock] | None]:
        """Initialize system prompt fields from constructor input.

        Maintains backwards compatibility by keeping system_prompt as str when string input
        provided, avoiding breaking existing consumers.

        Maps system_prompt input to both string and content block representations:
        - If string: system_prompt=string, _system_prompt_content=[{text: string}]
        - If list with text elements: system_prompt=concatenated_text, _system_prompt_content=list
        - If list without text elements: system_prompt=None, _system_prompt_content=list
        - If None: system_prompt=None, _system_prompt_content=None
        """
        if isinstance(system_prompt, str):
            return system_prompt, [{"text": system_prompt}]
        elif isinstance(system_prompt, list):
            # Concatenate all text elements for backwards compatibility, None if no text found
            text_parts = [block["text"] for block in system_prompt if "text" in block]
            system_prompt_str = "\n".join(text_parts) if text_parts else None
            return system_prompt_str, system_prompt
        else:
            return None, None

    async def _append_messages(self, *messages: Message) -> None:
        """Appends messages to history and invoke the callbacks for the MessageAddedEvent."""
        for message in messages:
            self.messages.append(message)
            await self.hooks.invoke_callbacks_async(MessageAddedEvent(agent=self, message=message))

    def _redact_user_content(self, content: list[ContentBlock], redact_message: str) -> list[ContentBlock]:
        """Redact user content preserving toolResult blocks.

        Args:
            content: content blocks to be redacted
            redact_message: redact message to be replaced

        Returns:
            Redacted content, as follows:
            - if the message contains at least a toolResult block,
                all toolResult blocks(s) are kept, redacting only the result content;
            - otherwise, the entire content of the message is replaced
                with a single text block with the redact message.
        """
        redacted_content = []
        for block in content:
            if "toolResult" in block:
                block["toolResult"]["content"] = [{"text": redact_message}]
                redacted_content.append(block)

        if not redacted_content:
            # Text content is added only if no toolResult blocks were found
            redacted_content = [{"text": redact_message}]

        return redacted_content
     
    def _append_message(self, message: Message) -> None:
        """Appends a message to the agent's list of messages and invokes the callbacks for the MessageCreatedEvent."""
        self.messages.append(message)
        self.hooks.invoke_callbacks(MessageAddedEvent(agent=self, message=message))

    @property
    def sub_agents(self) -> dict[str, "Agent"]:
        """Get a copy of the registered sub-agents.

        Returns:
            Dictionary mapping agent names to Agent instances
        """
        return self._sub_agents.copy()

    def add_sub_agent(self, agent: "Agent") -> None:
        """Add a new sub-agent dynamically.

        Args:
            agent: Agent to add as a sub-agent

        Raises:
            ValueError: If agent validation fails
        """
        self._validate_sub_agents([agent])
        if agent.name not in self._sub_agents:
            self._sub_agents[agent.name] = agent
            self._generate_delegation_tools([agent])

            # Invoke hook for consistency with agent lifecycle
            if hasattr(self, "hooks"):
                try:
                    from ..hooks import SubAgentAddedEvent

                    self.hooks.invoke_callbacks(
                        SubAgentAddedEvent(agent=self, sub_agent=agent, sub_agent_name=agent.name)
                    )
                except ImportError:
                    # Hooks module not available, skip hook invocation
                    pass

    def remove_sub_agent(self, agent_name: str) -> bool:
        """Remove a sub-agent and its delegation tool.

        Args:
            agent_name: Name of the sub-agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_name in self._sub_agents:
            removed_agent = self._sub_agents[agent_name]
            del self._sub_agents[agent_name]

            # Remove delegation tool from registry
            tool_name = f"handoff_to_{agent_name.lower().replace('-', '_')}"
            if tool_name in self.tool_registry.registry:
                del self.tool_registry.registry[tool_name]

            # Invoke hook for cleanup
            if hasattr(self, "hooks"):
                try:
                    from ..hooks import SubAgentRemovedEvent

                    self.hooks.invoke_callbacks(
                        SubAgentRemovedEvent(agent=self, sub_agent_name=agent_name, removed_agent=removed_agent)
                    )
                except ImportError:
                    # Hooks module not available, skip hook invocation
                    pass

            return True
        return False

    def _validate_sub_agents(self, sub_agents: Optional[list["Agent"]]) -> None:
        """Validate sub-agent configuration.

        Args:
            sub_agents: List of sub-agents to validate

        Raises:
            ValueError: If sub-agent configuration is invalid
        """
        if not sub_agents:
            return

        # Check for unique names
        names = [agent.name for agent in sub_agents]
        if len(names) != len(set(names)):
            raise ValueError("Sub-agent names must be unique")

        # Check for circular references
        if self in sub_agents:
            raise ValueError("Agent cannot delegate to itself")

        # Check for duplicate names with existing tools
        existing_tools = self.tool_names
        for agent in sub_agents:
            tool_name = f"handoff_to_{agent.name.lower().replace('-', '_')}"
            if tool_name in existing_tools:
                raise ValueError(f"Tool name conflict: {tool_name} already exists")

        # Check for model compatibility if applicable
        if hasattr(self, "model") and hasattr(self.model, "config"):
            orchestrator_provider = self.model.config.get("provider")
            if orchestrator_provider:
                for agent in sub_agents:
                    if hasattr(agent, "model") and hasattr(agent.model, "config"):
                        sub_agent_provider = agent.model.config.get("provider")
                        if sub_agent_provider and sub_agent_provider != orchestrator_provider:
                            # Just a warning, not an error, as cross-provider delegation may be intentional
                            logger.warning(
                                "Model provider mismatch: %s uses %s, but sub-agent %s uses %s",
                                self.name,
                                orchestrator_provider,
                                agent.name,
                                sub_agent_provider,
                            )

    def _generate_delegation_tools(self, sub_agents: list["Agent"]) -> None:
        """Generate delegation tools for sub-agents.

        Args:
            sub_agents: List of sub-agents to generate tools for
        """
        from strands.tools import tool

        for sub_agent in sub_agents:
            tool_name = f"handoff_to_{sub_agent.name.lower().replace('-', '_')}"

            # Create closure configuration to avoid memory leak from capturing self
            delegation_config = {
                "orchestrator_name": self.name,
                "max_delegation_depth": getattr(self, "max_delegation_depth", None),
                "delegation_state_transfer": self.delegation_state_transfer,
                "delegation_message_transfer": self.delegation_message_transfer,
            }

            @tool(name=tool_name)
            def delegation_tool(
                message: str,
                context: dict[str, Any] | None = None,
                transfer_state: bool | None = None,
                transfer_messages: bool | None = None,
                target_agent: str = sub_agent.name,
                delegation_chain: list[str] | None = None,
                delegation_config: dict[str, Any] = delegation_config,
            ) -> dict[str, Any]:
                """Transfer control completely to specified sub-agent.

                This tool completely delegates the current request to the target agent.
                The orchestrator will terminate and the sub-agent's response will become
                the final response with no additional processing.

                Args:
                    message: Message to pass to the target agent
                    context: Additional context to transfer (optional)
                    transfer_state: Override the default state transfer behavior (optional)
                    transfer_messages: Override the default message transfer behavior (optional)
                    target_agent: Internal target agent identifier
                    delegation_chain: Internal delegation tracking
                    delegation_config: Delegation configuration (internal)

                Returns:
                    This tool raises AgentDelegationException and does not return normally.
                """
                current_depth = len(delegation_chain or [])
                max_depth = delegation_config["max_delegation_depth"]
                if max_depth and current_depth >= max_depth:
                    raise ValueError(f"Maximum delegation depth ({delegation_config['max_delegation_depth']}) exceeded")

                orchestrator_name = delegation_config["orchestrator_name"]
                state_transfer_default = delegation_config["delegation_state_transfer"]

                raise AgentDelegationException(
                    target_agent=target_agent,
                    message=message,
                    context=context or {},
                    delegation_chain=(delegation_chain or []) + [orchestrator_name],
                    transfer_state=transfer_state if transfer_state is not None else state_transfer_default,
                    transfer_messages=transfer_messages
                    if transfer_messages is not None
                    else delegation_config["delegation_message_transfer"],
                )

            agent_description = sub_agent.description or f"Specialized agent named {sub_agent.name}"
            capabilities_hint = ""
            if hasattr(sub_agent, "tools") and sub_agent.tools:
                tool_names = [
                    getattr(tool, "tool_name", getattr(tool, "__name__", str(tool))) for tool in sub_agent.tools[:3]
                ]  # Show first 3 tools as hint
                if tool_names:
                    capabilities_hint = f" Capabilities include: {', '.join(tool_names)}."

            # Concise tool docstring to avoid prompt bloat
            delegation_tool.__doc__ = (
                f"Delegate to {sub_agent.name} ({agent_description}).{capabilities_hint}\n"
                f"Transfers control completely - orchestrator terminates and "
                f"{sub_agent.name}'s response becomes final.\n\n"
                f"Use for: {agent_description.lower()}.\n"
                f"Args:\n"
                f"    message: Message for {sub_agent.name} (required)\n"
                f"    context: Additional context (optional)\n"
                f"    transfer_state: Transfer orchestrator.state (optional)\n"
                f"    transfer_messages: Transfer conversation history (optional)\n"
                f"    target_agent: Internal identifier (hidden)\n"
                f"    delegation_chain: Delegation tracking (hidden)\n"
                f"    delegation_config: Delegation configuration (internal)"
            )

            # Set JSON schema for better validation and model understanding
            # DecoratedFunctionTool doesn't have __schema__ by default, but Python allows
            # setting arbitrary attributes dynamically
            delegation_tool.__schema__ = {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": f"Message to pass to {sub_agent.name}"},
                    "context": {"type": ["object", "null"], "description": "Additional context to transfer"},
                    "transfer_state": {
                        "type": ["boolean", "null"],
                        "description": "Whether to transfer orchestrator.state",
                    },
                    "transfer_messages": {
                        "type": ["boolean", "null"],
                        "description": "Whether to transfer conversation history",
                    },
                    "target_agent": {"type": "string", "description": "Internal target agent identifier"},
                    "delegation_chain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Internal delegation tracking",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            }

            # Register the tool
            self.tool_registry.register_tool(delegation_tool)


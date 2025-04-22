# Copyright (c) Microsoft. All rights reserved.


import asyncio
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable
from functools import partial

from autogen_core import AgentRuntime, MessageContext, TopicId, TypeSubscription, message_handler
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationActorBase, OrchestrationBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_function_from_method import KernelFunctionFromMethod
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


class HandoffConnection(KernelBaseModel):
    """A model representing a handoff connection."""

    agent_name: str
    description: str


class HandoffStartMessage(KernelBaseModel):
    """A start message type to kick off a handoff group chat."""

    body: ChatMessageContent


class HandoffEndMessage(KernelBaseModel):
    """A message to end a handoff group chat."""

    body: ChatMessageContent


class HandoffRequestMessage(KernelBaseModel):
    """A request message type for agents in a handoff group chat."""

    agent_name: str


class HandoffResponseMessage(KernelBaseModel):
    """A response message type from agents in a handoff group chat."""

    body: ChatMessageContent


class HandoffResetMessage(KernelBaseModel):
    """A message to reset a participant's chat history in a handoff group."""

    pass


TExternalIn = TypeVar("TExternalIn", default=HandoffStartMessage)
TExternalOut = TypeVar("TExternalOut", default=HandoffEndMessage)


class HandoffOrchestrationActor(
    OrchestrationActorBase[
        TExternalIn,
        HandoffStartMessage,
        HandoffEndMessage,
        TExternalOut,
    ],
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    def __init__(
        self,
        internal_topic_type: str,
        input_transition: Callable[[TExternalIn], Awaitable[HandoffStartMessage] | HandoffStartMessage],
        output_transition: Callable[[HandoffEndMessage], Awaitable[TExternalOut] | TExternalOut],
        *,
        initial_agent_name: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            input_transition (Callable): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable): A function that transforms the internal output message to the external
                output message.
            initial_agent_name (str): The name of the agent that will receive the first message.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message
                to.
            result_callback: A function that is called when the result is available.
        """
        self._initial_agent_name = initial_agent_name

        super().__init__(
            internal_topic_type=internal_topic_type,
            input_transition=input_transition,
            output_transition=output_transition,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )

    @override
    async def _handle_orchestration_input_message(
        self,
        # The following does not validate LSP because Python doesn't recognize the generic type
        message: HandoffStartMessage,  # type: ignore
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        await self.publish_message(
            HandoffResponseMessage(body=message.body),
            TopicId(self._internal_topic_type, self.id.key),
        )
        await self.publish_message(
            HandoffRequestMessage(agent_name=self._initial_agent_name),
            TopicId(self._internal_topic_type, self.id.key),
        )

    @override
    async def _handle_orchestration_output_message(
        self,
        message: HandoffEndMessage,
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration output message.")
        if inspect.isawaitable(self._output_transition):
            external_output_message: TExternalOut = await self._output_transition(message)
        else:
            external_output_message: TExternalOut = self._output_transition(message)  # type: ignore[no-redef]

        if self._external_topic_type:
            logger.debug(f"Relaying message to external topic: {self._external_topic_type}")
            await self.publish_message(
                external_output_message,
                TopicId(self._external_topic_type, self.id.key),
            )
        if self._direct_actor_type:
            logger.debug(f"Relaying message directly to actor: {self._direct_actor_type}")
            target_actor_id = await self.runtime.get(self._direct_actor_type)
            await self.send_message(
                external_output_message,
                target_actor_id,
            )
        if self._result_callback:
            self._result_callback(external_output_message)


HANDOFF_PLUGIN_NAME = "Handoff"


async def _handoff_function_filter(context: AutoFunctionInvocationContext, next):
    await next(context)
    if context.function.plugin_name == HANDOFF_PLUGIN_NAME:
        # Terminate when ever an agent decides to handoff the conversation to another agent
        context.terminate = True


class HandoffAgentActor(AgentActorBase):
    """An agent actor that handles handoff messages in a group chat."""

    def __init__(
        self,
        agent: Agent,
        internal_topic_type: str,
        orchestration_actor_type: str,
        handoff_topic_connections: list[HandoffConnection],
    ) -> None:
        """Initialize the handoff agent actor."""
        super().__init__(agent=agent, internal_topic_type=internal_topic_type)

        self._agent_thread: AgentThread | None = None
        # Chat history to temporarily store messages before the agent thread is created
        self._chat_history: ChatHistory = ChatHistory()
        self._handoff_topic_connections = handoff_topic_connections
        self._orchestration_actor_type = orchestration_actor_type

        self._kernel = agent.kernel.model_copy()
        self._add_handoff_functions()

    def _add_handoff_functions(self):
        """Add handoff functions to the agent's kernel."""
        functions: list[KernelFunctionFromMethod] = []
        for connection in self._handoff_topic_connections:
            function_name = f"transfer_to_{connection.agent_name}"
            function_description = connection.description
            parameters = []
            return_parameter = KernelParameterMetadata(
                name="return",
                description="",
                default_value=None,
                type_="None",
                type_object=None,
                is_required=False,
            )
            function_metadata = KernelFunctionMetadata(
                name=function_name,
                description=function_description,
                parameters=parameters,
                return_parameter=return_parameter,
                is_prompt=False,
                is_asynchronous=True,
                plugin_name=HANDOFF_PLUGIN_NAME,
                additional_properties={},
            )
            functions.append(
                KernelFunctionFromMethod.model_construct(
                    metadata=function_metadata,
                    method=partial(self._handoff_to_agent, connection.agent_name),
                )
            )
        functions.append(KernelFunctionFromMethod(self._end_task, plugin_name=HANDOFF_PLUGIN_NAME))
        self._kernel.add_plugin(plugin=KernelPlugin(name=HANDOFF_PLUGIN_NAME, functions=functions))
        self._kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, _handoff_function_filter)

    async def _handoff_to_agent(self, agent_name: str) -> None:
        """Handoff the conversation to another agent."""
        logger.debug(f"{self.id}: Handoff to agent {agent_name}.")
        await self.publish_message(
            HandoffRequestMessage(agent_name=agent_name),
            TopicId(self._internal_topic_type, self.id.key),
        )

    @kernel_function(description="End the task with a summary when needed.")
    async def _end_task(self, task_summary: str) -> None:
        """End the task with a summary."""
        logger.debug(f"{self.id}: Ending task with summary: {task_summary}")
        target_actor_id = await self.runtime.get(self._orchestration_actor_type)
        await self.send_message(
            HandoffEndMessage(body=ChatMessageContent(role=AuthorRole.ASSISTANT, content=task_summary)),
            target_actor_id,
        )

    @message_handler
    async def _handle_reset_message(self, message: HandoffResetMessage, cts: MessageContext) -> None:
        """Handle a reset message to clear the chat history."""
        if self._agent_thread is not None:
            await self._agent_thread.delete()
            self._agent_thread = None
        self._chat_history.clear()

    @message_handler
    async def _handle_response_message(self, message: HandoffResponseMessage, cts: MessageContext) -> None:
        """Handle a response message from an agent in the handoff group."""
        logger.debug(f"{self.id}: Received handoff response message.")
        if self._agent_thread is not None:
            if message.body.role != AuthorRole.USER:
                await self._agent_thread.on_new_message(
                    ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"Transferred to {message.body.name}",
                    )
                )
            await self._agent_thread.on_new_message(message.body)
        else:
            if message.body.role != AuthorRole.USER:
                self._chat_history.add_message(
                    ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"Transferred to {message.body.name}",
                    )
                )
            self._chat_history.add_message(message.body)

    @message_handler
    async def _handle_request_message(self, message: HandoffRequestMessage, cts: MessageContext) -> None:
        """Handle a request message from an agent in the handoff group."""
        if message.agent_name != self._agent.name:
            return
        logger.debug(f"{self.id}: Received handoff request message.")
        if self._agent_thread is None:
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
                )
            )
            responses = self._agent.invoke(
                messages=self._chat_history.messages,
                kernel=self._kernel,
            )
        else:
            responses = self._agent.invoke(
                messages=ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
                ),
                thread=self._agent_thread,
                kernel=self._kernel,
            )

        async for response_item in responses:
            if self._agent_thread is None:
                self._agent_thread = response_item.thread

            if response_item.message.role == AuthorRole.ASSISTANT:
                # The response can potentially be a TOOL message since we have added
                # a filter which will terminate the conversation when a function from
                # the handoff plugin is called. And we don't want to publish that message.
                # So we only publish if the response is an ASSISTANT message.
                logger.debug(f"{self.id} responded with: {response_item.message.content}")

                await self.publish_message(
                    HandoffResponseMessage(body=response_item.message),
                    TopicId(self._internal_topic_type, self.id.key),
                )


class HandoffOrchestration(
    OrchestrationBase[
        TExternalIn,
        HandoffStartMessage,
        HandoffEndMessage,
        TExternalOut,
    ]
):
    """An orchestration class for managing handoff agents in a group chat."""

    def __init__(
        self,
        members: list[Agent | OrchestrationBase],
        handoffs: dict[str, list[HandoffConnection]],
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[HandoffStartMessage] | HandoffStartMessage] | None = None,
        output_transition: Callable[[HandoffEndMessage], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the handoff orchestration.

        Args:
            members (list[Agent | OrchestrationBase]): A list of agents or orchestrations that are part of the
                handoff group. This first agent in the list will be the one that receives the first message.
            handoffs (dict[str, list[HandoffConnection]]): A dictionary mapping agent names to their handoff
                connections.
            name (str | None): The name of the orchestration.
            description (str | None): The description of the orchestration.
            input_transition (Callable | None): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable | None): A function that transforms the internal output message to the external
                output message.
        """
        self._handoffs = handoffs

        for member in members:
            if not isinstance(member, Agent):
                raise ValueError(f"All members must be of type Agent in HandoffOrchestration, but got {type(member)}")

        super().__init__(
            members=members,
            name=name,
            description=description,
            input_transition=input_transition,
            output_transition=output_transition,
        )

    @override
    async def _start(
        self,
        task: str | HandoffStartMessage | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the concurrent pattern."""
        if isinstance(task, str):
            message = HandoffStartMessage(
                body=ChatMessageContent(
                    role=AuthorRole.USER,
                    content=task,
                )
            )
        elif isinstance(task, ChatMessageContent):
            message = HandoffStartMessage(body=task)

        target_actor_id = await runtime.get(self._get_orchestration_actor_type(internal_topic_type))
        await runtime.send_message(message, target_actor_id)

    @override
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> str:
        """Register the actors and orchestrations with the runtime and add the required subscriptions."""
        await self._register_members(runtime, internal_topic_type)
        await self._register_orchestration_actor(
            runtime,
            internal_topic_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )
        await self._add_subscriptions(runtime, internal_topic_type)

        return self._get_orchestration_actor_type(internal_topic_type)

    async def _register_members(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Register the members with the runtime."""
        member_names = {m.name for m in self._members if isinstance(m, Agent)}
        for member in self._members:
            if isinstance(member, Agent):
                handoff_connections = self._handoffs.get(member.name, [])
                for connection in handoff_connections:
                    if connection.agent_name not in member_names:
                        logger.warning(f"Agent {connection.agent_name} is not a member of the handoff group.")

                await HandoffAgentActor.register(
                    runtime,
                    self._get_agent_actor_type(member, internal_topic_type),
                    lambda member=member, handoff_connections=handoff_connections: HandoffAgentActor(
                        member,
                        internal_topic_type,
                        self._get_orchestration_actor_type(internal_topic_type),
                        handoff_connections,
                    ),
                )

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        await HandoffOrchestrationActor[self.t_external_in, self.t_external_in].register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: HandoffOrchestrationActor[self.t_external_in, self.t_external_out](
                internal_topic_type,
                self._input_transition,
                self._output_transition,
                initial_agent_name=self._members[0].name,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
        )

    async def _add_subscriptions(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Add subscriptions to the runtime."""
        subscriptions: list[TypeSubscription] = [
            TypeSubscription(
                internal_topic_type,
                self._get_agent_actor_type(member, internal_topic_type),
            )
            for member in self._members
            if isinstance(member, Agent)
        ]
        subscriptions.append(
            TypeSubscription(
                internal_topic_type,
                self._get_orchestration_actor_type(internal_topic_type),
            )
        )

        await asyncio.gather(*[runtime.add_subscription(subscription) for subscription in subscriptions])

    def _get_agent_actor_type(self, agent: Agent | str, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        if isinstance(agent, Agent):
            agent = agent.name
        return f"{agent}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{HandoffOrchestrationActor.__name__}_{internal_topic_type}"

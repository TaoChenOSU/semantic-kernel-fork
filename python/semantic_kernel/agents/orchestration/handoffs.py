# Copyright (c) Microsoft. All rights reserved.


import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from functools import partial

from autogen_core import AgentRuntime, CancellationToken, MessageContext, TopicId, TypeSubscription, message_handler

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import (
    DefaultExternalTypeAlias,
    OrchestrationBase,
    TExternalIn,
    TExternalOut,
)
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

# region Messages and Types


class HandoffConnection(KernelBaseModel):
    """A model representing a handoff connection."""

    agent_name: str
    description: str


class HandoffStartMessage(KernelBaseModel):
    """A start message type to kick off a handoff group chat."""

    body: DefaultExternalTypeAlias


class HandoffRequestMessage(KernelBaseModel):
    """A request message type for agents in a handoff group chat."""

    agent_name: str


class HandoffResponseMessage(KernelBaseModel):
    """A response message type from agents in a handoff group chat."""

    body: ChatMessageContent


HANDOFF_PLUGIN_NAME = "Handoff"


# endregion Messages and Types

# region HandoffAgentActor


class HandoffAgentActor(AgentActorBase):
    """An agent actor that handles handoff messages in a group chat."""

    def __init__(
        self,
        agent: Agent,
        internal_topic_type: str,
        handoff_topic_connections: list[HandoffConnection],
        result_callback: Callable[[DefaultExternalTypeAlias], Awaitable[None]] | None = None,
        observer: Callable[[str | DefaultExternalTypeAlias], Awaitable[None] | None] | None = None,
    ) -> None:
        """Initialize the handoff agent actor."""
        self._handoff_topic_connections = handoff_topic_connections
        self._result_callback = result_callback

        self._kernel = agent.kernel.model_copy()
        self._add_handoff_functions()

        super().__init__(agent=agent, internal_topic_type=internal_topic_type, observer=observer)

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
        self._kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, self._handoff_function_filter)

    async def _handoff_to_agent(self, agent_name: str) -> None:
        """Handoff the conversation to another agent."""
        logger.debug(f"{self.id}: Handoff to agent {agent_name}.")
        await self.publish_message(
            HandoffRequestMessage(agent_name=agent_name),
            TopicId(self._internal_topic_type, self.id.key),
        )

    async def _handoff_function_filter(self, context: AutoFunctionInvocationContext, next):
        """A filter to terminate an agent when it decides to handoff the conversation to another agent."""
        await next(context)
        if context.function.plugin_name == HANDOFF_PLUGIN_NAME:
            context.terminate = True

    @kernel_function(description="End the task with a summary when needed.")
    async def _end_task(self, task_summary: str) -> None:
        """End the task with a summary."""
        logger.debug(f"{self.id}: Ending task with summary: {task_summary}")
        if self._result_callback:
            await self._result_callback(ChatMessageContent(role=AuthorRole.ASSISTANT, content=task_summary))

    @message_handler
    async def _handle_start_message(self, message: HandoffStartMessage, cts: MessageContext) -> None:
        logger.debug(f"{self.id}: Received handoff start message.")
        if isinstance(message.body, ChatMessageContent):
            if self._agent_thread:
                await self._agent_thread.on_new_message(message.body)
            else:
                self._chat_history.add_message(message.body)
        elif isinstance(message.body, list) and all(isinstance(m, ChatMessageContent) for m in message.body):
            for m in message.body:
                if self._agent_thread:
                    await self._agent_thread.on_new_message(m)
                else:
                    self._chat_history.add_message(m)
        else:
            raise ValueError(f"Invalid message body type: {type(message.body)}. Expected {DefaultExternalTypeAlias}.")

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

            await self._notify_observer(response_item.message)

            if response_item.message.role == AuthorRole.ASSISTANT:
                # The response can potentially be a TOOL message from the Handoff plugin
                # since we have added a filter which will terminate the conversation when
                # a function from the handoff plugin is called. And we don't want to publish
                # that message. So we only publish if the response is an ASSISTANT message.
                logger.debug(f"{self.id} responded with: {response_item.message.content}")

                await self.publish_message(
                    HandoffResponseMessage(body=response_item.message),
                    TopicId(self._internal_topic_type, self.id.key),
                    cancellation_token=cts.cancellation_token,
                )


# endregion HandoffAgentActor

# region HandoffOrchestration


class HandoffOrchestration(OrchestrationBase[TExternalIn, TExternalOut]):
    """An orchestration class for managing handoff agents in a group chat."""

    def __init__(
        self,
        members: list[Agent | OrchestrationBase],
        handoffs: dict[str, list[HandoffConnection]],
        name: str | None = None,
        description: str | None = None,
        input_transform: Callable[[TExternalIn], Awaitable[DefaultExternalTypeAlias] | DefaultExternalTypeAlias]
        | None = None,
        output_transform: Callable[[DefaultExternalTypeAlias], Awaitable[TExternalOut] | TExternalOut] | None = None,
        observer: Callable[[str | DefaultExternalTypeAlias], Awaitable[None] | None] | None = None,
    ) -> None:
        """Initialize the handoff orchestration.

        Args:
            members (list[Agent | OrchestrationBase]): A list of agents or orchestrations that are part of the
                handoff group. This first agent in the list will be the one that receives the first message.
            handoffs (dict[str, list[HandoffConnection]]): A dictionary mapping agent names to their handoff
                connections.
            name (str | None): The name of the orchestration.
            description (str | None): The description of the orchestration.
            input_transform (Callable | None): A function that transforms the external input message.
            output_transform (Callable | None): A function that transforms the internal output message.
            observer (Callable | None): A function that is called when a response is produced by the agents.
        """
        self._handoffs = handoffs

        super().__init__(
            members=members,
            name=name,
            description=description,
            input_transform=input_transform,
            output_transform=output_transform,
            observer=observer,
        )

    @override
    async def _start(
        self,
        task: DefaultExternalTypeAlias,
        runtime: AgentRuntime,
        internal_topic_type: str,
        cancellation_token: CancellationToken,
    ) -> None:
        """Start the handoff pattern.

        This ensures that all initial messages are sent to the individual actors
        and processed before the group chat begins. It's important because if the
        manager actor processes its start message too quickly (or other actors are
        too slow), it might send a request to the next agent before the other actors
        have the necessary context.
        """

        async def send_start_message(agent: Agent) -> None:
            target_actor_id = await runtime.get(self._get_agent_actor_type(agent, internal_topic_type))
            await runtime.send_message(
                HandoffStartMessage(body=task),
                target_actor_id,
                cancellation_token=task.cancellation_token,
            )

        await asyncio.gather(*[send_start_message(agent) for agent in self._members])

        # Send the handoff request message to the first agent in the list
        target_actor_id = await runtime.get(self._get_agent_actor_type(self._members[0], internal_topic_type))
        await runtime.send_message(
            HandoffRequestMessage(agent_name=self._members[0].name),
            target_actor_id,
            cancellation_token=cancellation_token,
        )

    @override
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        """Register the actors and orchestrations with the runtime and add the required subscriptions."""
        await self._register_members(runtime, internal_topic_type, result_callback)
        await self._add_subscriptions(runtime, internal_topic_type)

    async def _register_members(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        result_callback: Callable[[DefaultExternalTypeAlias], Awaitable[None]] | None = None,
    ) -> None:
        """Register the members with the runtime."""
        member_names = {m.name for m in self._members if isinstance(m, Agent)}
        for member in self._members:
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
                    handoff_connections,
                    result_callback=result_callback,
                    observer=self._observer,
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
        ]

        await asyncio.gather(*[runtime.add_subscription(subscription) for subscription in subscriptions])

    def _get_agent_actor_type(self, agent: Agent, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{agent.name}_{internal_topic_type}"


# endregion HandoffOrchestration

# Copyright (c) Microsoft. All rights reserved.

import inspect
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Union

from autogen_core import AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationActorBase, OrchestrationBase
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


class SequentialRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class SequentialResultMessage(KernelBaseModel):
    """A result message type for concurrent agents."""

    body: ChatMessageContent


TExternalIn = TypeVar("TExternalIn", default=SequentialRequestMessage)
TExternalOut = TypeVar("TExternalOut", default=SequentialResultMessage)


class SequentialOrchestrationActor(
    OrchestrationActorBase[
        TExternalIn,
        SequentialRequestMessage,
        SequentialResultMessage,
        TExternalOut,
    ],
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    def __init__(
        self,
        internal_topic_type: str,
        input_transition: Callable[[TExternalIn], Awaitable[SequentialRequestMessage] | SequentialRequestMessage],
        output_transition: Callable[[SequentialResultMessage], Awaitable[TExternalOut] | TExternalOut],
        *,
        initial_actor_type: str,
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
            initial_actor_type (str): The actor type of the first actor in the sequence.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message
                to.
            result_callback: A function that is called when the result is available.
        """
        self._initial_actor_type = initial_actor_type

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
        message: SequentialRequestMessage,  # type: ignore
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        logger.debug(f"Relaying message to agent: {self._initial_actor_type}")
        target_actor_id = await self.runtime.get(self._initial_actor_type)
        await self.send_message(message, target_actor_id)

    @override
    async def _handle_orchestration_output_message(
        self,
        message: SequentialResultMessage,
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
            await self.send_message(external_output_message, target_actor_id)
        if self._result_callback:
            self._result_callback(external_output_message)


class SequentialAgentActor(AgentActorBase):
    """A agent actor for sequential agents that process tasks."""

    def __init__(self, agent: Agent, internal_topic_type: str, next_agent_type: str) -> None:
        """Initialize the agent actor."""
        self._next_agent_type = next_agent_type
        super().__init__(agent=agent, internal_topic_type=internal_topic_type)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(f"Sequential actor (Actor ID: {self.id}; Agent name: {self._agent.name}) started processing...")

        response = await self._agent.get_response(messages=message.body)

        logger.debug(f"Sequential actor (Actor ID: {self.id}; Agent name: {self._agent.name}) finished processing.")

        target_actor_id = await self.runtime.get(self._next_agent_type)
        await self.send_message(SequentialRequestMessage(body=response.message), target_actor_id)


class CollectionActor(RoutedAgent):
    """A agent container for collection results from the last agent in the sequence."""

    def __init__(self, description: str, orchestration_agent_type: str) -> None:
        """Initialize the collection agent container."""
        self._orchestration_agent_type = orchestration_agent_type

        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        target_actor_id = await self.runtime.get(self._orchestration_agent_type)
        await self.send_message(SequentialResultMessage(body=message.body), target_actor_id)


class SequentialOrchestration(
    OrchestrationBase[
        TExternalIn,
        SequentialRequestMessage,
        SequentialResultMessage,
        TExternalOut,
    ],
):
    """A sequential multi-agent pattern orchestration."""

    def __init__(
        self,
        members: list[Union[Agent, "OrchestrationBase"]],
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[SequentialRequestMessage] | SequentialRequestMessage]
        | None = None,
        output_transition: Callable[[SequentialResultMessage], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            members (list[Union[Agent, OrchestrationBase]]): The list of agents or orchestrations to be used.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transition (Callable): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable): A function that transforms the internal output message to the external
                output message.
        """
        super().__init__(
            members,
            name=name,
            description=description,
            input_transition=input_transition,
            output_transition=output_transition,
        )

    @override
    async def _start(
        self,
        task: SequentialRequestMessage | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the sequential pattern."""
        if isinstance(task, ChatMessageContent):
            message = SequentialRequestMessage(body=task)

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
        """Register the actors and orchestrations with the runtime and add the required subscriptions.

        Args:
            runtime (AgentRuntime): The agent runtime.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
                Since the sequential orchestration doesn't broadcast messages internally, this is only used to
                uniquely identify the orchestration.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            result_callback: A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        initial_actor_type = await self._register_members(runtime, internal_topic_type)
        await self._register_orchestration_actor(
            runtime,
            internal_topic_type,
            initial_actor_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )
        await self._register_collection_actor(runtime, internal_topic_type)
        await self._add_subscriptions(runtime, internal_topic_type, external_topic_type)

        return self._get_orchestration_actor_type(internal_topic_type)

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        initial_actor_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        """Register the orchestration actor."""
        await SequentialOrchestrationActor[self.t_external_in, self.t_external_in].register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: SequentialOrchestrationActor[self.t_external_in, self.t_external_in](
                internal_topic_type,
                self._input_transition,
                self._output_transition,
                initial_actor_type=initial_actor_type,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
        )

    async def _register_members(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> str:
        """Register the members.

        The members will be registered in the reverse order so that the actor type of the next worker
        is available when the current worker is registered. This is important for the sequential
        orchestration, where actors need to know its next actor type to send the message to.

        Args:
            runtime (AgentRuntime): The agent runtime.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.

        Returns:
            str: The first actor type in the sequence.
        """
        next_actor_type = self._get_collection_actor_type(internal_topic_type)
        for index, worker in enumerate(reversed(self._members)):
            if isinstance(worker, Agent):
                await SequentialAgentActor.register(
                    runtime,
                    self._get_agent_actor_type(worker, internal_topic_type),
                    lambda worker=worker, next_actor_type=next_actor_type: SequentialAgentActor(  # type: ignore[misc]
                        worker,
                        internal_topic_type,
                        next_agent_type=next_actor_type,
                    ),
                )
                logger.debug(
                    f"Registered agent actor of type {self._get_agent_actor_type(worker, internal_topic_type)}"
                )
                next_actor_type = self._get_agent_actor_type(worker, internal_topic_type)
            elif isinstance(worker, OrchestrationBase):
                worker_orchestration_actor_type = await worker.prepare(
                    runtime,
                    direct_actor_type=next_actor_type,
                )
                logger.debug(f"Registered orchestration actor of type {worker_orchestration_actor_type}")
                next_actor_type = worker_orchestration_actor_type
            else:
                raise TypeError(f"Unsupported node type: {type(worker)}")

        return next_actor_type

    async def _register_collection_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Register the collection actor."""
        await CollectionActor.register(
            runtime,
            self._get_collection_actor_type(internal_topic_type),
            lambda: CollectionActor(
                description="An internal agent that is responsible for collection results",
                orchestration_agent_type=self._get_orchestration_actor_type(internal_topic_type),
            ),
        )

    async def _add_subscriptions(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
    ) -> None:
        """Add subscriptions to the runtime."""
        if external_topic_type:
            await runtime.add_subscription(
                TypeSubscription(
                    external_topic_type,
                    self._get_orchestration_actor_type(internal_topic_type),
                )
            )

    def _get_agent_actor_type(self, agent: Agent, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{agent.name}_{internal_topic_type}"

    def _get_collection_actor_type(self, internal_topic_type: str) -> str:
        """Get the collection actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{CollectionActor.__name__}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{SequentialOrchestrationActor.__name__}_{internal_topic_type}"

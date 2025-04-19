# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable

from autogen_core import AgentId, AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
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


class ConcurrentRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentResponseMessage(KernelBaseModel):
    """A response message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentResultMessage(KernelBaseModel):
    """A result message type for concurrent agents."""

    body: dict[str, ChatMessageContent]


TExternalIn = TypeVar("TExternalIn", default=ConcurrentRequestMessage)
TExternalOut = TypeVar("TExternalOut", default=ConcurrentResultMessage)


class ConcurrentOrchestrationActor(
    OrchestrationActorBase[
        TExternalIn,
        ConcurrentRequestMessage,
        ConcurrentResultMessage,
        TExternalOut,
    ],
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    @override
    async def _handle_orchestration_input_message(
        self,
        # The following does not validate LSP because Python doesn't recognize the generic type
        message: ConcurrentRequestMessage,  # type: ignore
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        logger.debug(f"Broadcasting message to topic {self._internal_topic_type}.")
        await self.publish_message(message, TopicId(self._internal_topic_type, "default"))

    @override
    async def _handle_orchestration_output_message(
        self,
        message: ConcurrentResultMessage,
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
                TopicId(self._external_topic_type, "default"),
            )
        if self._direct_actor_type:
            logger.debug(f"Relaying message directly to actor: {self._direct_actor_type}")
            await self.send_message(
                external_output_message,
                AgentId(self._direct_actor_type, "default"),
            )
        if self._result_callback:
            self._result_callback(external_output_message)


class ConcurrentAgentActor(AgentActorBase):
    """A agent actor for concurrent agents that process tasks."""

    def __init__(self, agent: Agent, collection_agent_type: str) -> None:
        """Initialize the agent actor."""
        self._collection_agent_type = collection_agent_type
        super().__init__(agent=agent)

    @message_handler
    async def _handle_message(self, message: ConcurrentRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(f"Concurrent actor (Actor ID: {self.id}; Agent name: {self._agent.name}) started processing...")

        response = await self._agent.get_response(messages=message.body)

        logger.debug(f"Concurrent actor (Actor ID: {self.id}; Agent name: {self._agent.name}) finished processing.")

        await self.send_message(
            ConcurrentResponseMessage(body=response.message),
            AgentId(
                type=self._collection_agent_type,
                key="default",
            ),
        )


class CollectionActor(RoutedAgent):
    """A agent container for collecting results from concurrent agents."""

    def __init__(self, description: str, expected_answer_count: int, orchestration_actor_type: str) -> None:
        """Initialize the collection agent container."""
        self._expected_answer_count = expected_answer_count
        self._orchestration_actor_type = orchestration_actor_type
        self._results: dict[str, ChatMessageContent] = {}
        self._lock = asyncio.Lock()

        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: ConcurrentResponseMessage, ctx: MessageContext) -> None:
        async with self._lock:
            self._results[f"{ctx.sender.type}_{ctx.sender.key}"] = message.body

        if len(self._results) == self._expected_answer_count:
            logger.debug(f"Collection actor (Actor ID: {self.id}) finished processing all responses.")
            await self.send_message(
                ConcurrentResultMessage(body=self._results),
                AgentId(
                    type=self._orchestration_actor_type,
                    key="default",
                ),
            )


class ConcurrentOrchestration(
    OrchestrationBase[
        TExternalIn,
        ConcurrentRequestMessage,
        ConcurrentResultMessage,
        TExternalOut,
    ],
):
    """A concurrent multi-agent pattern orchestration."""

    def __init__(
        self,
        workers: list[Agent | OrchestrationBase],
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[ConcurrentRequestMessage] | ConcurrentRequestMessage]
        | None = None,
        output_transition: Callable[[ConcurrentResultMessage], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            workers (list[Union[Agent, OrchestrationBase]]): The list of agents or orchestrations to be used.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transition (Callable): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable): A function that transforms the internal output message to the external
                output message.
        """
        super().__init__(
            workers,
            name=name,
            description=description,
            input_transition=input_transition,
            output_transition=output_transition,
        )

    @override
    async def _start(
        self,
        task: ConcurrentRequestMessage | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the concurrent pattern."""
        if isinstance(task, ChatMessageContent):
            message = ConcurrentRequestMessage(body=task)

        await runtime.send_message(
            message,
            AgentId(
                type=self._get_orchestration_actor_type(internal_topic_type),
                key="default",
            ),
        )

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
        await asyncio.gather(*[
            self._register_orchestration_actor(
                runtime,
                internal_topic_type,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
            self._register_workers(
                runtime,
                internal_topic_type,
            ),
            self._register_collection_actor(
                runtime,
                internal_topic_type,
            ),
            self._add_subscriptions(
                runtime,
                internal_topic_type,
            ),
        ])

        return self._get_orchestration_actor_type(internal_topic_type)

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        """Register the orchestration actor."""
        await ConcurrentOrchestrationActor[self.t_external_in, self.t_external_in].register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: ConcurrentOrchestrationActor[self.t_external_in, self.t_external_in](
                internal_topic_type,
                input_transition=self._input_transition,
                output_transition=self._output_transition,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
        )

    async def _register_workers(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Register the workers."""

        async def _internal_helper(worker: Agent | OrchestrationBase) -> None:
            if isinstance(worker, Agent):
                await ConcurrentAgentActor.register(
                    runtime,
                    self._get_agent_actor_type(worker, internal_topic_type),
                    lambda agent=worker: ConcurrentAgentActor(  # type: ignore[misc]
                        agent,
                        collection_agent_type=self._get_collection_actor_type(internal_topic_type),
                    ),
                )
            elif isinstance(worker, OrchestrationBase):
                await worker.prepare(
                    runtime,
                    external_topic_type=internal_topic_type,
                    direct_actor_type=self._get_collection_actor_type(internal_topic_type),
                )

        await asyncio.gather(*[_internal_helper(worker) for worker in self._workers])

    async def _register_collection_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        await CollectionActor.register(
            runtime,
            self._get_collection_actor_type(internal_topic_type),
            lambda: CollectionActor(
                description="An internal agent that is responsible for collection results",
                expected_answer_count=len(self._workers),
                orchestration_actor_type=self._get_orchestration_actor_type(internal_topic_type),
            ),
        )

    async def _add_subscriptions(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    internal_topic_type,
                    self._get_agent_actor_type(agent, internal_topic_type),
                )
            )
            for agent in self._workers
            if isinstance(
                agent, Agent
            )  # Only register agent actors since orchestrations will add their own subscriptions
        ])

    def _get_agent_actor_type(self, worker: Agent, internal_topic_type: str) -> str:
        """Get the container type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{worker.name}_{internal_topic_type}"

    def _get_collection_actor_type(self, internal_topic_type: str) -> str:
        """Get the collection agent type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{CollectionActor.__name__}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration agent type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{ConcurrentOrchestrationActor.__name__}_{internal_topic_type}"

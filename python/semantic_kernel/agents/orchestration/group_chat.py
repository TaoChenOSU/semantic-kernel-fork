# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from autogen_core import AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationActorBase, OrchestrationBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


class GroupChatStartMessage(KernelBaseModel):
    """A message to start a group chat."""

    body: ChatMessageContent


class GroupChatEndMessage(KernelBaseModel):
    """A message to end a group chat."""

    body: ChatMessageContent


class GroupChatRequestMessage(KernelBaseModel):
    """A request message type for agents in a group chat."""

    agent_name: str


class GroupChatResponseMessage(KernelBaseModel):
    """A response message type from agents in a group chat."""

    body: ChatMessageContent


class GroupChatResetMessage(KernelBaseModel):
    """A message to reset a participant's chat history in a group chat."""

    pass


TExternalIn = TypeVar("TExternalIn", default=GroupChatStartMessage)
TExternalOut = TypeVar("TExternalOut", default=GroupChatEndMessage)


class GroupChatOrchestrationActor(
    OrchestrationActorBase[
        TExternalIn,
        GroupChatStartMessage,
        GroupChatEndMessage,
        TExternalOut,
    ],
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    @override
    async def _handle_orchestration_input_message(
        self,
        # The following does not validate LSP because Python doesn't recognize the generic type
        message: GroupChatStartMessage,  # type: ignore
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        await self.publish_message(
            GroupChatResponseMessage(body=message.body),
            TopicId(self._internal_topic_type, self.id.key),
        )

    @override
    async def _handle_orchestration_output_message(
        self,
        message: GroupChatEndMessage,
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


class GroupChatAgentActor(AgentActorBase):
    """An agent actor that process messages in a group chat."""

    def __init__(self, agent: Agent, internal_topic_type: str):
        """Initialize the group chat agent container."""
        super().__init__(agent=agent, internal_topic_type=internal_topic_type)

        self._agent_thread: AgentThread | None = None
        # Chat history to temporarily store messages before the agent thread is created
        self._chat_history = ChatHistory()

    @message_handler
    async def _on_group_chat_reset(self, message: GroupChatResetMessage, ctx: MessageContext) -> None:
        self._chat_history.clear()
        if self._agent_thread:
            await self._agent_thread.delete()
            self._agent_thread = None

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        logger.debug(f"{self.id}: Received group chat response message.")
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
    async def _handle_request_message(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        if message.agent_name != self._agent.name:
            return

        logger.debug(f"{self.id}: Received group chat request message.")
        if self._agent_thread is None:
            # Add a user message to steer the agent to respond more closely to the instructions.
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
                )
            )
            response_item = await self._agent.get_response(messages=self._chat_history.messages)
            self._agent_thread = response_item.thread
        else:
            # Add a user message to steer the agent to respond more closely to the instructions.
            new_message = ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
            )
            response_item = await self._agent.get_response(messages=new_message, thread=self._agent_thread)

        logger.debug(f"{self.id} responded with {response_item.message.content}.")

        await self.publish_message(
            GroupChatResponseMessage(body=response_item.message),
            TopicId(self._internal_topic_type, self.id.key),
        )


class BoolWithReason(KernelBaseModel):
    """A class to represent a boolean value with a reason."""

    value: bool
    reason: str

    def __bool__(self) -> bool:
        """Return the boolean value."""
        return self.value


class StringWithReason(KernelBaseModel):
    """A class to represent a string value with a reason."""

    value: str
    reason: str


class ChatMessageContentWithReason(KernelBaseModel):
    """A class to represent an object value with a reason."""

    value: ChatMessageContent
    reason: str


class GroupChatManager(KernelBaseModel, ABC):
    """A group chat manager that manages the flow of a group chat."""

    current_round: int = 0
    max_rounds: int | None = None

    user_input_func: Callable[[ChatHistory], Awaitable[str]] | None = None

    @abstractmethod
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should terminate.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringWithReason:
        """Select the next agent to speak.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ChatMessageContentWithReason:
        """Filter the results of the group chat.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        raise NotImplementedError


class RoundRobinGroupChatManager(GroupChatManager):
    """A round-robin group chat manager."""

    current_index: int = 0

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input."""
        return BoolWithReason(
            value=False,
            reason="The default round-robin group chat manager does not request user input.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None:
            return BoolWithReason(
                value=self.current_round > self.max_rounds,
                reason="Maximum rounds reached."
                if self.current_round > self.max_rounds
                else "Not reached maximum rounds.",
            )
        return BoolWithReason(value=False, reason="No maximum rounds set.")

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringWithReason:
        """Select the next agent to speak."""
        next_agent = list(participant_descriptions.keys())[self.current_index]
        self.current_index = (self.current_index + 1) % len(participant_descriptions)
        self.current_round += 1
        return StringWithReason(value=next_agent, reason="Round-robin selection.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ChatMessageContentWithReason:
        """Filter the results of the group chat."""
        return ChatMessageContentWithReason(
            value=chat_history.messages[-1],
            reason="The last message in the chat history is the result in the default round-robin group chat manager.",
        )


class GroupChatManagerActor(RoutedAgent):
    """A group chat manager actor."""

    def __init__(
        self,
        manager: GroupChatManager,
        internal_topic_type: str,
        participant_descriptions: dict[str, str],
    ):
        """Initialize the group chat manager container."""
        self._manager = manager
        self._internal_topic_type = internal_topic_type
        self._chat_history = ChatHistory()
        self._participant_descriptions = participant_descriptions

        super().__init__(description="A container for the group chat manager.")

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        if message.body.role != AuthorRole.USER:
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {message.body.name}",
                )
            )
        self._chat_history.add_message(message.body)

        # User input state
        should_request_user_input = await self._manager.should_request_user_input(self._chat_history)
        if should_request_user_input and self._manager.user_input_func:
            logger.debug(f"Group chat manager requested user input. Reason: {should_request_user_input.reason}")
            user_input = await self._manager.user_input_func(self._chat_history)
            if user_input:
                self._chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
                await self.publish_message(
                    GroupChatResponseMessage(body=ChatMessageContent(role=AuthorRole.USER, content=user_input)),
                    TopicId(self._internal_topic_type, self.id.key),
                )
                logger.debug("User input received and added to chat history.")

        # Determine if the group chat should terminate
        should_terminate = await self._manager.should_terminate(self._chat_history)
        if should_terminate:
            logger.debug(f"Group chat manager decided to terminate the group chat. Reason: {should_terminate.reason}")
            result = await self._manager.filter_results(self._chat_history)
            await self.publish_message(
                GroupChatEndMessage(body=result.value),
                TopicId(self._internal_topic_type, self.id.key),
            )
            return

        # Select the next agent to speak if the group chat is not terminating
        next_agent = await self._manager.select_next_agent(self._chat_history, self._participant_descriptions)
        logger.debug(
            f"Group chat manager selected agent: {next_agent} on round {self._manager.current_round}. "
            f"Reason: {next_agent.reason}"
        )

        await self.publish_message(
            GroupChatRequestMessage(agent_name=next_agent.value),
            TopicId(self._internal_topic_type, self.id.key),
        )


class GroupChatOrchestration(
    OrchestrationBase[
        TExternalIn,
        GroupChatStartMessage,
        GroupChatEndMessage,
        TExternalOut,
    ]
):
    """A group chat multi-agent pattern orchestration."""

    def __init__(
        self,
        members: list[Agent | OrchestrationBase],
        manager: GroupChatManager,
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[GroupChatStartMessage] | GroupChatStartMessage]
        | None = None,
        output_transition: Callable[[GroupChatEndMessage], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the handoff orchestration.

        Args:
            members (list[Agent | OrchestrationBase]): A list of agents or orchestrations that are part of the
                handoff group. This first agent in the list will be the one that receives the first message.
            manager (GroupChatManager): The group chat manager that manages the flow of the group chat.
            name (str | None): The name of the orchestration.
            description (str | None): The description of the orchestration.
            input_transition (Callable | None): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable | None): A function that transforms the internal output message to the external
                output message.
        """
        self._manager = manager

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
        task: str | GroupChatStartMessage | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the group chat pattern."""
        if isinstance(task, str):
            message = GroupChatStartMessage(body=ChatMessageContent(AuthorRole.USER, content=task))
        elif isinstance(task, ChatMessageContent):
            message = GroupChatStartMessage(body=task)

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
        await self._register_members(runtime, internal_topic_type)
        await self._register_manager(runtime, internal_topic_type)
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
        """Register the agents."""
        await asyncio.gather(*[
            GroupChatAgentActor.register(
                runtime,
                self._get_agent_actor_type(agent, internal_topic_type),
                lambda agent=agent: GroupChatAgentActor(agent, internal_topic_type),
            )
            for agent in self._members
            if isinstance(agent, Agent)
        ])
        # TODO(@taochen): Orchestration

    async def _register_manager(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Register the group chat manager."""
        await GroupChatManagerActor.register(
            runtime,
            self._get_manager_actor_type(internal_topic_type),
            lambda: GroupChatManagerActor(
                self._manager,
                internal_topic_type=internal_topic_type,
                participant_descriptions={agent.name: agent.description for agent in self._members},
            ),
        )
        # TODO(@taochen): Orchestration

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        await GroupChatOrchestrationActor[self.t_external_in, self.t_external_out].register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: GroupChatOrchestrationActor[self.t_external_in, self.t_external_out](
                internal_topic_type,
                self._input_transition,
                self._output_transition,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
        )

    async def _add_subscriptions(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Add subscriptions."""
        subscriptions: list[TypeSubscription] = []
        for agent in [member for member in self._members if isinstance(member, Agent)]:
            subscriptions.append(
                TypeSubscription(internal_topic_type, self._get_agent_actor_type(agent, internal_topic_type))
            )
            # TODO(@taochen): Orchestration
        subscriptions.append(TypeSubscription(internal_topic_type, self._get_manager_actor_type(internal_topic_type)))
        subscriptions.append(
            TypeSubscription(internal_topic_type, self._get_orchestration_actor_type(internal_topic_type))
        )
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])

    def _get_agent_actor_type(self, agent: Agent | str, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        if isinstance(agent, Agent):
            agent = agent.name
        return f"{agent}_{internal_topic_type}"

    def _get_manager_actor_type(self, internal_topic_type: str) -> str:
        """Get the actor type for the group chat manager.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{GroupChatManagerActor.__name__}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{GroupChatOrchestrationActor.__name__}_{internal_topic_type}"

# Copyright (c) Microsoft. All rights reserved.


import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from autogen_core import MessageContext, RoutedAgent

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.orchestration.orchestration_base import DefaultExternalTypeAlias
from semantic_kernel.contents.chat_history import ChatHistory

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class ActorBase(RoutedAgent):
    """A base class for actors running in the AgentRuntime."""

    @override
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any | None:
        """Handle a message.

        Stop the handling of the message if the cancellation token is cancelled.
        """
        if ctx.cancellation_token.is_cancelled():
            return None

        return await super().on_message_impl(message, ctx)


class AgentActorBase(ActorBase):
    """A agent actor for multi-agent orchestration running on Agent runtime."""

    def __init__(
        self,
        agent: Agent,
        internal_topic_type: str,
        observer: Callable[[str | DefaultExternalTypeAlias], Awaitable[None] | None] | None = None,
    ) -> None:
        """Initialize the agent container.

        Args:
            agent (Agent): An agent to be run in the container.
            internal_topic_type (str): The topic type of the internal topic.
            observer (Callable | None): A function that is called when a response is produced by the agents.
        """
        self._agent = agent
        self._internal_topic_type = internal_topic_type
        self._observer = observer

        self._agent_thread: AgentThread | None = None
        # Chat history to temporarily store messages before the agent thread is created
        self._chat_history = ChatHistory()

        RoutedAgent.__init__(self, description=agent.description or "Semantic Kernel Agent")

    async def _notify_observer(
        self,
        message: str | DefaultExternalTypeAlias,
    ) -> None:
        """Call the observer function if it is set.

        Args:
            message (str | DefaultExternalTypeAlias): The message to be sent to the observer.
        """
        if self._observer:
            if inspect.iscoroutinefunction(self._observer):
                await self._observer(message)
            else:
                self._observer(message)

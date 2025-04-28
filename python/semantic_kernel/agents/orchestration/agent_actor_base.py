# Copyright (c) Microsoft. All rights reserved.


from autogen_core import RoutedAgent

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.contents.chat_history import ChatHistory


class AgentActorBase(RoutedAgent):
    """A agent actor for multi-agent orchestration running on Agent runtime."""

    def __init__(self, agent: Agent, internal_topic_type: str) -> None:
        """Initialize the agent container.

        Args:
            agent (Agent): An agent to be run in the container.
            internal_topic_type (str): The topic type of the internal topic.
        """
        self._agent = agent
        self._internal_topic_type = internal_topic_type

        self._agent_thread: AgentThread | None = None
        # Chat history to temporarily store messages before the agent thread is created
        self._chat_history = ChatHistory()

        RoutedAgent.__init__(self, description=agent.description or "Semantic Kernel Agent")

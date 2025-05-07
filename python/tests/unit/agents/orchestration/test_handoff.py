# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from collections.abc import AsyncIterable, Awaitable, Callable
from unittest.mock import patch

import pytest
from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.agent import Agent, AgentResponseItem, AgentThread
from semantic_kernel.agents.orchestration.handoffs import (
    HANDOFF_PLUGIN_NAME,
    HandoffAgentActor,
    HandoffConnection,
    HandoffOrchestration,
)
from semantic_kernel.agents.orchestration.orchestration_base import DefaultTypeAlias
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from tests.unit.agents.orchestration.conftest import MockAgent, MockAgentThread, MockRuntime

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class MockAgentWithHandoffFunctionCall(Agent):
    """A mock agent with handoff function call for testing purposes."""

    target_agent: Agent

    def __init__(self, target_agent: Agent):
        super().__init__(target_agent=target_agent)

    @override
    async def get_response(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
        thread: AgentThread | None = None,
        kernel: Kernel | None = None,
        **kwargs,
    ) -> AgentResponseItem[ChatMessageContent]:
        # Simulate some processing time
        await asyncio.sleep(0.1)
        await kernel.invoke_function_call(
            function_call=FunctionCallContent(
                function_name=f"transfer_to_{self.target_agent.name}",
                plugin_name=HANDOFF_PLUGIN_NAME,
                call_id="test_call_id",
                id="test_id",
            ),
            chat_history=ChatHistory(),
        )

        return AgentResponseItem[ChatMessageContent](
            message=ChatMessageContent(
                role=AuthorRole.TOOL,
                items=[
                    FunctionResultContent(
                        call_id="test_call_id",
                        id="test_id",
                        function_name=f"transfer_to_{self.target_agent.name}",
                        plugin_name=HANDOFF_PLUGIN_NAME,
                    )
                ],
            ),
            thread=thread or MockAgentThread(),
        )

    @override
    async def invoke(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
        thread: AgentThread | None = None,
        on_intermediate_message: Callable[[ChatMessageContent], Awaitable[None]] | None = None,
        **kwargs,
    ) -> AgentResponseItem[ChatMessageContent]:
        pass

    @override
    async def invoke_stream(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
        thread: AgentThread | None = None,
        on_intermediate_message: Callable[[ChatMessageContent], Awaitable[None]] | None = None,
        **kwargs,
    ) -> AsyncIterable[AgentResponseItem[StreamingChatMessageContent]]:
        pass


# region HandoffOrchestration


def test_init_without_handoffs():
    """Test the initialization of HandoffOrchestration without handoffs."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    with pytest.raises(ValueError):
        HandoffOrchestration(members=[agent_a, agent_b], handoffs={})


def test_init_with_invalid_handoff():
    """Test the initialization of HandoffOrchestration with invalid handoff."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    with pytest.raises(ValueError):
        HandoffOrchestration(
            members=[agent_a, agent_b],
            handoffs={
                agent_a.name: [
                    HandoffConnection(agent_name=agent_b.name, description="test"),
                    HandoffConnection(agent_name="invalid_agent_name", description="test"),
                ],
                agent_b.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
            },
        )

    with pytest.raises(ValueError):
        HandoffOrchestration(
            members=[agent_a, agent_b],
            handoffs={
                "invalid_agent_name": [HandoffConnection(agent_name=agent_b.name, description="test")],
                agent_b.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
            },
        )

    # Cannot handoff to self
    with pytest.raises(ValueError):
        HandoffOrchestration(
            members=[agent_a, agent_b],
            handoffs={
                agent_a.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
                agent_b.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
            },
        )


async def test_prepare():
    """Test the prepare method of the HandoffOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()
    agent_c = MockAgent()

    runtime = MockRuntime()

    package_path = "semantic_kernel.agents.orchestration.handoffs"
    with (
        patch(f"{package_path}.HandoffOrchestration._start"),
        patch(f"{package_path}.HandoffAgentActor.register") as mock_agent_actor_register,
        patch.object(runtime, "add_subscription") as mock_add_subscription,
    ):
        orchestration = HandoffOrchestration(
            members=[agent_a, agent_b, agent_c],
            handoffs={
                agent_a.name: [HandoffConnection(agent_name=agent_b.name, description="test")],
                agent_b.name: [HandoffConnection(agent_name=agent_c.name, description="test")],
                agent_c.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
            },
        )
        await orchestration.invoke(task="test_message", runtime=runtime)

        assert mock_agent_actor_register.call_count == 3
        assert mock_add_subscription.call_count == 3


async def test_invoke():
    """Test the prepare method of the HandoffOrchestration."""
    with (
        patch.object(HandoffAgentActor, "__init__", wraps=HandoffAgentActor.__init__, autospec=True) as mock_init,
        patch.object(MockAgent, "get_response", wraps=MockAgent.get_response, autospec=True) as mock_get_response,
    ):
        agent_a = MockAgent()
        agent_b = MockAgent()
        agent_c = MockAgent()

        runtime = SingleThreadedAgentRuntime()
        runtime.start()

        try:
            orchestration = HandoffOrchestration(
                members=[agent_a, agent_b, agent_c],
                handoffs={
                    agent_a.name: [
                        HandoffConnection(agent_name=agent_b.name, description="test"),
                        HandoffConnection(agent_name=agent_c.name, description="test"),
                    ],
                    agent_b.name: [HandoffConnection(agent_name=agent_a.name, description="test")],
                },
            )
            orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
            await orchestration_result.get()

            assert mock_init.call_args_list[0][0][3] == [
                HandoffConnection(agent_name=agent_b.name, description="test"),
                HandoffConnection(agent_name=agent_c.name, description="test"),
            ]
            assert isinstance(mock_get_response.call_args_list[0][1]["kernel"], Kernel)
            kernel = mock_get_response.call_args_list[0][1]["kernel"]
            assert HANDOFF_PLUGIN_NAME in kernel.plugins
            assert (
                len(kernel.plugins[HANDOFF_PLUGIN_NAME].functions) == 3
            )  # two handoff functions + complete task function
        finally:
            await runtime.stop_when_idle()


async def test_invoke_with_list():
    """Test the invoke method of the HandoffOrchestration with a list of messages."""
    with (
        patch.object(MockAgent, "get_response", wraps=MockAgent.get_response, autospec=True) as mock_get_response,
    ):
        agent_a = MockAgent()
        agent_b = MockAgent()

        runtime = SingleThreadedAgentRuntime()
        runtime.start()

        messages = [
            ChatMessageContent(role=AuthorRole.USER, content="test_message_1"),
            ChatMessageContent(role=AuthorRole.USER, content="test_message_2"),
        ]

        try:
            orchestration = HandoffOrchestration(
                members=[agent_a, agent_b],
                handoffs={
                    agent_a.name: [
                        HandoffConnection(agent_name=agent_b.name, description="test"),
                    ],
                },
            )
            orchestration_result = await orchestration.invoke(task=messages, runtime=runtime)
            await orchestration_result.get()
        finally:
            await runtime.stop_when_idle()

        assert mock_get_response.call_count == 1
        # Two messages + one message added internally to steer the conversation
        assert len(mock_get_response.call_args_list[0][1]["messages"]) == 3


async def test_invoke_with_response_callback():
    """Test the invoke method of the HandoffOrchestration with a response callback."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    responses: list[DefaultTypeAlias] = []
    try:
        orchestration = HandoffOrchestration(
            members=[agent_a, agent_b],
            handoffs={
                agent_a.name: [
                    HandoffConnection(agent_name=agent_b.name, description="test"),
                ],
            },
            agent_response_callback=lambda x: responses.append(x),
        )
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
        await orchestration_result.get(1.0)
    finally:
        await runtime.stop_when_idle()

    assert len(responses) == 1
    assert all(isinstance(item, ChatMessageContent) for item in responses)
    assert all(item.content == "mock_response" for item in responses)


async def test_invoke_with_handoff_function_call():
    """Test the invoke method of the HandoffOrchestration with a handoff function call."""
    agent_b = MockAgent()
    agent_a = MockAgentWithHandoffFunctionCall(agent_b)

    with (
        patch.object(
            HandoffAgentActor, "_handoff_to_agent", wraps=HandoffAgentActor._handoff_to_agent, autospec=True
        ) as mock_handoff_to_agent,
    ):
        runtime = SingleThreadedAgentRuntime()
        runtime.start()

        try:
            orchestration = HandoffOrchestration(
                members=[agent_a, agent_b],
                handoffs={
                    agent_a.name: [
                        HandoffConnection(agent_name=agent_b.name, description="test"),
                    ],
                },
            )
            orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
            await orchestration_result.get()
        finally:
            await runtime.stop_when_idle()

        assert mock_handoff_to_agent.call_count == 1
        assert mock_handoff_to_agent.call_args_list[0][0][1] == agent_b.name


async def test_invoke_cancel_before_completion():
    """Test the invoke method of the HandoffOrchestration with cancellation before completion."""
    with (
        patch.object(MockAgent, "get_response", wraps=MockAgent.get_response, autospec=True) as mock_get_response,
    ):
        agent_a = MockAgent()
        agent_b = MockAgent()

        runtime = SingleThreadedAgentRuntime()
        runtime.start()

        try:
            orchestration = HandoffOrchestration(
                members=[agent_a, agent_b],
                handoffs={
                    agent_a.name: [
                        HandoffConnection(agent_name=agent_b.name, description="test"),
                    ],
                },
            )
            orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

            # Cancel before first agent completes
            await asyncio.sleep(0.05)
            orchestration_result.cancel()
        finally:
            await runtime.stop_when_idle()

        assert mock_get_response.call_count == 1


async def test_invoke_cancel_after_completion():
    """Test the invoke method of the HandoffOrchestration with cancellation after completion."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = HandoffOrchestration(
            members=[agent_a, agent_b],
            handoffs={
                agent_a.name: [
                    HandoffConnection(agent_name=agent_b.name, description="test"),
                ],
            },
        )

        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Wait for the orchestration to complete
        await orchestration_result.get(1.0)

        with pytest.raises(RuntimeError, match="The invocation has already been completed."):
            orchestration_result.cancel()
    finally:
        await runtime.stop_when_idle()

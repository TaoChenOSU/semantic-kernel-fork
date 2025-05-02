# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from collections.abc import AsyncIterable, Awaitable, Callable
from unittest.mock import patch

import pytest
from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.agent import Agent, AgentResponseItem, AgentThread
from semantic_kernel.agents.orchestration.orchestration_base import DefaultTypeAlias, OrchestrationResult
from semantic_kernel.agents.orchestration.sequential import SequentialOrchestration
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from tests.unit.agents.orchestration.conftest import MockAgentThread, MockRuntime

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class MockAgent(Agent):
    """A mock agent for testing purposes."""

    @override
    async def get_response(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
        thread: AgentThread | None = None,
        **kwargs,
    ) -> AgentResponseItem[ChatMessageContent]:
        # Simulate some processing time
        await asyncio.sleep(0.1)
        return AgentResponseItem[ChatMessageContent](
            message=ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="mock_response",
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


async def test_prepare():
    """Test the prepare method of the SequentialOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = MockRuntime()

    package_path = "semantic_kernel.agents.orchestration.sequential"
    with (
        patch(f"{package_path}.SequentialOrchestration._start"),
        patch(f"{package_path}.SequentialAgentActor.register") as mock_agent_actor_register,
        patch(f"{package_path}.CollectionActor.register") as mock_collection_actor_register,
        patch.object(runtime, "add_subscription") as mock_add_subscription,
    ):
        orchestration = SequentialOrchestration(members=[agent_a, agent_b])
        await orchestration.invoke(task="test_message", runtime=runtime)

        assert mock_agent_actor_register.call_count == 2
        assert mock_collection_actor_register.call_count == 1
        assert mock_add_subscription.call_count == 0


async def test_invoke():
    """Test the invoke method of the SequentialOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = SequentialOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
        result = await orchestration_result.get(1.0)

        assert isinstance(orchestration_result, OrchestrationResult)
        assert isinstance(result, ChatMessageContent)
    finally:
        await runtime.stop_when_idle()


async def test_invoke_with_response_callback():
    """Test the invoke method of the SequentialOrchestration with a response callback."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    responses: list[DefaultTypeAlias] = []
    try:
        orchestration = SequentialOrchestration(
            members=[agent_a, agent_b],
            agent_response_callback=lambda x: responses.append(x),
        )
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
        await orchestration_result.get(1.0)

        assert len(responses) == 2
        assert all(isinstance(item, ChatMessageContent) for item in responses)
        assert all(item.content == "mock_response" for item in responses)
    finally:
        await runtime.stop_when_idle()


async def test_invoke_cancel_before_completion():
    """Test the invoke method of the SequentialOrchestration with cancellation before completion."""
    with (
        patch.object(MockAgent, "get_response", wraps=MockAgent.get_response, autospec=True) as mock_get_response,
    ):
        agent_a = MockAgent()
        agent_b = MockAgent()

        runtime = SingleThreadedAgentRuntime()
        runtime.start()

        try:
            orchestration = SequentialOrchestration(members=[agent_a, agent_b])
            orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

            # Cancel while the first agent is processing
            await asyncio.sleep(0.05)
            orchestration_result.cancel()
        finally:
            await runtime.stop_when_idle()

        assert mock_get_response.call_count == 1


async def test_invoke_cancel_after_completion():
    """Test the invoke method of the SequentialOrchestration with cancellation after completion."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = SequentialOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Wait for the orchestration to complete
        await orchestration_result.get(1.0)

        with pytest.raises(RuntimeError, match="The invocation has already been completed."):
            orchestration_result.cancel()
    finally:
        await runtime.stop_when_idle()


async def test_invoke_with_double_get_result():
    """Test the invoke method of the SequentialOrchestration with double get result."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = SequentialOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Get result before completion
        with pytest.raises(asyncio.TimeoutError):
            await orchestration_result.get(0.1)
        # The invocation should still be in progress and getting the result again should not raise an error
        result = await orchestration_result.get()

        assert isinstance(result, ChatMessageContent)
        assert result.content == "mock_response"
    finally:
        await runtime.stop_when_idle()

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from collections.abc import AsyncIterable, Awaitable, Callable
from unittest.mock import patch

import pytest
from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.agent import Agent, AgentResponseItem, AgentThread
from semantic_kernel.agents.orchestration.concurrent import ConcurrentOrchestration
from semantic_kernel.agents.orchestration.orchestration_base import DefaultTypeAlias, OrchestrationResult
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class MockAgentThread(AgentThread):
    """A mock agent thread for testing purposes."""

    @override
    async def _create(self) -> str:
        return "mock_thread_id"

    @override
    async def _delete(self) -> None:
        pass

    @override
    async def _on_new_message(self, new_message: ChatMessageContent) -> None:
        pass


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
        await asyncio.sleep(0.5)
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


def test_orchestration_init():
    """Test the initialization of the ConcurrentOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()
    agent_c = MockAgent()

    orchestration = ConcurrentOrchestration(
        members=[agent_a, agent_b, agent_c],
        name="test_orchestration",
        description="Test Orchestration",
    )

    assert orchestration.name == "test_orchestration"
    assert orchestration.description == "Test Orchestration"

    assert len(orchestration._members) == 3
    assert orchestration._input_transform is not None
    assert orchestration._output_transform is not None
    assert orchestration._agent_response_callback is None


def test_orchestration_init_with_default_values():
    """Test the initialization of the ConcurrentOrchestration with default values."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])

    assert orchestration.name
    assert orchestration.description

    assert len(orchestration._members) == 2
    assert orchestration._input_transform is not None
    assert orchestration._output_transform is not None
    assert orchestration._agent_response_callback is None


def test_orchestration_init_with_empty_members():
    """Test the initialization of the ConcurrentOrchestration with empty members."""
    with pytest.raises(ValueError):
        _ = ConcurrentOrchestration(members=[])


def test_orchestration_set_types():
    """Test the set_types method of the ConcurrentOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    # Test with default types
    orchestration_a = ConcurrentOrchestration(members=[agent_a, agent_b])
    orchestration_a._set_types()

    assert orchestration_a.t_in is DefaultTypeAlias
    assert orchestration_a.t_out is DefaultTypeAlias

    # Test with a custom input type and default output type
    orchestration_c = ConcurrentOrchestration[int](members=[agent_a, agent_b])
    orchestration_c._set_types()

    assert orchestration_c.t_in is int
    assert orchestration_c.t_out is DefaultTypeAlias

    # Test with a custom input type and custom output type
    orchestration_b = ConcurrentOrchestration[str, int](members=[agent_a, agent_b])
    orchestration_b._set_types()

    assert orchestration_b.t_in is str
    assert orchestration_b.t_out is int

    # Test with an incorrect number of types
    with pytest.raises(TypeError):
        orchestration_d = ConcurrentOrchestration[str, str, str](members=[agent_a, agent_b])
        orchestration_d._set_types()


async def test_prepare():
    """Test the prepare method of the ConcurrentOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()

    package_path = "semantic_kernel.agents.orchestration.concurrent"
    with (
        patch(f"{package_path}.ConcurrentOrchestration._start"),
        patch(f"{package_path}.ConcurrentAgentActor.register") as mock_agent_actor_register,
        patch(f"{package_path}.CollectionActor.register") as mock_collection_actor_register,
        patch.object(runtime, "add_subscription") as mock_add_subscription,
    ):
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        await orchestration.invoke(task="test_message", runtime=runtime)

        assert mock_agent_actor_register.call_count == 2
        assert mock_collection_actor_register.call_count == 1
        assert mock_add_subscription.call_count == 2


async def test_invoke():
    """Test the invoke method of the ConcurrentOrchestration."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)
        result = await orchestration_result.get(1.0)

        assert isinstance(orchestration_result, OrchestrationResult)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, ChatMessageContent) for item in result)
    finally:
        await runtime.stop_when_idle()


async def test_invoke_with_timeout_error():
    """Test the invoke method of the ConcurrentOrchestration with a timeout error."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        with pytest.raises(RuntimeError):
            await orchestration_result.get(timeout=0.1)
    finally:
        await runtime.stop_when_idle()


async def test_invoke_with_response_callback():
    """Test the invoke method of the ConcurrentOrchestration with a response callback."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    responses: list[DefaultTypeAlias] = []
    try:
        orchestration = ConcurrentOrchestration(
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
    """Test the invoke method of the ConcurrentOrchestration with cancellation before completion."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Cancel the orchestration before completion
        orchestration_result.cancel()

        with pytest.raises(RuntimeError, match="The orchestration was canceled before it could complete."):
            await orchestration_result.get()
    finally:
        await runtime.stop_when_idle()


async def test_invoke_cancel_after_completion():
    """Test the invoke method of the ConcurrentOrchestration with cancellation after completion."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Wait for the orchestration to complete
        await orchestration_result.get(1.0)

        with pytest.raises(RuntimeError, match="The orchestration has already been completed."):
            orchestration_result.cancel()
    finally:
        await runtime.stop_when_idle()


async def test_invoke_with_double_cancel():
    """Test the invoke method of the ConcurrentOrchestration with double cancel."""
    agent_a = MockAgent()
    agent_b = MockAgent()

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        orchestration = ConcurrentOrchestration(members=[agent_a, agent_b])
        orchestration_result = await orchestration.invoke(task="test_message", runtime=runtime)

        # Cancel before completion
        orchestration_result.cancel()
        # Cancelling again should raise an error
        with pytest.raises(RuntimeError, match="The orchestration has already been canceled."):
            orchestration_result.cancel()
    finally:
        await runtime.stop_when_idle()

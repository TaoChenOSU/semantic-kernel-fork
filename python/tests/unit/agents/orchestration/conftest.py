# Copyright (c) Microsoft. All rights reserved.

import sys

from autogen_core import AgentRuntime

from semantic_kernel.agents.agent import AgentThread
from semantic_kernel.contents.chat_message_content import ChatMessageContent

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


class MockRuntime(AgentRuntime):
    """A mock agent runtime for testing purposes."""

    pass

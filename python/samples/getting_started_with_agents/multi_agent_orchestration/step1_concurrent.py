# Copyright (c) Microsoft. All rights reserved.

import asyncio

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.concurrent import ConcurrentOrchestration
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent


async def main():
    """Main function to run the agents."""
    physics_agent = ChatCompletionAgent(
        name="PhysicsExpert",
        description="An expert in physics",
        instructions="You are an expert in physics.",
        service=OpenAIChatCompletion(),
    )
    chemistry_agent = ChatCompletionAgent(
        name="ChemistryExpert",
        description="An expert in chemistry",
        instructions="You are an expert in chemistry.",
        service=OpenAIChatCompletion(),
    )

    concurrent_orchestration = ConcurrentOrchestration(
        members=[physics_agent, chemistry_agent],
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    orchestration_result = await concurrent_orchestration.invoke(
        task="Why is the sky blue in one sentence?",
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=10)
    if isinstance(value, list) and all(isinstance(item, ChatMessageContent) for item in value):
        for item in value:
            print(f"{item.name}: {item.content}")
    else:
        print("Unexpected result type:", type(value))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

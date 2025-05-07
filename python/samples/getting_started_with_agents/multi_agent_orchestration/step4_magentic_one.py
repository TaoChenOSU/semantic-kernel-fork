# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.open_ai.open_ai_assistant_agent import OpenAIAssistantAgent
from semantic_kernel.agents.orchestration.magentic_one import MagenticOneManager, MagenticOneOrchestration
from semantic_kernel.agents.runtime.in_process.in_process_runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"**{message.name}**\n{message.content}")


async def main():
    """Main function to run the agents."""
    research_agent = ChatCompletionAgent(
        name="ResearchAgent",
        description="A helpful assistant with access to web search. Ask it to perform web searches.",
        instructions=("You are a Researcher. You find information."),
        service=OpenAIChatCompletion(ai_model_id="gpt-4o-search-preview"),
    )

    # Create an OpenAI Assistant agent with code interpreter capability
    client, model = OpenAIAssistantAgent.setup_resources()
    code_interpreter_tool, code_interpreter_tool_resources = OpenAIAssistantAgent.configure_code_interpreter_tool()
    definition = await client.beta.assistants.create(
        model=model,
        name="CoderAgent",
        description="A helpful assistant with code interpreter capability.",
        instructions="You solve questions using code.",
        tools=code_interpreter_tool,
        tool_resources=code_interpreter_tool_resources,
    )
    coder_agent = OpenAIAssistantAgent(
        client=client,
        definition=definition,
    )

    magentic_one_orchestration = MagenticOneOrchestration(
        members=[research_agent, coder_agent],
        manager=MagenticOneManager(
            chat_completion_service=OpenAIChatCompletion(),
            prompt_execution_settings=OpenAIPromptExecutionSettings(),
        ),
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime()
    runtime.start()

    orchestration_result = await magentic_one_orchestration.invoke(
        task=(
            "What are the 50 tallest buildings in the world? Create a table with their names"
            " and heights grouped by country with a column of the average height of the buildings"
            " in each country."
        ),
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(value)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

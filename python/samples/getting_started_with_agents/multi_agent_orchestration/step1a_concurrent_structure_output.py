# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.concurrent import ConcurrentOrchestration
from semantic_kernel.agents.orchestration.tools import structure_output_transform
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.kernel_pydantic import KernelBaseModel


class ArticleAnalysis(KernelBaseModel):
    """A model to hold the analysis of an article."""

    themes: list[str]
    sentiments: list[str]
    entities: list[str]


async def main():
    """Main function to run the agents."""
    theme_agent = ChatCompletionAgent(
        name="ThemeAgent",
        description="An expert in identifying themes in articles",
        instructions="You are an expert in identifying themes in articles. Given an article, identify the main themes.",
        service=OpenAIChatCompletion(),
    )
    sentiment_agent = ChatCompletionAgent(
        name="SentimentAgent",
        description="An expert in sentiment analysis",
        instructions="You are an expert in sentiment analysis. Given an article, identify the sentiment.",
        service=OpenAIChatCompletion(),
    )
    entity_agent = ChatCompletionAgent(
        name="EntityAgent",
        description="An expert in entity recognition",
        instructions="You are an expert in entity recognition. Given an article, extract the entities.",
        service=OpenAIChatCompletion(),
    )

    concurrent_orchestration = ConcurrentOrchestration[str, ArticleAnalysis](
        members=[theme_agent, sentiment_agent, entity_agent],
        output_transform=structure_output_transform(ArticleAnalysis, OpenAIChatCompletion()),
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    with open(os.path.join(os.path.dirname(__file__), "../resources", "Hamlet_full_play_summary.txt")) as file:
        task = file.read()

    orchestration_result = await concurrent_orchestration.invoke(
        task=task,
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=10)
    if isinstance(value, ArticleAnalysis):
        print(value.model_dump_json(indent=2))
    else:
        print("Unexpected result type:", type(value))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

# Copyright (c) Microsoft. All rights reserved.

import asyncio
from enum import Enum

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.handoffs import HandoffConnection, HandoffOrchestration
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel


class GitHubLabels(Enum):
    """Enum representing GitHub labels."""

    PYTHON = "python"
    DOTNET = ".NET"
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    QUESTION = "question"
    VECTORSTORE = "vectorstore"
    AGENT = "agent"


class GithubIssue(KernelBaseModel):
    """Model representing a GitHub issue."""

    id: str
    title: str
    body: str
    labels: list[str] = []


class GithubPlugin:
    """Plugin for GitHub related operations."""

    @kernel_function
    async def add_labels(self, issue_id: str, labels: list[GitHubLabels]) -> None:
        """Add labels to a GitHub issue."""
        await asyncio.sleep(1)  # Simulate network delay


GithubIssue_12345 = GithubIssue(
    id="12345",
    title=(
        "Bug: SQLite Error 1: 'ambiguous column name:' when including VectorStoreRecordKey in "
        "VectorSearchOptions.Filter"
    ),
    body=(
        "Describe the bug"
        "When using column names marked as [VectorStoreRecordData(IsFilterable = true)] in "
        "VectorSearchOptions.Filter, the query runs correctly."
        "However, using the column name marked as [VectorStoreRecordKey] in VectorSearchOptions.Filter, the query "
        "throws exception 'SQLite Error 1: ambiguous column name: StartUTC"
        ""
        "To Reproduce"
        "Add a filter for the column marked [VectorStoreRecordKey]. Since that same column exists in both the "
        "vec_TestTable and TestTable, the data for both columns cannot be returned."
        ""
        "Expected behavior"
        "The query should explicitly list the vec_TestTable column names to retrieve and should omit the "
        "[VectorStoreRecordKey] column since it will be included in the primary TestTable columns."
        ""
        "Platform"
        ""
        "Microsoft.SemanticKernel.Connectors.Sqlite v1.46.0-preview"
        "Additional context"
        "Normal DBContext logging shows only normal context queries. Queries run by VectorizedSearchAsync() don't "
        "appear in those logs and I could not find a way to enable logging in semantic search so that I could "
        "actually see the exact query that is failing. It would have been very useful to see the failing semantic "
        "query."
    ),
    labels=[],
)


# The default input transform will attempt to serialize an object into a string by using
# `json.dump()`. However, an object of a Pydantic model type cannot be directly serialize
# by `json.dump()`. Thus, we will need a custom transform.
def custom_input_transform(input_message: GithubIssue) -> ChatMessageContent:
    return ChatMessageContent(role=AuthorRole.USER, content=input_message.model_dump_json())


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"{message.name}: {message.content}")


async def main():
    """Main function to run the agents."""
    triage_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="An agent that triages GitHub issues",
        instructions="Given a GitHub issue, triage it.",
        service=OpenAIChatCompletion(),
    )
    python_agent = ChatCompletionAgent(
        name="PythonAgent",
        description="An agent that handles Python related issues",
        instructions="You are an agent that handles Python related GitHub issues.",
        service=OpenAIChatCompletion(),
        plugins=[GithubPlugin()],
    )
    dotnet_agent = ChatCompletionAgent(
        name="DotNetAgent",
        description="An agent that handles .NET related issues",
        instructions="You are an agent that handles .NET related GitHub issues.",
        service=OpenAIChatCompletion(),
        plugins=[GithubPlugin()],
    )

    handoff_orchestration = HandoffOrchestration[GithubIssue, ChatMessageContent](
        members=[triage_agent, python_agent, dotnet_agent],
        handoffs={
            triage_agent.name: [
                HandoffConnection(
                    agent_name=python_agent.name,
                    description="Transfer to this agent if the issue is Python related",
                ),
                HandoffConnection(
                    agent_name=dotnet_agent.name,
                    description="Transfer to this agent if the issue is .NET related",
                ),
            ]
        },
        input_transform=custom_input_transform,
        agent_response_callback=agent_response_callback,
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    orchestration_result = await handoff_orchestration.invoke(
        task=GithubIssue_12345,
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=100)
    print(value)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

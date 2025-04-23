# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from enum import Enum

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.handoffs import HandoffConnection, HandoffOrchestration
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.handoffs").setLevel(
    logging.DEBUG
)  # Enable DEBUG for concurrent pattern


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

    handoff_pattern = HandoffOrchestration(
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
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    github_issue = GithubIssue(
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

    orchestration_result = await handoff_pattern.invoke(
        task=github_issue.model_dump_json(),
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=100)
    print(value.body)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

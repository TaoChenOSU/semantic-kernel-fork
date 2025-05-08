# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.group_chat import (
    BoolWithReason,
    GroupChatOrchestration,
    RoundRobinGroupChatManager,
)
from semantic_kernel.agents.runtime.in_process.in_process_runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

"""
The following sample demonstrates how to create a group chat orchestration with human
in the loop. Human in the loop is achieved by overriding the default round robin manager
to allow user input after the reviewer agent's message.

Human in the loop is useful when you want to have a human participant in the group chat
orchestration. Human in the loop can be supported by all group chat managers. 

This sample demonstrates the basic steps of creating and starting a runtime, creating
a group chat orchestration with a group chat manager, invoking the orchestration,
and finally waiting for the results.

There are two agents in this orchestration: a writer and a reviewer. They work iteratively
to refine a slogan for a new electric SUV.
"""


def agents() -> list[ChatCompletionAgent]:
    """Return a list of agents that will participate in the group style discussion.

    Feel free to add or remove agents.
    """
    writer = ChatCompletionAgent(
        name="Writer",
        description="A content writer.",
        instructions=(
            "You are an excellent content writer. You create new content and edit contents based on the feedback."
        ),
        service=OpenAIChatCompletion(),
    )
    reviewer = ChatCompletionAgent(
        name="Reviewer",
        description="A content reviewer.",
        instructions=(
            "You are an excellent content reviewer. You review the content and provide feedback to the writer."
        ),
        service=OpenAIChatCompletion(),
    )

    # The order of the agents in the list will be the order in which they will be picked by the round robin manager
    return [writer, reviewer]


class CustomRoundRobinGroupChatManager(RoundRobinGroupChatManager):
    """Custom round robin group chat manager to enable user input.

    The default round robin manager does not allow user input.
    """

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input."""
        if len(chat_history.messages) == 0:
            return BoolWithReason(
                value=False,
                reason="No agents have spoken yet.",
            )
        last_message = chat_history.messages[-1]
        if last_message.name == "Reviewer":
            return BoolWithReason(
                value=True,
                reason="User input is needed after the reviewer's message.",
            )

        return BoolWithReason(
            value=False,
            reason="User input is not needed if the last message is not from the reviewer.",
        )


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"**{message.name}**\n{message.content}")


async def human_response_function(chat_histoy: ChatHistory) -> ChatMessageContent:
    """Function to get user input."""
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def main():
    """Main function to run the agents."""
    # 1. Create a group chat orchestration with a round robin manager
    group_chat_orchestration = GroupChatOrchestration(
        members=agents(),
        # max_rounds is odd, so that the writer gets the last round
        manager=CustomRoundRobinGroupChatManager(
            max_rounds=5,
            human_response_function=human_response_function,
        ),
        agent_response_callback=agent_response_callback,
    )

    # 2. Create a runtime and start it
    runtime = InProcessRuntime()
    runtime.start()

    # 3. Invoke the orchestration with a task and the runtime
    orchestration_result = await group_chat_orchestration.invoke(
        task="Create a slogon for a new eletric SUV that is affordable and fun to drive.",
        runtime=runtime,
    )

    # 4. Wait for the results
    value = await orchestration_result.get()
    print(f"***** Result *****\n{value}")

    # 5. Stop the runtime after the invocation is complete
    await runtime.stop_when_idle()

    """
    **Writer**
    "Electrify Your Drive: Affordable Fun for Everyone!"
    **Reviewer**
    This slogan, "Electrify Your Drive: Affordable Fun for Everyone!" does a great job of conveying the core benefits
    of an electric SUV. Here's some feedback to consider:

    ...

    Consider testing this slogan with focus groups or within your target market to gather insights on resonance and
    perception. Overall, it is a compelling and engaging statement that successfully captures the essence of your
    electric SUV.
    User: Make it rhyme
    **Writer**
    "Drive Electric, Feel the Thrill, Affordable Fun That Fits the Bill!"
    **Reviewer**
    The slogan, "Drive Electric, Feel the Thrill, Affordable Fun That Fits the Bill!" successfully incorporates rhyme,
    adding a catchy and memorable element to your marketing message. Here's some detailed feedback on this version:

    ...

    Overall, this rhyming slogan is an improvement for making the tagline more memorable and appealing. It captures the
    excitement and accessibility of the product effectively. Consider checking how it resonates with your target
    demographic to ensure it aligns well with their preferences and expectations.
    User: Nice!
    **Writer**
    Thank you! I'm glad you liked the feedback. If you need help with anything else, like tailoring the slogan for
    specific platforms or audiences, just let me know!
    ***** Result *****
    Thank you! I'm glad you liked the feedback. If you need help with anything else, like tailoring the slogan for
    specific platforms or audiences, just let me know!
    """


if __name__ == "__main__":
    asyncio.run(main())

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.group_chat import (
    BoolWithReason,
    ChatMessageContentWithReason,
    GroupChatManager,
    GroupChatOrchestration,
    StringWithReason,
)
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.group_chat").setLevel(
    logging.DEBUG
)  # Enable DEBUG for group chat pattern


def agents() -> list[ChatCompletionAgent]:
    """Return a list of agents that will participate in the group style discussion.

    Feel free to add or remove agents.
    """
    farmer = ChatCompletionAgent(
        name="Farmer",
        description="A rural farmer from Southeast Asia.",
        instructions=(
            "You're a farmer from Southeast Asia. "
            "Your life is deeply connected to land and family. "
            "You value tradition and sustainability."
        ),
        service=OpenAIChatCompletion(),
    )
    developer = ChatCompletionAgent(
        name="Developer",
        description="An urban software developer from the United States.",
        instructions=(
            "You're a software developer from the United States. "
            "Your life is fast-paced and technology-driven. "
            "You value innovation, freedom, and work-life balance."
        ),
        service=OpenAIChatCompletion(),
    )
    teacher = ChatCompletionAgent(
        name="Teacher",
        description="A retired history teacher from Eastern Europe",
        instructions=(
            "You're a retired history teacher from Eastern Europe. "
            "You bring historical and philosophical perspectives to discussions. "
            "You value legacy, learning, and cultural continuity."
        ),
        service=OpenAIChatCompletion(),
    )
    activist = ChatCompletionAgent(
        name="Activist",
        description="A young activist from South America.",
        instructions=(
            "You're a young activist from South America. "
            "You focus on social justice, environmental rights, and generational change."
        ),
        service=OpenAIChatCompletion(),
    )
    spiritual_leader = ChatCompletionAgent(
        name="SpiritualLeader",
        description="A spiritual leader from the Middle East.",
        instructions=(
            "You're a spiritual leader from the Middle East. "
            "You provide insights grounded in religion, morality, and community service."
        ),
        service=OpenAIChatCompletion(),
    )
    artist = ChatCompletionAgent(
        name="Artist",
        description="An artist from Africa.",
        instructions=(
            "You're an artist from Africa. "
            "You view life through creative expression, storytelling, and collective memory."
        ),
        service=OpenAIChatCompletion(),
    )
    immigrant = ChatCompletionAgent(
        name="Immigrant",
        description="An immigrant entrepreneur from Asia living in Canada.",
        instructions=(
            "You're an immigrant entrepreneur from Asia living in Canada. "
            "You balance trandition with adaption. "
            "You focus on family success, risk, and opportunity."
        ),
        service=OpenAIChatCompletion(),
    )
    doctor = ChatCompletionAgent(
        name="Doctor",
        description="A doctor from Scandinavia.",
        instructions=(
            "You're a doctor from Scandinavia. "
            "Your perspective is shaped by public health, equity, and structured societal support."
        ),
        service=OpenAIChatCompletion(),
    )

    return [farmer, developer, teacher, activist, spiritual_leader, artist, immigrant, doctor]


class ChatCompletionGroupChatManager(GroupChatManager):
    """A simple chat completion base group chat manager.

    This chat completion service requires a model that supports structured output.
    """

    service: ChatCompletionClientBase

    topic: str

    termination_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You need to determine if the discussion has reached a conclusion. "
        "If you would like to end the discussion, please respond with True. Otherwise, respond with False."
    )

    selection_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You need to select the next participant to speak. "
        "Here are the names and descriptions of the participants: "
        "{{$participants}}\n"
        "Please respond with only the name of the participant you would like to select."
    )

    result_filter_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You have just concluded the discussion. "
        "Please summarize the discussion and provide a closing statement."
    )

    def __init__(self, topic: str, service: ChatCompletionClientBase, **kwargs) -> None:
        """Initialize the group chat manager."""
        super().__init__(topic=topic, service=service, **kwargs)

    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments."""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input."""
        return BoolWithReason(
            value=False,
            reason="This group chat manager does not require user input.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None and self.current_round >= self.max_rounds:
            return BoolWithReason(
                value=True,
                reason=f"Maximum rounds reached: {self.max_rounds}.",
            )

        self.current_round += 1

        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.termination_prompt,
                    KernelArguments(topic=self.topic),
                ),
            )
        )

        response = await self.service.get_chat_message_content(
            chat_history,
            settings=PromptExecutionSettings(response_format=BoolWithReason),
        )

        return BoolWithReason.model_validate_json(response.content)

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringWithReason:
        """Select the next agent to speak."""
        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.selection_prompt,
                    KernelArguments(
                        topic=self.topic,
                        participants="\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()]),
                    ),
                ),
            )
        )

        response = await self.service.get_chat_message_content(
            chat_history,
            settings=PromptExecutionSettings(response_format=StringWithReason),
        )

        participant_name_with_reason = StringWithReason.model_validate_json(response.content)

        if participant_name_with_reason.value in participant_descriptions:
            return participant_name_with_reason

        raise RuntimeError(f"Unknown participant selected: {response.content}.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ChatMessageContentWithReason:
        """Filter the results of the group chat."""
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        chat_history_clone = chat_history.model_copy(deep=True)
        chat_history_clone.add_message(
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.result_filter_prompt,
                    KernelArguments(topic=self.topic),
                ),
            )
        )
        response = await self.service.get_chat_message_content(
            chat_history_clone,
            settings=PromptExecutionSettings(response_format=StringWithReason),
        )
        string_with_reason = StringWithReason.model_validate_json(response.content)

        return ChatMessageContentWithReason(
            value=ChatMessageContent(role=AuthorRole.ASSISTANT, content=string_with_reason.value),
            reason=string_with_reason.reason,
        )


async def main():
    """Main function to run the agents."""
    topic = "What does a good life mean to you personally?"

    group_chat_orchestration = GroupChatOrchestration(
        members=agents(),
        manager=ChatCompletionGroupChatManager(
            topic=topic,
            service=OpenAIChatCompletion(),
            max_rounds=10,
        ),
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    orchestration_result = await group_chat_orchestration.invoke(
        task="Please start the discussion.",
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=100)
    print(value)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

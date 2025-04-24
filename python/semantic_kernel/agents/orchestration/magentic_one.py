# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable

from autogen_core import AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationActorBase, OrchestrationBase
from semantic_kernel.agents.orchestration.prompts._magentic_one_prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
)
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)


# region Messages and Types
class MagenticOneStartMessage(KernelBaseModel):
    """A message to start a magentic one group chat."""

    body: ChatMessageContent


class MagenticOneEndMessage(KernelBaseModel):
    """A message to end a magentic one group chat."""

    body: ChatMessageContent


class MagenticOneRequestMessage(KernelBaseModel):
    """A request message type for agents in a magentic one group chat."""

    agent_name: str


class MagenticOneResponseMessage(KernelBaseModel):
    """A response message type from agents in a magentic one group chat."""

    body: ChatMessageContent


class MagenticOneResetMessage(KernelBaseModel):
    """A message to reset a participant's chat history in a magentic one group chat."""

    pass


TExternalIn = TypeVar("TExternalIn", default=MagenticOneStartMessage)
TExternalOut = TypeVar("TExternalOut", default=MagenticOneEndMessage)


class ProgressLedgerItem(KernelBaseModel):
    """A progress ledger item."""

    reason: str
    answer: str | bool


class ProgressLedger(KernelBaseModel):
    """A progress ledger."""

    is_request_satisfied: ProgressLedgerItem
    is_in_loop: ProgressLedgerItem
    is_progress_being_made: ProgressLedgerItem
    next_speaker: ProgressLedgerItem
    instruction_or_question: ProgressLedgerItem


# endregion Messages and Types

# region MagenticOneManager


class MagenticOneManager(KernelBaseModel):
    """Container for the Magentic One pattern."""

    chat_completion_service: ChatCompletionClientBase
    prompt_execution_settings: PromptExecutionSettings

    participant_descriptions: dict[str, str]
    max_stall_count: int = 3

    task_ledger_facts_prompt: str = ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT
    task_ledger_plan_prompt: str = ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT
    task_ledger_full_prompt: str = ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
    task_ledger_facts_update_prompt: str = ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT
    task_ledger_plan_update_prompt: str = ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT
    progress_ledger_prompt: str = ORCHESTRATOR_PROGRESS_LEDGER_PROMPT
    final_answer_prompt: str = ORCHESTRATOR_FINAL_ANSWER_PROMPT

    async def create_facts_and_plan(
        self,
        chat_history: ChatHistory,
        task: ChatMessageContent,
        old_facts: ChatMessageContent | None = None,
    ) -> tuple[ChatMessageContent, ChatMessageContent]:
        """Create facts and plan for the task.

        Args:
            chat_history (ChatHistory): The chat history. This chat history will be modified by the function.
            task (ChatMessageContent): The task.
            old_facts (ChatMessageContent | None): The old facts. If provided, the facts and plan update
                prompts will be used.

        Returns:
            tuple[ChatMessageContent, ChatMessageContent]: The facts and plan.
        """
        # 1. Update the facts
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(
                template=self.task_ledger_facts_update_prompt if old_facts else self.task_ledger_facts_prompt
            )
        )
        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    Kernel(),
                    KernelArguments(task=task.content, old_facts=old_facts.content)
                    if old_facts
                    else KernelArguments(task=task.content),
                ),
            )
        )
        facts = await self.chat_completion_service.get_chat_message_content(
            chat_history,
            self.prompt_execution_settings,
        )
        chat_history.add_message(facts)

        # 2. Update the plan
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(
                template=self.task_ledger_plan_update_prompt if old_facts else self.task_ledger_plan_prompt
            )
        )
        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    Kernel(),
                    KernelArguments(team=self.participant_descriptions),
                ),
            )
        )
        plan = await self.chat_completion_service.get_chat_message_content(
            chat_history,
            self.prompt_execution_settings,
        )

        return facts, plan

    async def create_task_ledger(
        self,
        task: ChatMessageContent,
        facts: ChatMessageContent,
        plan: ChatMessageContent,
    ) -> str:
        """Create a task ledger.

        Args:
            task (ChatMessageContent): The task.
            facts (ChatMessageContent): The facts.
            plan (ChatMessageContent): The plan.

        Returns:
            str: The task ledger.
        """
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=self.task_ledger_full_prompt)
        )

        return await prompt_template.render(
            Kernel(),
            KernelArguments(
                task=task.content,
                team=self.participant_descriptions,
                facts=facts.content,
                plan=plan.content,
            ),
        )

    async def create_progress_ledger(self, chat_history: ChatHistory, task: ChatMessageContent) -> ProgressLedger:
        """Create a progress ledger.

        Args:
            chat_history (ChatHistory): The chat history. This chat history will be modified by the function.
            task (ChatMessageContent): The task.

        Returns:
            ProgressLedger: The progress ledger.
        """
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=self.progress_ledger_prompt)
        )
        progress_ledger_prompt = await prompt_template.render(
            Kernel(),
            KernelArguments(
                task=task.content,
                team=self.participant_descriptions,
                names=", ".join(self.participant_descriptions.keys()),
            ),
        )
        chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=progress_ledger_prompt))

        prompt_execution_settings_clone = PromptExecutionSettings.from_prompt_execution_settings(
            self.prompt_execution_settings
        )
        prompt_execution_settings_clone.update_from_prompt_execution_settings(
            # TODO(@taochen): Double check how to make sure the service support json output.
            PromptExecutionSettings(extension_data={"response_format": ProgressLedger})
        )

        response = await self.chat_completion_service.get_chat_message_content(
            chat_history,
            prompt_execution_settings_clone,
        )

        return ProgressLedger.model_validate_json(response.content)

    async def prepare_final_answer(self, chat_history: ChatHistory, task: ChatMessageContent) -> ChatMessageContent:
        """Prepare the final answer.

        Args:
            chat_history (ChatHistory): The chat history. This chat history will be modified by the function.
            task (ChatMessageContent): The task.

        Returns:
            ChatMessageContent: The final answer.
        """
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=self.final_answer_prompt)
        )
        chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(Kernel(), KernelArguments(task=task)),
            )
        )

        return await self.chat_completion_service.get_chat_message_content(
            chat_history,
            self.prompt_execution_settings,
        )


# endregion MagenticOneManager

# region MagenticOneManagerActor


class MagenticOneManagerActor(RoutedAgent):
    """Actor for the Magentic One manager."""

    def __init__(self, manager: MagenticOneManager, internal_topic_type: str) -> None:
        """Initialize the Magentic One manager actor.

        Args:
            manager (MagenticOneManager): The Magentic One manager.
            internal_topic_type (str): The internal topic type.
        """
        self._manager = manager
        self._internal_topic_type = internal_topic_type
        self._chat_history = ChatHistory()
        super().__init__(description="Magentic One Manager")

    @message_handler
    async def _on_task_start_message(self, message: MagenticOneStartMessage, ctx: MessageContext) -> None:
        self._task = message.body
        # TODO(@taochen): Check if the task is already started.
        self._round_count = 0
        self._stall_count = 0
        self._facts, self._plan = await self._manager.create_facts_and_plan(
            self._chat_history.model_copy(deep=True),
            self._task,
        )

        logger.debug(f"{self.id}: Running outer loop.")
        await self._run_outer_loop()

    @message_handler
    async def _on_group_chat_message(self, message: MagenticOneResponseMessage, ctx: MessageContext) -> None:
        if message.body.role != AuthorRole.USER:
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {message.body.name}",
                )
            )
        self._chat_history.add_message(message.body)

        logger.debug(f"{self.id}: Running inner loop.")
        await self._run_inner_loop()

    async def _run_outer_loop(self):
        # 1. Create a task ledger.
        task_ledger = await self._manager.create_task_ledger(self._task, self._facts, self._plan)

        # 2. Publish the task ledger to the group chat.
        # Need to add the task ledger to the orchestrator's chat history
        # since the publisher won't receive the message it sends even though
        # the publisher also subscribes to the topic.
        self._chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=task_ledger,
                name=self.__class__.__name__,
            )
        )

        logger.debug(f"Initial task ledger:\n{task_ledger}")
        await self.publish_message(
            MagenticOneResponseMessage(
                body=self._chat_history.messages[-1],
            ),
            TopicId(self._internal_topic_type, self.id.key),
        )

        # 3. Start the inner loop.
        await self._run_inner_loop()

    async def _run_inner_loop(self) -> None:
        self._round_count += 1

        # 1. Create a progress ledger
        current_progress_ledger = await self._manager.create_progress_ledger(
            self._chat_history.model_copy(deep=True),
            self._task,
        )
        logger.debug(f"Current progress ledger:\n{current_progress_ledger.model_dump_json(indent=2)}")

        # 2. Process the progress ledger
        # 2.1 Check for task completion
        if current_progress_ledger.is_request_satisfied.answer:
            logger.debug("Task completed.")
            await self._prepare_final_answer()
            return
        # 2.2 Check for stalling or looping
        if not current_progress_ledger.is_progress_being_made.answer or current_progress_ledger.is_in_loop.answer:
            self._stall_count += 1
        else:
            self._stall_count = max(0, self._stall_count - 1)

        if self._stall_count > self._manager.max_stall_count:
            logger.debug("Stalling detected. Resetting the task.")
            self._facts, self._plan = await self._manager.create_facts_and_plan(
                self._chat_history.model_copy(deep=True),
                self._task,
                old_facts=self._facts,
            )
            await self._reset_for_outer_loop()
            logger.debug("Restarting outer loop.")
            await self._run_outer_loop()
            return

        # 2.3 Publish for next step
        next_step = current_progress_ledger.instruction_or_question.answer
        self._chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=next_step,
                name=self.__class__.__name__,
            )
        )
        await self.publish_message(
            MagenticOneResponseMessage(
                body=self._chat_history.messages[-1],
            ),
            TopicId(self._internal_topic_type, self.id.key),
        )

        # 2.4 Request the next speaker to speak
        next_speaker = current_progress_ledger.next_speaker.answer
        if next_speaker not in self._manager.participant_descriptions:
            raise ValueError(f"Unknown speaker: {next_speaker}")

        logger.debug(f"Magentic One manager selected agent: {next_speaker}")

        await self.publish_message(
            MagenticOneRequestMessage(agent_name=next_speaker),
            TopicId(self._internal_topic_type, self.id.key),
        )

    async def _reset_for_outer_loop(self) -> None:
        await self.publish_message(
            MagenticOneResetMessage(),
            TopicId(self._internal_topic_type, self.id.key),
        )
        self._chat_history.clear()
        self._stall_count = 0

    async def _prepare_final_answer(self) -> None:
        final_answer = await self._manager.prepare_final_answer(
            self._chat_history.model_copy(deep=True),
            self._task,
        )

        await self.publish_message(
            MagenticOneEndMessage(body=final_answer),
            TopicId(self._internal_topic_type, self.id.key),
        )


# endregion MagenticOneManagerActor

# region MagenticOneAgentActor


class MagenticOneAgentActor(AgentActorBase):
    """An agent actor that process messages in a Magentic One group chat."""

    def __init__(self, agent: Agent, internal_topic_type: str):
        """Initialize the Magentic One agent actor.

        Args:
            agent (Agent): The agent to be used.
            internal_topic_type (str): The internal topic type.
        """
        super().__init__(agent=agent, internal_topic_type=internal_topic_type)

        self._agent_thread: AgentThread | None = None
        # Chat history to temporarily store messages before the agent thread is created
        self._chat_history = ChatHistory()

    @message_handler
    async def _handle_response_message(self, message: MagenticOneResponseMessage, ctx: MessageContext) -> None:
        logger.debug(f"{self.id}: Received response message.")
        if self._agent_thread is not None:
            if message.body.role != AuthorRole.USER:
                await self._agent_thread.on_new_message(
                    ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"Transferred to {message.body.name}",
                    )
                )
            await self._agent_thread.on_new_message(message.body)
        else:
            if message.body.role != AuthorRole.USER:
                self._chat_history.add_message(
                    ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"Transferred to {message.body.name}",
                    )
                )
            self._chat_history.add_message(message.body)

    @message_handler
    async def _handle_request_message(self, message: MagenticOneRequestMessage, ctx: MessageContext) -> None:
        if message.agent_name != self._agent.name:
            return

        logger.debug(f"{self.id}: Received request message.")
        if self._agent_thread is None:
            # Add a user message to steer the agent to respond more closely to the instructions.
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
                )
            )
            response_item = await self._agent.get_response(messages=self._chat_history.messages)
            self._agent_thread = response_item.thread
        else:
            # Add a user message to steer the agent to respond more closely to the instructions.
            new_message = ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Transferred to {self._agent.name}, adopt the persona immediately.",
            )
            response_item = await self._agent.get_response(messages=new_message, thread=self._agent_thread)

        logger.debug(f"{self.id} responded with {response_item.message.content}.")

        await self.publish_message(
            MagenticOneResponseMessage(body=response_item.message),
            TopicId(self._internal_topic_type, self.id.key),
        )


# endregion MagenticOneAgentActor

# region MagenticOneOrchestrationActor


class MagenticOneOrchestrationActor(
    OrchestrationActorBase[
        TExternalIn,
        MagenticOneStartMessage,
        MagenticOneEndMessage,
        TExternalOut,
    ],
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    def __init__(
        self,
        internal_topic_type: str,
        input_transition: Callable[[TExternalIn], Awaitable[MagenticOneStartMessage] | MagenticOneStartMessage],
        output_transition: Callable[[MagenticOneEndMessage], Awaitable[TExternalOut] | TExternalOut],
        *,
        manager_actor_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ):
        """Initialize the Magentic One orchestration actor.

        Args:
            internal_topic_type (str): The internal topic type.
            input_transition (Callable[[TExternalIn], Awaitable[TInternalIn] | TInternalIn]):
                A function to transform the input message.
            output_transition (Callable[[TInternalOut], Awaitable[TExternalOut] | TExternalOut]):
                A function to transform the output message.
            manager_actor_type (str): The manager actor type.
            external_topic_type (str | None): The external topic type.
            direct_actor_type (str | None): The direct actor type.
            result_callback (Callable[[TExternalOut], None] | None): A callback function for the result.
        """
        self._manager_actor_type = manager_actor_type

        super().__init__(
            internal_topic_type=internal_topic_type,
            input_transition=input_transition,
            output_transition=output_transition,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )

    @override
    async def _handle_orchestration_input_message(
        self,
        # The following does not validate LSP because Python doesn't recognize the generic type
        message: MagenticOneStartMessage,  # type: ignore
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        target_actor_id = await self.runtime.get(self._manager_actor_type)
        await self.send_message(message, target_actor_id)

    @override
    async def _handle_orchestration_output_message(
        self,
        message: MagenticOneEndMessage,
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration output message.")
        if inspect.isawaitable(self._output_transition):
            external_output_message: TExternalOut = await self._output_transition(message)
        else:
            external_output_message: TExternalOut = self._output_transition(message)  # type: ignore[no-redef]

        if self._external_topic_type:
            logger.debug(f"Relaying message to external topic: {self._external_topic_type}")
            await self.publish_message(
                external_output_message,
                TopicId(self._external_topic_type, self.id.key),
            )
        if self._direct_actor_type:
            logger.debug(f"Relaying message directly to actor: {self._direct_actor_type}")
            target_actor_id = await self.runtime.get(self._direct_actor_type)
            await self.send_message(external_output_message, target_actor_id)
        if self._result_callback:
            self._result_callback(external_output_message)


# endregion MagenticOneOrchestrationActor

# region MagenticOneOrchestration


class MagenticOneOrchestration(
    OrchestrationBase[
        TExternalIn,
        MagenticOneStartMessage,
        MagenticOneEndMessage,
        TExternalOut,
    ]
):
    """The Magentic One pattern orchestration."""

    def __init__(
        self,
        members: list[Agent | OrchestrationBase],
        manager: MagenticOneManager,
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[MagenticOneStartMessage] | MagenticOneStartMessage]
        | None = None,
        output_transition: Callable[[MagenticOneEndMessage], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the Magentic One orchestration.

        Args:
            members (list[Agent | OrchestrationBase]): A list of agents or orchestration bases.
            manager (MagenticOneManager): The manager for the Magentic One pattern.
            name (str | None): The name of the orchestration.
            description (str | None): The description of the orchestration.
            input_transition (Callable): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable): A function that transforms the internal output message to the external
                output message.
        """
        self._manager = manager

        super().__init__(
            members=members,
            name=name,
            description=description,
            input_transition=input_transition,
            output_transition=output_transition,
        )

    @override
    async def _start(
        self,
        task: str | MagenticOneStartMessage | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the Magentic One pattern."""
        if isinstance(task, str):
            message = MagenticOneStartMessage(body=ChatMessageContent(AuthorRole.USER, content=task))
        elif isinstance(task, ChatMessageContent):
            message = MagenticOneStartMessage(body=task)

        target_actor_id = await runtime.get(self._get_orchestration_actor_type(internal_topic_type))
        await runtime.send_message(message, target_actor_id)

    @override
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> str:
        await self._register_members(runtime, internal_topic_type)
        await self._register_manager(runtime, internal_topic_type)
        await self._register_orchestration_actor(
            runtime,
            internal_topic_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )
        await self._add_subscriptions(runtime, internal_topic_type)

        return self._get_orchestration_actor_type(internal_topic_type)

    @override
    async def _register_members(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            MagenticOneAgentActor.register(
                runtime,
                self._get_agent_actor_type(agent, internal_topic_type),
                lambda agent=agent: MagenticOneAgentActor(agent, internal_topic_type),
            )
            for agent in self._members
            if isinstance(agent, Agent)
        ])
        # TODO(@taochen): Orchestration

    async def _register_manager(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Register the group chat manager."""
        await MagenticOneManagerActor.register(
            runtime,
            self._get_manager_actor_type(internal_topic_type),
            lambda: MagenticOneManagerActor(
                self._manager,
                internal_topic_type=internal_topic_type,
            ),
        )

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        await MagenticOneOrchestrationActor[self.t_external_in, self.t_external_out].register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: MagenticOneOrchestrationActor[self.t_external_in, self.t_external_out](
                internal_topic_type,
                self._input_transition,
                self._output_transition,
                manager_actor_type=self._get_manager_actor_type(internal_topic_type),
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                result_callback=result_callback,
            ),
        )

    async def _add_subscriptions(self, runtime: AgentRuntime, internal_topic_type: str) -> None:
        subscriptions: list[TypeSubscription] = []
        for agent in [member for member in self._members if isinstance(member, Agent)]:
            subscriptions.append(
                TypeSubscription(internal_topic_type, self._get_agent_actor_type(agent, internal_topic_type))
            )
            # TODO(@taochen): Orchestration
        subscriptions.append(TypeSubscription(internal_topic_type, self._get_manager_actor_type(internal_topic_type)))
        subscriptions.append(
            TypeSubscription(internal_topic_type, self._get_orchestration_actor_type(internal_topic_type))
        )
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])

    def _get_agent_actor_type(self, agent: Agent | str, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        if isinstance(agent, Agent):
            agent = agent.name
        return f"{agent}_{internal_topic_type}"

    def _get_manager_actor_type(self, internal_topic_type: str) -> str:
        """Get the actor type for the group chat manager.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{MagenticOneManagerActor.__name__}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{MagenticOneOrchestrationActor.__name__}_{internal_topic_type}"


# endregion MagenticOneOrchestration

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar, Union, cast, get_args

from autogen_core import AgentRuntime, BaseAgent, MessageContext

from semantic_kernel.agents.agent import Agent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


TInternalIn = TypeVar("TInternalIn")
TInternalOut = TypeVar("TInternalOut")
TExternalIn = TypeVar("TExternalIn")
TExternalOut = TypeVar("TExternalOut")


class OrchestrationActorBase(
    BaseAgent,
    Generic[
        TExternalIn,
        TInternalIn,
        TInternalOut,
        TExternalOut,
    ],
):
    """An orchestrator actor that is part of the orchestration.

    This actor is responsible for relaying external messages to the internal topic or actor and vice versa.
    """

    t_external_in: type[TExternalIn] = None
    t_internal_in: type[TInternalIn] = None
    t_internal_out: type[TInternalOut] = None
    t_external_out: type[TExternalOut] = None

    def __init__(
        self,
        internal_topic_type: str,
        input_transition: Callable[[TExternalIn], Awaitable[TInternalIn] | TInternalIn],
        output_transition: Callable[[TInternalOut], Awaitable[TExternalOut] | TExternalOut],
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            input_transition (Callable): A function that transforms the external input message to the internal
                input message.
            output_transition (Callable): A function that transforms the internal output message to the external
                output message.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output
                message to.
            result_callback (Callable | None): A function that is called when the result is available.
        """
        self._internal_topic_type = internal_topic_type
        self._input_transition = input_transition
        self._output_transition = output_transition

        self._external_topic_type = external_topic_type
        self._direct_actor_type = direct_actor_type
        self._result_callback = result_callback

        super().__init__(description="Orchestration Agent")

    def _set_types(self) -> None:
        """Set the external input and output types from the class arguments.

        This method can only be run after the class has been initialized because it relies on the
        `__orig_class__` attributes to determine the type parameters.

        This class requires all four type parameters to be explicitly given, no defaults are allowed,
        unlike in the `OrchestrationBase` class.

        This method will get the TInternalIn, TInternalOut types from the `__orig_bases__` attribute since
        they are specified in the generic type parameters of the derived class.

        When the orchestration actor is registered to the runtime by the orchestration, TExternalIn and
        TExternalOut must be explicitly given, since the orchestration already knows the types of the
        external input and output messages.
        """
        if all([
            self.t_external_in is not None,
            self.t_internal_in is not None,
            self.t_internal_out is not None,
            self.t_external_out is not None,
        ]):
            return

        args = get_args(self.__orig_bases__[0])

        if len(args) != 4:
            raise TypeError("OrchestrationActor must be have four type parameters.")

        # TInternalIn, TInternalOut are the second and third type parameters
        self.t_internal_in = args[1] if isinstance(args[1], type) else None
        self.t_internal_out = args[2] if isinstance(args[2], type) else None

        try:
            args = self.__orig_class__.__args__
            if len(args) != 2:
                raise TypeError("OrchestrationActor must have external input and output types.")
            self.t_external_in = args[0]
            self.t_external_out = args[1]
        except AttributeError as e:
            raise TypeError("OrchestrationActor must have concrete external input and output types.") from e

        if any([
            self.t_external_in is None or not isinstance(self.t_external_in, type),
            self.t_internal_in is None or not isinstance(self.t_internal_in, type),
            self.t_internal_out is None or not isinstance(self.t_internal_out, type),
            self.t_external_out is None or not isinstance(self.t_external_out, type),
        ]):
            raise TypeError("OrchestrationActor must have concrete types for all type parameters.")

    @override
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> None:
        """Handle incoming messages.

        This method is called when a message is received by the actor.

        Args:
            message (object): The incoming message.
            ctx (MessageContext): The message context.
        """
        self._set_types()

        if isinstance(message, self.t_internal_in):
            await self._handle_orchestration_input_message(message, ctx)
        elif isinstance(message, self.t_external_in):
            if inspect.isawaitable(self._input_transition):
                transition_message: TInternalIn = await self._input_transition(message)
            else:
                transition_message: TInternalIn = self._input_transition(message)  # type: ignore[no-redef]
            await self._handle_orchestration_input_message(transition_message, ctx)
        elif isinstance(message, self.t_internal_out):
            await self._handle_orchestration_output_message(message, ctx)
        else:
            # Since the orchestration actor subscribes to the external topic type,
            # it may receive messages that are not of the expected type.
            pass

    @abstractmethod
    async def _handle_orchestration_input_message(
        self,
        message: TExternalIn,
        ctx: MessageContext,
    ) -> None:
        """Handle the orchestration input message."""
        pass

    @abstractmethod
    async def _handle_orchestration_output_message(
        self,
        message: TInternalOut,
        ctx: MessageContext,
    ) -> None:
        """Handle the orchestration output message."""
        pass


class OrchestrationBase(
    ABC,
    Generic[
        TExternalIn,
        TInternalIn,
        TInternalOut,
        TExternalOut,
    ],
):
    """Base class for multi-agent orchestration."""

    t_external_in: type[TExternalIn] = None
    t_external_out: type[TExternalOut] = None

    def __init__(
        self,
        workers: list[Union[Agent, "OrchestrationBase"]],
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalIn], Awaitable[TInternalIn] | TInternalIn] | None = None,
        output_transition: Callable[[TInternalOut], Awaitable[TExternalOut] | TExternalOut] | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            workers (list[Union[Agent, OrchestrationBase]]): The list of agents or orchestrations to be used.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transition (Callable | None):  function that transforms the external input message to the
                internal input message.
            output_transition (Callable | None): A function that transforms the internal output message to the
                external output message.
        """
        self.name = name or f"{self.__class__.__name__}_{uuid.uuid4().hex}"
        self.description = description or "A multi-agent orchestration."

        if input_transition is None:

            def input_transition_func(input_message: TExternalIn) -> TInternalIn:
                return cast(TInternalIn, input_message)

            self._input_transition = input_transition_func
        else:
            self._input_transition = input_transition  # type: ignore[assignment]

        if output_transition is None:

            def output_transition_func(output_message: TInternalOut) -> TExternalOut:
                return cast(TExternalOut, output_message)

            self._output_transition = output_transition_func
        else:
            self._output_transition = output_transition  # type: ignore[assignment]

        self._workers = workers

    def _set_types(self) -> None:
        """Set the external input and output types from the class arguments.

        This method can only be run after the class has been initialized because it relies on the
        `__orig_class__` attributes to determine the type parameters.

        This method will first try to get the type parameters from the class itself. The `__orig_class__`
        attribute will contain the external input and output types if they are explicitly given, for example:
        ```
        class MyOrchestration(OrchestrationBase[TExternalIn, int, int, TExternalOut]):
            pass


        my_orchestration = MyOrchestration[str, str](...)
        ```
        If the type parameters are not explicitly given, for example when the TypeVars has defaults, for example:
        ```
        TExternalIn = TypeVar("TExternalIn", default=str)
        TExternalOut = TypeVar("TExternalOut", default=str)


        class MyOrchestration(OrchestrationBase[TExternalIn, int, int, TExternalOut]):
            pass


        my_orchestration = MyOrchestration(...)
        ```
        The type parameters can be inferred from the `__orig_bases__` attribute.
        """
        if all([self.t_external_in is not None, self.t_external_out is not None]):
            return

        try:
            args = self.__orig_class__.__args__
            if len(args) != 2:
                raise TypeError("Orchestration must have external input and output types.")
            self.t_external_in = args[0]
            self.t_external_out = args[1]
        except AttributeError:
            args = get_args(self.__orig_bases__[0])

            if len(args) != 4:
                raise TypeError("Orchestration must be subclassed with four type parameters.")
            self.t_external_in = args[0] if isinstance(args[0], type) else getattr(args[0], "__default__", None)
            self.t_external_out = args[3] if isinstance(args[3], type) else getattr(args[3], "__default__", None)

        if any([
            self.t_external_in is None or not isinstance(self.t_external_in, type),
            self.t_external_out is None or not isinstance(self.t_external_out, type),
        ]):
            raise TypeError("Orchestration must have concrete types for all type parameters.")

    async def invoke(
        self,
        task: str | ChatMessageContent | TExternalIn,
        runtime: AgentRuntime,
        time_out: int | None = None,
    ) -> TExternalOut:
        """Invoke the multi-agent orchestration and return the result.

        This method is a blocking call that waits for the orchestration to finish
        and returns the result.

        Args:
            task (str): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
            time_out (int | None): The timeout (seconds) for the orchestration. If None, wait indefinitely.
        """
        orchestration_result: TExternalOut | None = None
        orchestration_result_event = asyncio.Event()

        def result_callback(result: TExternalOut) -> None:
            nonlocal orchestration_result
            orchestration_result = result
            orchestration_result_event.set()

        self._set_types()

        # This unique topic type is used to isolate the orchestration run from others.
        internal_topic_type = uuid.uuid4().hex

        await self._prepare(
            runtime,
            internal_topic_type=internal_topic_type,
            result_callback=result_callback,
        )

        if isinstance(task, str):
            prepared_task = ChatMessageContent(role=AuthorRole.USER, content=task)
        elif isinstance(task, ChatMessageContent):
            prepared_task = task
        elif isinstance(task, self.t_external_in):
            if inspect.isawaitable(self._input_transition):
                prepared_task: TInternalIn = await self._input_transition(task)  # type: ignore[no-redef]
            else:
                prepared_task: TInternalIn = self._input_transition(task)  # type: ignore[no-redef]
        else:
            raise TypeError(
                f"Invalid task type: {type(task)}. Expected str, {self.t_external_in}, or ChatMessageContent."
            )

        await self._start(prepared_task, runtime, internal_topic_type)

        # Wait for the orchestration result
        if time_out is not None:
            await asyncio.wait_for(orchestration_result_event.wait(), timeout=time_out)
        else:
            await orchestration_result_event.wait()

        if orchestration_result is None:
            raise RuntimeError("Orchestration result is None.")
        return orchestration_result

    async def prepare(
        self,
        runtime: AgentRuntime,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> str:
        """Prepare the orchestration with the runtime.

        Args:
            runtime (AgentRuntime): The runtime environment for the agents.
            external_topic_type (str | None): The external topic type for the orchestration to broadcast
                and receive messages. If set, the orchestration will subscribe itself to this topic.
            direct_actor_type (str | None): The direct actor type for which the orchestration
                actor will relay the output message to.
            result_callback (Callable[[TExternalOut], None] | None):
                A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        self._set_types()
        # This unique topic type is used to isolate the orchestration run from others.
        internal_topic_type = uuid.uuid4().hex

        return await self._prepare(
            runtime,
            internal_topic_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )

    @abstractmethod
    async def _start(
        self,
        task: TInternalIn | ChatMessageContent,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Start the multi-agent orchestration.

        Args:
            task (TInternalIn | ChatMessageContent): The message to be sent to the orchestration.
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
        """
        pass

    @abstractmethod
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOut], None] | None = None,
    ) -> str:
        """Register the actors and orchestrations with the runtime and add the required subscriptions.

        Args:
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            result_callback (Callable[[TExternalOut], None] | None):
                A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        pass

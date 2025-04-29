# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, Union, get_args

from autogen_core import AgentRuntime, CancellationToken
from pydantic import Field
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    pass  # pragma: no cover
else:
    pass  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


DefaultExternalTypeAlias = Union[ChatMessageContent, list[ChatMessageContent]]

TExternalIn = TypeVar("TExternalIn", default=DefaultExternalTypeAlias)
TExternalOut = TypeVar("TExternalOut", default=DefaultExternalTypeAlias)


class OrchestrationResult(KernelBaseModel, Generic[TExternalOut]):
    """The result of the orchestration.

    This class is used to store the result of the orchestration and the context of the message.
    """

    value: TExternalOut | None = None
    event: asyncio.Event = Field(default_factory=lambda: asyncio.Event())
    cancellation_token: CancellationToken = Field(default_factory=lambda: CancellationToken())

    async def get(self, timeout: int | None = None) -> TExternalOut:
        """Get the result of the orchestration.

        Args:
            timeout (int | None): The timeout (seconds) for the orchestration. If None, wait indefinitely.

        Returns:
            TExternalOut: The result of the orchestration.
        """
        # TODO(@taochen): Cancel the task is an exception is raised inside the orchestration.
        try:
            if timeout is not None:
                await asyncio.wait_for(self.event.wait(), timeout=timeout)
            else:
                await self.event.wait()
        except asyncio.TimeoutError:
            self.cancellation_token.cancel()
            raise RuntimeError(f"The orchestration timed out after {timeout} seconds.")

        if self.value is None:
            if self.cancellation_token.is_cancelled():
                raise RuntimeError("The orchestration was canceled before it could complete.")
            raise RuntimeError("Result is None. An error may have occurred during the invocation.")
        return self.value

    def cancel(self) -> None:
        """Cancel the orchestration.

        This method will cancel the orchestration and set the cancellation token.
        Actors that have received messages will continue to process them, but no new messages will be sent.
        """
        if self.cancellation_token.is_cancelled():
            raise RuntimeError("The orchestration has already been canceled.")
        if self.event.is_set():
            raise RuntimeError("The orchestration has already been completed.")

        self.cancellation_token.cancel()
        self.event.set()


class OrchestrationBase(ABC, Generic[TExternalIn, TExternalOut]):
    """Base class for multi-agent orchestration."""

    t_external_in: type[TExternalIn] = None
    t_external_out: type[TExternalOut] = None

    def __init__(
        self,
        members: list[Agent],
        name: str | None = None,
        description: str | None = None,
        input_transform: Callable[[TExternalIn], Awaitable[DefaultExternalTypeAlias] | DefaultExternalTypeAlias]
        | None = None,
        output_transform: Callable[[DefaultExternalTypeAlias], Awaitable[TExternalOut] | TExternalOut] | None = None,
        observer: Callable[[str | DefaultExternalTypeAlias], Awaitable[None] | None] | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            members (list[Agent]): The list of agents or orchestrations to be used.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transform (Callable | None): A function that transforms the external input message.
            output_transform (Callable | None): A function that transforms the internal output message.
            observer (Callable | None): A function that is called when a response is produced by the agents.
        """
        if not members:
            raise ValueError("The members list cannot be empty.")
        self._members = members

        self.name = name or f"{self.__class__.__name__}_{uuid.uuid4().hex}"
        self.description = description or "A multi-agent orchestration."

        self._input_transform = input_transform or self._default_input_transform
        self._output_transform = output_transform or self._default_output_transform

        self._observer = observer

    def _set_types(self) -> None:
        """Set the external input and output types from the class arguments.

        This method can only be run after the class has been initialized because it relies on the
        `__orig_class__` attributes to determine the type parameters.

        This method will first try to get the type parameters from the class itself. The `__orig_class__`
        attribute will contain the external input and output types if they are explicitly given, for example:
        ```
        class MyOrchestration(OrchestrationBase[TExternalIn, TExternalOut]):
            pass


        my_orchestration = MyOrchestration[str, str](...)
        ```
        If the type parameters are not explicitly given, for example when the TypeVars has defaults, for example:
        ```
        TExternalIn = TypeVar("TExternalIn", default=str)
        TExternalOut = TypeVar("TExternalOut", default=str)


        class MyOrchestration(OrchestrationBase[TExternalIn, TExternalOut]):
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

            if len(args) != 2:
                raise TypeError("Orchestration must be subclassed with four type parameters.")
            self.t_external_in = args[0] if isinstance(args[0], type) else getattr(args[0], "__default__", None)
            self.t_external_out = args[1] if isinstance(args[1], type) else getattr(args[1], "__default__", None)

        if any([self.t_external_in is None, self.t_external_out is None]):
            raise TypeError("Orchestration must have concrete types for all type parameters.")

    async def invoke(
        self,
        task: str | DefaultExternalTypeAlias | TExternalIn,
        runtime: AgentRuntime,
    ) -> OrchestrationResult[TExternalOut]:
        """Invoke the multi-agent orchestration and return the result.

        This method is a blocking call that waits for the orchestration to finish
        and returns the result.

        Args:
            task (str, DefaultExternalTypeAlias, TExternalIn): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
        """
        self._set_types()

        orchestration_result = OrchestrationResult[self.t_external_out]()

        async def result_callback(result: DefaultExternalTypeAlias) -> None:
            nonlocal orchestration_result
            if inspect.iscoroutinefunction(self._output_transform):
                transformed_result = await self._output_transform(result)
            else:
                transformed_result = self._output_transform(result)

            orchestration_result.value = transformed_result
            orchestration_result.event.set()

        # This unique topic type is used to isolate the orchestration run from others.
        internal_topic_type = uuid.uuid4().hex

        await self._prepare(
            runtime,
            internal_topic_type=internal_topic_type,
            result_callback=result_callback,
        )

        if isinstance(task, str):
            prepared_task = ChatMessageContent(role=AuthorRole.USER, content=task)
        else:
            if inspect.iscoroutinefunction(self._input_transform):
                prepared_task: DefaultExternalTypeAlias = await self._input_transform(task)
            else:
                prepared_task: DefaultExternalTypeAlias = self._input_transform(task)

        asyncio.create_task(  # noqa: RUF006
            self._start(
                prepared_task,
                runtime,
                internal_topic_type,
                orchestration_result.cancellation_token,
            )
        )
        return orchestration_result

    @abstractmethod
    async def _start(
        self,
        task: DefaultExternalTypeAlias,
        runtime: AgentRuntime,
        internal_topic_type: str,
        cancellation_token: CancellationToken,
    ) -> None:
        """Start the multi-agent orchestration.

        Args:
            task (ChatMessageContent | list[ChatMessageContent]): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            cancellation_token (CancellationToken): The cancellation token for the orchestration.
        """
        pass

    @abstractmethod
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        result_callback: Callable[[DefaultExternalTypeAlias], Awaitable[None]] | None = None,
    ) -> None:
        """Register the actors and orchestrations with the runtime and add the required subscriptions.

        Args:
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            result_callback (Callable[[DefaultExternalTypeAlias], None] | None):
                A function that is called when the result is available.
        """
        pass

    def _default_input_transform(self, input_message: TExternalIn) -> DefaultExternalTypeAlias:
        """Default input transform function.

        This function transforms the external input message to chat message content(s).
        If the input message is already in the correct format, it is returned as is.

        Args:
            input_message (TExternalIn): The input message to be transformed.

        Returns:
            DefaultExternalTypeAlias: The transformed input message.
        """
        if isinstance(input_message, ChatMessageContent):
            return input_message

        if isinstance(input_message, list) and all(isinstance(item, ChatMessageContent) for item in input_message):
            return input_message

        if isinstance(input_message, self.t_external_in):
            return ChatMessageContent(
                role=AuthorRole.USER,
                content=json.dumps(input_message),
            )

        raise TypeError(f"Invalid input message type: {type(input_message)}. Expected {self.t_external_in}.")

    def _default_output_transform(self, output_message: DefaultExternalTypeAlias) -> TExternalOut:
        """Default output transform function.

        This function transforms the internal output message to the external output message.
        If the output message is already in the correct format, it is returned as is.

        Args:
            output_message (DefaultExternalTypeAlias): The output message to be transformed.

        Returns:
            TExternalOut: The transformed output message.
        """
        if self.t_external_out == DefaultExternalTypeAlias or self.t_external_out in get_args(DefaultExternalTypeAlias):
            if isinstance(output_message, ChatMessageContent) or (
                isinstance(output_message, list)
                and all(isinstance(item, ChatMessageContent) for item in output_message)
            ):
                return output_message
            raise TypeError(f"Invalid output message type: {type(output_message)}. Expected {self.t_external_out}.")

        if isinstance(output_message, ChatMessageContent):
            return json.loads(output_message.content, object_hook=lambda content: self.t_external_out(content))

        raise TypeError(f"Unable to transform output message of type {type(output_message)} to {self.t_external_out}.")

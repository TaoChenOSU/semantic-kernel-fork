# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.handoffs import HandoffConnection, HandoffOrchestration
from semantic_kernel.agents.runtime.in_process.in_process_runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class OrderStatusPlugin:
    @kernel_function
    def check_order_status(self, order_id: str) -> str:
        """Check the status of an order."""
        # Simulate checking the order status
        return f"Order {order_id} is shipped and will arrive in 2-3 days."


class OrderRefundPlugin:
    @kernel_function
    def process_refund(self, order_id: str, reason: str) -> str:
        """Process a refund for an order."""
        # Simulate processing a refund
        print(f"Processing refund for order {order_id} due to: {reason}")
        return f"Refund for order {order_id} has been processed successfully."


class OrderReturnPlugin:
    @kernel_function
    def process_return(self, order_id: str, reason: str) -> str:
        """Process a return for an order."""
        # Simulate processing a return
        print(f"Processing return for order {order_id} due to: {reason}")
        return f"Return for order {order_id} has been processed successfully."


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"{message.name}: {message.content}")


def human_response_function() -> ChatMessageContent:
    """Observer function to print the messages from the agents."""
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def main():
    """Main function to run the agents."""
    support_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="A customer support agent that triages issues.",
        instructions="Handle customer requests.",
        service=OpenAIChatCompletion(),
    )

    refund_agent = ChatCompletionAgent(
        name="RefundAgent",
        description="A customer support agent that handles refunds.",
        instructions="Handle refund requests.",
        service=OpenAIChatCompletion(),
        plugins=[OrderRefundPlugin()],
    )

    order_status_agent = ChatCompletionAgent(
        name="OrderStatusAgent",
        description="A customer support agent that checks order status.",
        instructions="Handle order status requests.",
        service=OpenAIChatCompletion(),
        plugins=[OrderStatusPlugin()],
    )

    order_return_agent = ChatCompletionAgent(
        name="OrderReturnAgent",
        description="A customer support agent that handles order returns.",
        instructions="Handle order return requests.",
        service=OpenAIChatCompletion(),
        plugins=[OrderReturnPlugin()],
    )

    handoffs: dict[str, list[HandoffConnection]] = {
        support_agent.name: [
            HandoffConnection(
                agent_name=refund_agent.name,
                description="Transfer to this agent if the issue is refund related",
            ),
            HandoffConnection(
                agent_name=order_status_agent.name,
                description="Transfer to this agent if the issue is order status related",
            ),
            HandoffConnection(
                agent_name=order_return_agent.name,
                description="Transfer to this agent if the issue is order return related",
            ),
        ],
        refund_agent.name: [
            HandoffConnection(
                agent_name=support_agent.name,
                description="Transfer to this agent if the issue is not refund related",
            )
        ],
        order_status_agent.name: [
            HandoffConnection(
                agent_name=support_agent.name,
                description="Transfer to this agent if the issue is not order status related",
            )
        ],
        order_return_agent.name: [
            HandoffConnection(
                agent_name=support_agent.name,
                description="Transfer to this agent if the issue is not order return related",
            )
        ],
    }

    handoff_orchestration = HandoffOrchestration(
        members=[
            support_agent,
            refund_agent,
            order_status_agent,
            order_return_agent,
        ],
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=human_response_function,
    )

    runtime = InProcessRuntime()
    runtime.start()

    orchestration_result = await handoff_orchestration.invoke(
        task="A customer is on the line.",
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(value)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from acp_sdk.client import Client
from acp_sdk.models import (
    Message,
    MessagePart,
)


async def client() -> None:
    while True:
        user_message = input(">>> ")
        user_message_input = Message(parts=[MessagePart(content=user_message, role="user")])

        async with Client(base_url="http://localhost:8000") as client:
            run = await client.run_sync(
                agent="acp_doc_agent", input=[user_message_input]
            )
            print(run.output[0].parts[0].content)

if __name__ == "__main__":
    asyncio.run(client())
from collections.abc import AsyncGenerator
import asyncio
from contextlib import AsyncExitStack

import beeai_framework
from acp_sdk import Message
from acp_sdk.models import MessagePart
from acp_sdk.server import Context, Server
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.backend import AssistantMessage, Role, UserMessage
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.mcp import MCPTool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class SessionManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.mcpdoctools = None
        self.llm = None
        self.initialized = False
        self.server_params = StdioServerParameters(
            command="python",
            args=['mcpdoctool.py'],
            transport="stdio",
        )
    
    async def initialize(self):
        if self.initialized:
            return
        
        try:
            # Setup stdio client with exit stack to manage resources
            stdio_context = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
            read_stream, write_stream = stdio_context
            
            # Setup session
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            
            # Initialize the connection
            await session.initialize()
            
            # Initialize LLM
            self.llm = ChatModel.from_name("ollama:qwen3:8b", ChatModelParameters(temperature=0))
            
            # Get tools
            self.mcpdoctools = await MCPTool.from_client(session)
            
            self.initialized = True
            print("Session initialized with tools:", self.mcpdoctools)
        except Exception as e:
            print(f"Error initializing session: {e}")
            # Clean up any resources that were set up
            await self.exit_stack.aclose()
            raise

    async def cleanup(self):
        await self.exit_stack.aclose()

    def to_framework_message(self, role: Role, content: str) -> beeai_framework.backend.Message:
        match role:
            case Role.USER:
                return UserMessage(content)
            case Role.ASSISTANT:
                return AssistantMessage(content)
            case _:
                raise ValueError(f"Unsupported role {role}")


# Create server and session manager
server = Server()
session_manager = SessionManager()

@server.agent()
async def chat_agent(input: list[Message], context: Context) -> AsyncGenerator:
    """
    The agent is an AI-powered conversational system with memory, supporting real-time search, Wikipedia lookups,
    and weather updates through integrated tools.
    """
    # Ensure session is initialized
    if not session_manager.initialized:
        print("Session not initialized, initializing now...")
        await session_manager.initialize()
    
    # Create agent with memory and tools
    agent = ReActAgent(
        llm=session_manager.llm, 
        tools=session_manager.mcpdoctools, 
        memory=TokenMemory(session_manager.llm)
    )
    
    # Process messages
    framework_messages = [
        session_manager.to_framework_message(Role(message.parts[0].role), str(message)) 
        for message in input
    ]
    await agent.memory.add_many(framework_messages)
    
    async for data, event in agent.run():
        match (data, event.name):
            case (ReActAgentUpdateEvent(), "partial_update"):
                update = data.update.value
                if not isinstance(update, str):
                    update = update.get_text_content()
                match data.update.key:
                    case "thought" | "tool_name" | "tool_input" | "tool_output":
                        yield {data.update.key: update}
                    case "final_answer":
                        yield MessagePart(content=update, role="assistant")


if __name__ == "__main__":
    # Initialize session before starting the server
    loop = asyncio.get_event_loop()
    loop.create_task(session_manager.initialize())
    
    try:
        server.run()
    finally:
        # Ensure cleanup on exit
        if not loop.is_closed():
            loop.run_until_complete(session_manager.cleanup())
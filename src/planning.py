#!/usr/bin/env python3
from dotenv import load_dotenv

load_dotenv()

from prompts import PLANNER_PROMPT

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

async def run_planning_agent():
    async with streamablehttp_client("http://localhost:4444/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            agent = create_react_agent("openai:o3-mini", tools)
            response = await agent.ainvoke({"messages": f"{PLANNER_PROMPT}"})
            print("Agent response: ", response)
            return response

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_planning_agent())

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent / ".env")

SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_server" / "server.py")


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            model = GigaChat(
                model="GigaChat-2-Max",
                streaming=False,
                verify_ssl_certs=False,
            )
            agent = create_react_agent(model, tools)

            query = (
                "Профилируй файл data/raw/constrained.csv. "
                "Покажи результат в виде таблицы, не меняй форматирование."
            )
            response = await agent.ainvoke({"messages": [("user", query)]})
            print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
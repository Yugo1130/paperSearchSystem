import asyncio

from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings


# =========================
# LLM 設定（Ollama / gpt-oss:20b）
# =========================

llm = Ollama(
    model="llama3.1:8b",
    base_url="http://172.28.192.1:11434",
    request_timeout=300.0,
)

Settings.llm = llm


# =========================
# Tool 定義（公式例と同等）
# =========================

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    print(f"Multiplying {a} and {b}")
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers."""
    print(f"Adding {a} and {b}")
    return a + b


multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
    description="Multiply two numbers."
)

add_tool = FunctionTool.from_defaults(
    fn=add,
    name="add",
    description="Add two numbers."
)


# =========================
# Agent 初期化
# =========================

agent = ReActAgent(
    tools=[multiply_tool, add_tool],
    llm=Settings.llm,
    verbose=True,
    context="You are a helpful assistant that can use tools."
)


# =========================
# Agent 実行（公式例準拠）
# =========================

async def run_query(query: str):
    print("\nUSER:", query)
    print("Thinking...\n")

    handler = agent.run(query)

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)

    result = await handler

    print("\n" + "=" * 40)
    print("FINAL ANSWER")
    print("=" * 40)
    print(result)


# =========================
# main
# =========================

async def main():
    await run_query("What is 20 + (2 * 4)?")
    await run_query("Compute 5 + 3 * 2 please.")

if __name__ == "__main__":
    asyncio.run(main())

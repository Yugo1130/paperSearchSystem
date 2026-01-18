# RAGクエリシステムの実行スクリプト
# 機能:
# - ChromaDB に保存された研究文書を検索
# - ReAct Agent が必要に応じて DB を検索
# - streaming で回答を表示

import sys
import requests
import chromadb
import asyncio
from chromadb.errors import NotFoundError

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent, AgentStream


# =========================
# 設定
# =========================

OLLAMA_HOST = "172.28.192.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

LLM_MODEL_NAME = "llama3.1:8b"
EMBED_MODEL_NAME = "mxbai-embed-large"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_collection"

LLM_TIMEOUT_SECONDS = 300.0
TOKEN_SIZE_LIMIT = 8192


# =========================
# RAG エンジン初期化
# =========================

def initialize_rag_engine():
    print("--- RAGエンジンを初期化中 ---")

    llm = LlamaIndexOllama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=LLM_TIMEOUT_SECONDS,
        context_window=TOKEN_SIZE_LIMIT,
        additional_kwargs={"num_ctx": TOKEN_SIZE_LIMIT},
    )

    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = TOKEN_SIZE_LIMIT

    try:
        db = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            chroma_collection = db.get_collection(COLLECTION_NAME)
        except NotFoundError:
            print(f"エラー: コレクション '{COLLECTION_NAME}' が見つかりません")
            return None

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = index.as_query_engine(similarity_top_k=5)

        print(">> RAGクエリエンジンのロード完了")
        return query_engine

    except requests.exceptions.ConnectionError:
        print("Ollama に接続できません")
        return None
    except Exception as e:
        print(f"初期化エラー: {e}")
        return None


# =========================
# Tool: 研究DB検索
# =========================
def make_search_research_db(rag_engine):
    def search_research_db(query: str) -> str:
        response = rag_engine.query(query)

        if not response.source_nodes:
            return (
                "検索結果が見つかりませんでした。"
                "別のキーワードで再検索する必要があります。"
            )

        texts = []
        for i, node in response.source_nodes:
            text = node.node.text.strip()
            if text:
                texts.append(text)

        if not texts:
            return "関連文書は存在しますが、有効な本文を取得できませんでした。"

        return "\n\n".join(texts[:3])

    return search_research_db




# =========================
# Agent 初期化
# =========================

def initialize_agent(rag_engine):
    search_fn = make_search_research_db(rag_engine)

    search_tool = FunctionTool.from_defaults(
        fn=search_fn,
        name="search_research_db",
        description=(
            "研究論文データベースを検索するツール。"
            "質問に答えるために外部情報が必要な場合のみ使用する。"
            "入力は短いキーワードまたは短文にすること。"
        ),
    )

    agent = ReActAgent(
        tools=[search_tool],
        llm=Settings.llm,
        verbose=True,
        context=(
            "あなたは研究アシスタントです。"
            "必要に応じて research DB を検索し、得られた情報を基に回答してください。"
        ),
    )

    return agent


# =========================
# Agent 実行（streaming）
# =========================

async def run_agent_chat(agent, user_prompt: str):
    print("\nThinking... (Agentが思考中...)\n")

    # try:
    handler = agent.run(user_prompt)

    # ストリーミング出力
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)

    response = await handler

    print("\n" + "=" * 50)
    print("【AIの回答 完了】")
    print("=" * 50)

    # except Exception as e:
    #     print(f"\nAgent実行中にエラーが発生しました: {e}")


# =========================
# main
# =========================

async def main():
    rag_engine = initialize_rag_engine()
    if rag_engine is None:
        sys.exit(1)

    agent = initialize_agent(rag_engine)

    print("\nシステム準備完了。質問を入力してください。")
    print("終了するには 'q', 'quit', 'exit'")

    while True:
        # try:
            user_input = input("\nあなた: ")

            if user_input.lower() in ("q", "quit", "exit"):
                print("終了します。")
                break

            if not user_input.strip():
                continue

            await run_agent_chat(agent, user_input)

        # except KeyboardInterrupt:
        #     print("\n中断しました。")
        #     break
        # except EOFError:
        #     print("\nEOF")
        #     break

if __name__ == "__main__":
    asyncio.run(main())

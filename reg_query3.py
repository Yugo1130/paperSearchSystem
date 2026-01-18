#!/usr/bin/env python3
"""
完成版 RAG クエリシステム（検索結果を LLM に渡す版）
- 関連語生成 → ベクトル検索 → 回答生成
- ReactAgent 不使用
- grep ではなく embedding ベース
"""

import sys
import requests
import chromadb
from chromadb.errors import NotFoundError

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding


# ==============================
# 設定
# ==============================

OLLAMA_HOST = "172.28.192.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

LLM_MODEL_NAME = "gpt-oss:20b"
EMBED_MODEL_NAME = "mxbai-embed-large"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_collection"

TOKEN_SIZE_LIMIT = 8192
LLM_TIMEOUT_SECONDS = 300.0

SIMILARITY_TOP_K = 8


# ==============================
# 初期化
# ==============================

def initialize_engine():
    print("=== RAGエンジン初期化中 ===")

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
            collection = db.get_collection(COLLECTION_NAME)
        except NotFoundError:
            print(f"コレクション '{COLLECTION_NAME}' が見つかりません")
            return None, None

        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K
        )

        print(">>> 初期化完了")
        return query_engine, llm

    except requests.exceptions.ConnectionError:
        print("Ollama に接続できません")
        return None, None


# ==============================
# 関連語生成
# ==============================

def generate_related_terms(llm, user_query: str) -> str:
    """
    検索用の関連語・同義語・表記ゆれを生成
    """

    prompt = f"""
        次の質問に対して、関連する技術・手法・モデル名などを単語で最大20個挙げてください。

        条件:
        - 表記ゆれは禁止（ほぼ同じ意味の言葉は出さない）
        - 具体的な用語
        - 抽象概念は禁止

        出力形式:
        - 数字などは記述せずに列挙してください

        質問:
        {user_query}
        """

    response = llm.complete(prompt)
    return response.text.strip()


# ==============================
# RAG 実行
# ==============================

def run_rag(query_engine, llm, user_query: str):
    print("\n--- 関連語生成中 ---")
    related_terms = generate_related_terms(llm, user_query)
    print(related_terms)

    # 1. まず検索
    print("\n--- 検索中 ---")
    response = query_engine.query(related_terms)

    # 2. 検索結果をまとめて文字列化
    docs_text = ""
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.metadata
        fname = meta.get("file_name", "不明")
        page = meta.get("page_label", "-")
        preview = node.text.replace("\n", " ")[:300]  # 先頭300文字だけ
        docs_text += f"[{i}] {fname} P.{page}: {preview}\n"

    # 3. LLM に質問 + 関連語 + 検索結果を渡す
    combined_prompt = f"""
        あなたは研究室内の過去の論文データベースを活用してユーザの研究アイデアに対して有益なアドバイスを行うAIです
        回答は必ず日本語で行ってください

        禁止事項:
        - 存在しない論文や著者名を捏造しない
        - 関連論文が見つからない場合は正直に「直接的な先行研究は見当たりませんでした」と答える

        質問:
        {user_query}

        文献情報:
        {docs_text}
        """


    print("\n--- 回答生成中 ---")
    final_response = llm.complete(combined_prompt)
    print("\n" + "=" * 60)
    print("【AIの回答】")
    print(final_response.text.strip())
    print("=" * 60)

    # 参照ドキュメントも表示
    print("\n【参照ドキュメント】")
    if not response.source_nodes:
        print("なし")
    else:
        for i, node in enumerate(response.source_nodes, 1):
            meta = node.metadata
            fname = meta.get("file_name", "不明")
            page = meta.get("page_label", "-")
            score = node.score or 0.0
            preview = node.text[:60].replace("\n", " ") + "..."
            print(f"[{i}] {fname} P.{page} | 類似度 {score:.4f}")
            # print(f"    {preview}")


# ==============================
# メイン
# ==============================

def main():
    query_engine, llm = initialize_engine()
    if query_engine is None:
        sys.exit(1)

    print("\n質問を入力してください（exit / quit / q で終了）")

    while True:
        try:
            q = input("\nあなた: ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                print("終了します")
                break
            if not q:
                continue

            run_rag(query_engine, llm, q)

        except KeyboardInterrupt:
            print("\n中断")
            break
        except Exception as e:
            print(f"エラー: {e}")
            break


if __name__ == "__main__":
    main()

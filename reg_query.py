# RAGクエリシステムの実行スクリプト
# 機能: ChromaDBからインデックスを読み込み、Ollama(LLM)を使用して回答を生成する

import os
import sys
import requests
import chromadb
from chromadb.errors import NotFoundError

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- 1. 設定 (Configuration) ---

# Ollama 接続情報
# ※ WSL2や別サーバーの場合はIPを指定。ローカル実行なら "localhost" や "127.0.0.1"
OLLAMA_HOST = "172.28.192.1" 
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# モデル情報
LLM_MODEL_NAME = "gpt-oss-ja-lab:latest"  # 回答生成用LLM
EMBED_MODEL_NAME = "mxbai-embed-large"    # ベクトル化用モデル

# データベース情報
CHROMA_PATH = "./chroma_db"               # 永続化データの保存先
COLLECTION_NAME = "research_collection"   # コレクション名

# LLM設定
LLM_TIMEOUT_SECONDS = 300.0  # タイムアウト時間（秒）
TOKEN_SIZE_LIMIT = 8192      # コンテキストウィンドウの上限（入力+出力）


def initialize_rag_engine():
    """
    RAGクエリエンジンを初期化し、ロードする関数
    
    Returns:
        query_engine: 初期化されたクエリエンジンオブジェクト。失敗時は None。
    """
    print("--- RAGエンジンを初期化中 ---")

    # 1. Ollama LLM の設定
    # context_window: LlamaIndex側の計算用リミッター
    # num_ctx: Ollamaサーバー側のメモリ確保用パラメータ
    # ※ 両方を一致させることで、予期せぬ切り捨てやエラーを防ぐ
    llm = LlamaIndexOllama(
        model=LLM_MODEL_NAME, 
        base_url=OLLAMA_BASE_URL,
        request_timeout=LLM_TIMEOUT_SECONDS,
        context_window=TOKEN_SIZE_LIMIT,
        additional_kwargs={
            "num_ctx": TOKEN_SIZE_LIMIT, 
        }
    )
    
    # 2. 埋め込みモデルの設定
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME, 
        base_url=OLLAMA_BASE_URL,
    )
    
    # 3. グローバル設定の上書き (v0.10以降の推奨設定)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = TOKEN_SIZE_LIMIT

    try:
        # 4. ChromaDB クライアントの接続
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # コレクションの存在確認
        try:
            chroma_collection = db.get_collection(COLLECTION_NAME)
        except NotFoundError:
            print(f"エラー: コレクション '{COLLECTION_NAME}' がパス '{CHROMA_PATH}' に見つかりません。")
            print("ヒント: 先にインデックス作成スクリプト (index_building.py) を実行してください。")
            return None

        # 5. ベクトルストアとインデックスのロード
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 既存のDBからインデックスを再構築（コストはかかりません）
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        
        # 6. クエリエンジンの作成
        # similarity_top_k: 検索で取得する類似チャンクの数
        # 注意: チャンクサイズが1024の場合、k=10だと約10,000トークンになり、
        #       TOKEN_SIZE_LIMIT(8192)を超えてエラーまたは切り捨てが発生する可能性があります。
        #       安全のため 5 程度に設定するか、チャンクサイズに応じて調整してください。
        query_engine = index.as_query_engine(
            similarity_top_k=5  # 推奨: 5〜8程度 (8192トークン制限の場合)
        )
        
        print(">> RAGクエリエンジンのロード完了")
        return query_engine

    except requests.exceptions.ConnectionError:
        print(f"接続エラー: Ollamaサーバー ({OLLAMA_BASE_URL}) に接続できません。")
        print("Ollamaが起動しているか、IPアドレスが正しいか確認してください。")
        return None
    except Exception as e:
        print(f"予期せぬエラー: インデックスのロード中に問題が発生しました。\n詳細: {e}")
        return None


def run_rag_query(query_engine, user_prompt: str):
    """
    RAGクエリを実行し、結果を表示する関数
    """
    print(f"\nThinking... (質問: {user_prompt})")
    
    try:
        # クエリ実行（検索 + 生成）
        response = query_engine.query(user_prompt)

        print("\n" + "="*50)
        print("【AIの回答】")
        print(response.response.strip())
        print("="*50)

        # 参照元の表示 (Source Nodes)
        print("\n--- 参照したドキュメント ---")
        if not response.source_nodes:
            print("なし (LLMの知識のみで回答、または類似文書が見つかりませんでした)")
        
        for i, node in enumerate(response.source_nodes):
            filename = node.metadata.get('file_name', '不明なファイル')
            page_label = node.metadata.get('page_label', '-') # PDFなどの場合
            score = node.score if node.score is not None else 0.0
            
            # コンテンツの先頭50文字をプレビュー表示
            preview = node.text[:50].replace('\n', ' ') + "..."
            
            print(f"[{i+1}] {filename} (P.{page_label}) | 類似度: {score:.4f}")
            # print(f"    内容: {preview}") # 必要であればコメントアウトを解除

        return response.response.strip()

    except Exception as e:
        print(f"クエリ実行中にエラーが発生しました: {e}")
        return "エラーが発生しました。"


if __name__ == "__main__":
    # エンジンの初期化
    rag_engine = initialize_rag_engine()
    
    if rag_engine is None:
        sys.exit(1) # 初期化失敗時は終了コード1で終了
        
    print("\nシステム準備完了。質問を入力してください。")
    print("終了するには 'q', 'quit', 'exit' のいずれかを入力してください。")
    
    # 対話ループ
    while True:
        try:
            user_input = input("\nあなた: ")
            
            # 終了コマンドのチェック
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("終了します。")
                break
                
            # 空入力のスキップ
            if not user_input.strip():
                continue
                
            # クエリ実行
            run_rag_query(rag_engine, user_input)
            
        except KeyboardInterrupt:
            print("\n\n(中断) 対話セッションを終了します。")
            break
        except EOFError:
            print("\n(EOF) 対話セッションを終了します。")
            break
        except Exception as e:
            print(f"\n予期せぬエラーが発生しました: {e}")
            break
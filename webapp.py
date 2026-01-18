#!/usr/bin/env python3
import sys
import requests
import chromadb
from chromadb.errors import NotFoundError
from flask import Flask, request, jsonify, render_template_string

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
# Flask アプリケーション設定
# ==============================

app = Flask(__name__)

# グローバル変数としてエンジンを保持
query_engine = None
llm_client = None

# ==============================
# ロジック関数 (初期化・RAG)
# ==============================

def initialize_engine():
    """RAGエンジンの初期化（起動時に1回だけ実行）"""
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
            print(f"Error: コレクション '{COLLECTION_NAME}' が見つかりません")
            return None, None

        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

        q_engine = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K
        )

        print(">>> 初期化完了")
        return q_engine, llm

    except requests.exceptions.ConnectionError:
        print("Error: Ollama に接続できません")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def generate_related_terms(llm, user_query: str) -> str:
    """検索用の関連語生成"""
    prompt = f"""
        次の質問に対して、関連する技術・手法・モデル名などを単語で最大20個挙げてください。
        条件:
        - 表記ゆれは禁止
        - 具体的な用語
        - 抽象概念は禁止
        出力形式:
        - 数字などは記述せずに列挙してください
        質問:
        {user_query}
        """
    response = llm.complete(prompt)
    return response.text.strip()


def process_rag_query(user_query: str):
    """RAGプロセスの実行"""
    global query_engine, llm_client
    
    if not query_engine or not llm_client:
        return {"error": "Engine not initialized"}, 500

    # 1. 関連語生成
    print(f"Generating terms for: {user_query}")
    related_terms = generate_related_terms(llm_client, user_query)
    
    # 2. 検索実行 (関連語を使用)
    print(f"Searching with terms: {related_terms}")
    response = query_engine.query(related_terms)

    # 3. コンテキスト構築
    docs_text = ""
    source_list = []
    
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.metadata
        fname = meta.get("file_name", "不明")
        page = meta.get("page_label", "-")
        score = node.score or 0.0
        preview_text = node.text.replace("\n", " ")
        
        # プロンプト用テキスト
        docs_text += f"[{i}] {fname} P.{page}: {preview_text[:300]}\n"
        
        # フロントエンド返却用データ
        source_list.append({
            "filename": fname,
            "page": page,
            "score": round(score, 4),
            "preview": preview_text[:100] + "..."
        })

    # 4. 回答生成
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

    print("Generating final answer...")
    final_response = llm_client.complete(combined_prompt)

    return {
        "answer": final_response.text.strip(),
        "related_terms": related_terms,
        "sources": source_list
    }, 200

# ==============================
# ルーティング (API & UI)
# ==============================

@app.route('/')
def index():
    """簡易Webインターフェース"""
    html = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Research Assistant</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 100px; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            #result { margin-top: 20px; white-space: pre-wrap; line-height: 1.6; }
            .meta-info { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; font-size: 0.9em; border-left: 4px solid #28a745; }
            .source-item { margin-bottom: 5px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Research RAG Assistant</h1>
            <p>研究アイデアや質問を入力してください</p>
            <textarea id="query" placeholder="例: WebAssemblyを用いた分散コンピューティングについて教えて"></textarea>
            <button id="sendBtn" onclick="sendQuery()">送信</button>
            
            <div id="loading" class="loading" style="display:none;">回答生成中... (これには時間がかかります)</div>
            
            <div id="result-container" style="display:none;">
                <h3>AIの回答:</h3>
                <div id="result"></div>
                
                <div class="meta-info">
                    <h4>検索に使用した関連語:</h4>
                    <div id="terms"></div>
                    
                    <h4>参照ドキュメント:</h4>
                    <div id="sources"></div>
                </div>
            </div>
        </div>

        <script>
            async function sendQuery() {
                const query = document.getElementById('query').value;
                if (!query) return;

                const btn = document.getElementById('sendBtn');
                const loading = document.getElementById('loading');
                const resultContainer = document.getElementById('result-container');
                
                btn.disabled = true;
                loading.style.display = 'block';
                resultContainer.style.display = 'none';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('result').innerText = data.answer;
                        document.getElementById('terms').innerText = data.related_terms;
                        
                        const sourcesDiv = document.getElementById('sources');
                        sourcesDiv.innerHTML = '';
                        if (data.sources.length === 0) {
                            sourcesDiv.innerHTML = '<div>なし</div>';
                        } else {
                            data.sources.forEach((src, idx) => {
                                const div = document.createElement('div');
                                div.className = 'source-item';
                                div.innerHTML = `<strong>[${idx+1}] ${src.filename}</strong> (P.${src.page}) <br><span style="color:#666; font-size:0.8em;">類似度: ${src.score}</span>`;
                                sourcesDiv.appendChild(div);
                            });
                        }
                        resultContainer.style.display = 'block';
                    } else {
                        alert('エラーが発生しました: ' + (data.error || 'Unknown error'));
                    }
                } catch (e) {
                    alert('通信エラー: ' + e);
                } finally {
                    btn.disabled = false;
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """APIエンドポイント"""
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    result, status_code = process_rag_query(user_query)
    return jsonify(result), status_code

# ==============================
# メイン実行
# ==============================

if __name__ == "__main__":
    # サーバー起動前にエンジンを初期化
    q_engine, llm = initialize_engine()
    
    if q_engine is None:
        print("初期化に失敗しました。設定を確認してください。")
        sys.exit(1)
        
    # グローバル変数にセット
    query_engine = q_engine
    llm_client = llm

    # サーバー起動 (開発用サーバー)
    # 外部公開する場合は host='0.0.0.0' に変更
    app.run(host='0.0.0.0', port=5000, debug=False)
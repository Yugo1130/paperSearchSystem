import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

OLLAMA_HOST = "172.28.192.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
EMBED_MODEL_NAME = "mxbai-embed-large"

# 論文PDFが入っているディレクトリのパス
DATA_DIR = "./research_papers"
# ベクトルDBを保存するディレクトリ
CHROMA_PATH = "./chroma_db"

# OllamaをLlamaIndexの埋め込みモデルとして設定
# NOTE: LlamaIndexのデフォルトではOpenAIが使われるため、明示的にOllamaを設定
embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL_NAME, 
    base_url=OLLAMA_BASE_URL, # OllamaのデフォルトURL
    is_embedding_model=True           # これが埋め込みモデルであることを伝える
)

# データ読み込み
print(f"[{DATA_DIR}] 内のPDFを再帰的に読み込み中...")
# SimpleDirectoryReaderがPDFファイルを自動で処理
reader = SimpleDirectoryReader(
    input_dir=DATA_DIR,
    recursive=True, # サブディレクトリも再帰的に探索
    required_exts=[".pdf"] # PDFファイルのみ対象
)
documents = reader.load_data()

print(f"{len(documents)} 件のドキュメントを読み込みました。")

# ベクトルデータベースの初期化
# ChromaDBクライアントをローカルファイルとして起動
db = chromadb.PersistentClient(path=CHROMA_PATH)

# コレクション名（任意の名前）
COLLECTION_NAME = "research_collection"

# 既存のコレクションがあれば削除して再作成
try:
    db.delete_collection(COLLECTION_NAME)
    print(f"既存のコレクション [{COLLECTION_NAME}] を削除しました。")
except chromadb.errors.NotFoundError:
    pass

print(f"新しいコレクション [{COLLECTION_NAME}] を作成中...")
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

# LlamaIndexとChromaDBを接続
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# インデックスの作成と格納
print("インデックスを作成し、ChromaDBに格納中...")
# VectorStoreIndex.from_documentsが以下の処理を全て自動で実行します
# a. チャンク化
# b. mxbai-embed-large (Ollama)でベクトル化
# c. ChromaDBへ保存
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

print(f"\nインデックスの構築が完了しました。データは [{CHROMA_PATH}] に保存されました。")
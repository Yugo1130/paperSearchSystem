import requests
import json

# Ollamaサーバーの情報
OLLAMA_HOST = "172.28.192.1"
OLLAMA_PORT = 11434 # 指定されたポート番号
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
MODEL_NAME = "gpt-oss:20b" # 指定されたモデル名

# APIエンドポイント
GENERATE_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

def generate_text_with_ollama(prompt: str) -> str | None:
    """
    Ollama APIにリクエストを送信し、テキスト生成結果を取得する関数 (ストリームなし)。

    Args:
        prompt: モデルへの入力プロンプト。

    Returns:
        生成されたテキスト、またはエラー時にNone。
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # ストリームせず、一度に全応答を受け取る
    }
    headers = {"Content-Type": "application/json"}

    try:
        # POSTリクエストを送信
        response = requests.post(GENERATE_API_URL, headers=headers, json=payload, timeout=60) # タイムアウトを適宜設定
        response.raise_for_status() # ステータスコードが200番台以外なら例外を発生させる

        # レスポンスをJSONとして解析
        response_data = response.json()

        # 生成されたテキストを取得
        generated_text = response_data.get("response")
        if generated_text:
            return generated_text.strip()
        else:
            print("エラー: レスポンスに 'response' キーが含まれていません。")
            print("レスポンス内容:", response_data)
            return None

    except requests.exceptions.ConnectionError as e:
        print(f"エラー: Ollamaサーバー ({OLLAMA_BASE_URL}) に接続できませんでした。")
        print(f"詳細: {e}")
        print("Ollamaサーバーが起動しているか、ネットワーク設定（ファイアウォール等）を確認してください。")
        return None
    except requests.exceptions.Timeout as e:
        print(f"エラー: Ollamaサーバーへのリクエストがタイムアウトしました。")
        print(f"詳細: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"エラー: Ollama APIへのリクエスト中にエラーが発生しました。")
        print(f"URL: {e.request.url if e.request else 'N/A'}")
        if e.response is not None:
            print(f"ステータスコード: {e.response.status_code}")
            try:
                # エラーレスポンスの内容を表示試行
                error_content = e.response.json()
                print(f"エラー内容: {error_content}")
            except json.JSONDecodeError:
                print(f"エラーレスポンス (テキスト): {e.response.text}")
        else:
            print(f"詳細: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"エラー: Ollama APIからのレスポンスのJSON解析に失敗しました。")
        print(f"レスポンス内容: {response.text}")
        print(f"詳細: {e}")
        return None

if __name__ == "__main__":
    # プロンプトの例
    user_prompt = "夕日はなぜ赤い？"

    print(f"モデル '{MODEL_NAME}' にプロンプトを送信します...")
    print(f"プロンプト: {user_prompt}")
    print("-" * 30)

    # テキスト生成を実行
    generated_response = generate_text_with_ollama(user_prompt)

    if generated_response:
        print("モデルからの応答:")
        print(generated_response)
    else:
        print("テキスト生成に失敗しました。")
import asyncio
import sys
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

load_dotenv()

# Azure OpenAI クライアント初期化
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# MCPツールをホストするサーバースクリプト
server_params = StdioServerParameters(
    command="python",
    args=["server.py"]
)

def get_prompt(name: str) -> str:
    return (
        f"Create a dialogue between Mary and {name}. "
        f"There should be 6 messages in total. "
        f"{name} should yell every time and Mary should use sarcasm."
    )

# MCPツールをOpenAI向けに変換
async def get_tools_from_mcp(session: ClientSession):
    tool_defs = await session.list_tools()  # -> ToolsListResponse( meta=None, nextCursor=None, tools=[Tool(...), ... ] )
    
    # tool_defs.tools に実際のツール一覧が入っている
    mcp_tools = tool_defs.tools

    tools = []
    for t in mcp_tools:
        # t は Tool(name=..., description=..., inputSchema=...)
        name = t.name
        desc = t.description or "No description"
        params = t.inputSchema or {}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params
            }
        })

    return tools


# 実行関数
async def run_agent(name: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ツール定義を取得
            tools = await get_tools_from_mcp(session)

            # 最初のユーザー メッセージ
            messages = [
                {"role": "user", "content": get_prompt(name)}
            ]

            while True:
                response = await client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                choice = response.choices[0]
                msg = choice.message

                # もし終了理由が tool_calls なら、tool_calls を処理する
                if choice.finish_reason == "tool_calls":
                    # 複数ツール呼び出しがあり得る
                    calls = msg.tool_calls
                    if not calls:
                        # いちおう安全策
                        print("No calls found, but finish_reason=tool_calls. Exiting.")
                        break

                    # 各ツール呼び出しを順に処理
                    for call_info in calls:
                        fn_name = call_info.function.name
                        fn_args_json = call_info.function.arguments
                        # JSON文字列をPython dictに変換
                        fn_args = json.loads(fn_args_json)

                        print(f"[DEBUG] Calling tool: {fn_name} with {fn_args}")

                        # MCPツール実行
                        result = await session.call_tool(fn_name, fn_args)
                        
                        # MCP結果からテキストを抽出
                        # CallToolResultオブジェクトからテキストを取得
                        result_text = result.content[0].text if hasattr(result, 'content') else str(result)

                        # モデルに結果を返す： assistant role のメッセージを追加
                        messages.append({
                            "role": "assistant", 
                            "tool_calls": [
                                {
                                    "id": call_info.id,
                                    "type": "function",
                                    "function": {
                                        "name": fn_name,
                                        "arguments": fn_args_json
                                    }
                                }
                            ]
                        })
                        
                        # toolメッセージを追加
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_info.id,
                            "name": fn_name,
                            "content": result_text
                        })

                    # ループ先頭に戻り、再度 create() 呼び出しする
                    continue

                # それ以外の終了理由（通常の回答など）
                print("\n=== Dialogue Output ===\n")
                print(msg.content)
                break


# 実行部分
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <name>")
        sys.exit(1)

    name = sys.argv[1]
    asyncio.run(run_agent(name))

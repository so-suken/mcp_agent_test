import asyncio
import sys
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from typing import Any, Dict, List, Literal, Optional

load_dotenv()

class MCPClient:
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    This class manages the connection and communication with the MCP server.
    """

    def __init__(self, server_params: StdioServerParameters):
        """Initialize the MCP client with server parameters"""
        self.server_params = server_params
        self.session = None
        self._client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def connect(self):
        """Establishes connection to MCP server"""
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()
    
    async def get_available_tools(self) -> List[Any]:
        """
        Retrieve a list of available tools from the MCP server.
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        tool_defs = await self.session.list_tools()
        return tool_defs.tools
    
    async def get_openai_tools(self) -> List[Dict]:
        """
        Convert MCP tools to the format required by OpenAI API.
        """
        mcp_tools = await self.get_available_tools()
        
        tools = []
        for t in mcp_tools:
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
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Call a specific tool with the given arguments.
        
        Args:
            tool_name: The name of the tool to call
            arguments: The arguments to pass to the tool
            
        Returns:
            The response from the tool
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result


class DialogueAgent:
    """
    An agent that manages dialogue generation using LLMs and MCP tools.
    """
    
    def __init__(self, client: AsyncAzureOpenAI, deployment_name: str, mcp_client: MCPClient):
        """Initialize the agent with OpenAI client and MCP client"""
        self.client = client
        self.deployment_name = deployment_name
        self.mcp_client = mcp_client
    
    def get_prompt(self, name: str) -> str:
        """Generate the initial prompt for the dialogue."""
        return (
            f"Create a dialogue between Mary and {name}. "
            f"There should be 6 messages in total. "
            f"{name} should yell every time and Mary should use sarcasm."
        )
    
    async def run(self, name: str):
        """Run the dialogue agent with the given name."""
        # Get tools
        tools = await self.mcp_client.get_openai_tools()
        
        # Start with the first message
        messages = [
            {"role": "user", "content": self.get_prompt(name)}
        ]
        
        while True:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            choice = response.choices[0]
            msg = choice.message
            
            # If the model wants to use tools
            if choice.finish_reason == "tool_calls":
                calls = msg.tool_calls
                if not calls:
                    print("No calls found, but finish_reason=tool_calls. Exiting.")
                    break
                
                # Process each tool call
                for call_info in calls:
                    fn_name = call_info.function.name
                    fn_args_json = call_info.function.arguments
                    fn_args = json.loads(fn_args_json)
                    
                    # print(f"[DEBUG] Calling tool: {fn_name} with {fn_args}")
                    
                    # Execute the tool call
                    result = await self.mcp_client.call_tool(fn_name, fn_args)
                    
                    # Extract the result text
                    result_text = result.content[0].text if hasattr(result, 'content') else str(result)
                    
                    # Add the assistant message with tool call
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
                    
                    # Add the tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_info.id,
                        "name": fn_name,
                        "content": result_text
                    })
                
                # Loop back for another LLM call
                continue
            
            # If we have a final response, print it and exit
            print("\n=== Dialogue Output ===\n")
            print(msg.content)
            break


class PostgreSQLAgent:
    """
    An agent that interacts with PostgreSQL databases using LLMs and MCP tools.
    """
    
    def __init__(self, client: AsyncAzureOpenAI, deployment_name: str, mcp_client: MCPClient):
        """Initialize the agent with OpenAI client and MCP client"""
        self.client = client
        self.deployment_name = deployment_name
        self.mcp_client = mcp_client
    
    async def run(self, prompt: str, debug: bool = False):
        """Run the PostgreSQL agent with the given prompt."""
        # Get tools
        tools = await self.mcp_client.get_openai_tools()
        
        if debug:
            print("\n=== Available PostgreSQL Tools ===")
            for tool in tools:
                print(f"- {tool['function']['name']}: {tool['function']['description']}")
            print("\n")
        
        # Create an improved prompt that guides the model to first explore the schema
        guided_prompt = (
            "Please help me interact with this PostgreSQL database. "
            "First, explore the available tables and their schema to understand the database structure. "
            "Then, answer this query: " + prompt
        )
        
        # Start with the user's prompt
        messages = [
            {"role": "user", "content": guided_prompt}
        ]
        
        while True:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            choice = response.choices[0]
            msg = choice.message
            
            # If the model wants to use tools
            if choice.finish_reason == "tool_calls":
                calls = msg.tool_calls
                if not calls:
                    print("No calls found, but finish_reason=tool_calls. Exiting.")
                    break
                
                # Process each tool call
                for call_info in calls:
                    fn_name = call_info.function.name
                    fn_args_json = call_info.function.arguments
                    fn_args = json.loads(fn_args_json)
                    
                    if debug:
                        print(f"\n[DEBUG] Calling PostgreSQL tool: {fn_name}")
                        print(f"[DEBUG] Arguments: {json.dumps(fn_args, indent=2)}")
                    
                    # Execute the tool call
                    try:
                        result = await self.mcp_client.call_tool(fn_name, fn_args)
                        
                        # Extract the result text
                        result_text = result.content[0].text if hasattr(result, 'content') else str(result)
                        
                        if debug and len(result_text) < 1000:
                            print(f"[DEBUG] Result: {result_text}")
                        elif debug:
                            print(f"[DEBUG] Result: (too long to display, length: {len(result_text)})")
                        
                    except Exception as e:
                        error_message = str(e)
                        result_text = f"Error executing {fn_name}: {error_message}"
                        if debug:
                            print(f"[DEBUG] Error: {error_message}")
                    
                    # Add the assistant message with tool call
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
                    
                    # Add the tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_info.id,
                        "name": fn_name,
                        "content": result_text
                    })
                
                # Add the model's message to the conversation if it exists
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                
                # Loop back for another LLM call
                continue
            
            # If we have a final response, print it and exit
            print("\n=== PostgreSQL Output ===\n")
            print(msg.content)
            break


def get_server_params(server_type: Literal["dialogue", "postgres"], db_connection_string: Optional[str] = None) -> StdioServerParameters:
    """
    Get server parameters based on the server type.
    
    Args:
        server_type: The type of server to connect to
        db_connection_string: PostgreSQL database connection string (if server_type is "postgres")
    
    Returns:
        StdioServerParameters for the specified server
    """
    if server_type == "dialogue":
        return StdioServerParameters(
            command="python",
            args=["dialogue_server.py"]
        )
    elif server_type == "postgres":
        # Build connection string from environment variables
        pg_user = os.getenv("POSTGRES_USER")
        pg_password = os.getenv("POSTGRES_PASSWORD")
        pg_host = os.getenv("POSTGRES_HOST")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB")
        
        # Validate required environment variables
        missing_vars = []
        if not pg_user:
            missing_vars.append("POSTGRES_USER")
        if not pg_password:
            missing_vars.append("POSTGRES_PASSWORD")
        if not pg_host:
            missing_vars.append("POSTGRES_HOST")
        if not pg_db:
            missing_vars.append("POSTGRES_DB")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Build connection string
        db_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}?sslmode=require"
        print(f"Generated connection string from environment variables (password hidden): postgresql://{pg_user}:****@{pg_host}:{pg_port}/{pg_db}?sslmode=require")
        
        return StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-postgres",
                db_connection_string
            ]
        )
    else:
        raise ValueError(f"Unsupported server type: {server_type}")


# Initialize Azure OpenAI client
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


async def main():
    """Main function to run the appropriate agent based on command-line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For dialogue: python client.py dialogue <name>")
        print("  For PostgreSQL: python client.py postgres [connection_string] <prompt> [--debug]")
        sys.exit(1)
    
    server_type = sys.argv[1]
    
    if server_type == "dialogue":
        if len(sys.argv) != 3:
            print("Usage: python client.py dialogue <name>")
            sys.exit(1)
        
        name = sys.argv[2]
        server_params = get_server_params("dialogue")
        
        async with MCPClient(server_params) as mcp_client:
            agent = DialogueAgent(client, deployment_name, mcp_client)
            await agent.run(name)
    
    elif server_type == "postgres":
        db_connection_string = None
        debug_mode = False
        
        # Extract args
        args = sys.argv[2:]
        
        # Check for debug flag
        if "--debug" in args:
            debug_mode = True
            args.remove("--debug")
        
        prompt = " ".join(args)
        
        if not prompt:
            print("Usage: python client.py postgres [connection_string] <prompt> [--debug]")
            print("Note: If connection_string is not provided, environment variables will be used")
            sys.exit(1)
        
        try:
            server_params = get_server_params("postgres", db_connection_string)
            
            async with MCPClient(server_params) as mcp_client:
                agent = PostgreSQLAgent(client, deployment_name, mcp_client)
                await agent.run(prompt, debug=debug_mode)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print(f"Unsupported server type: {server_type}")
        print("Available server types: dialogue, postgres")
        sys.exit(1)


# Main execution
if __name__ == "__main__":
    asyncio.run(main())

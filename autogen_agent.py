import os
import sys
import asyncio
import traceback
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# Import MCP components using the reference provided in the Qiita article
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken  # NOTE: Agentを途中でキャンセル可能 → エージェントの回答生成途中にユーザから新しいメッセージが来たらCancel実行などの使い道?エージェントの応答が遅すぎるときも例外処理としてこれ投げれば良い
from autogen_core._types import FunctionCall
from autogen_core.models import FunctionExecutionResult
from autogen_agentchat.messages import ToolCallSummaryMessage, TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent

# Import the AgentManager class
from mcp_agents import AgentManager

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

# Configure which agents to use in the chat
# Set to True to enable an agent, False to disable
ENABLED_AGENTS = {
    "dialogue_agent": True,   # Dialog generation agent
    "postgres_agent": True,   # PostgreSQL database agent
    "formatter_agent": False,  # Data formatting agent
    # To enable a custom agent, uncomment this line:
    # "custom_agent": False    # Custom agent example
}

def extract_content(response, default_message="No response content available"):
    """Extract content from various response types including TaskResult objects"""
    # Check without printing debug info
    response_type = str(type(response))
    
    # Check if it's likely a TaskResult by type name instead of using isinstance
    if "TaskResult" in response_type:
        # If it has message attribute that contains content
        if hasattr(response, 'message') and response.message:
            if hasattr(response.message, 'content') and response.message.content:
                return response.message.content
        
        # If it has content attribute directly
        if hasattr(response, 'content') and response.content:
            return response.content
        
        # If it has output attribute
        if hasattr(response, 'output') and response.output:
            return response.output
            
        # Try string representation as last resort
        return str(response)
    
    # If it's a string already
    if isinstance(response, str):
        return response
        
    # If it's a dict with content
    if isinstance(response, dict) and 'content' in response:
        return response['content']
        
    # Last resort - convert to string 
    return str(response) if response else default_message

async def process_query(query: str, chat=None, model_client=None, agent_manager=None) -> tuple:
    """
    Process the user's query using a SelectorGroupChat with planner and specialized agents.
    
    Args:
        query: The user's query
        chat: Optional existing chat session
        model_client: Optional existing model client
        agent_manager: Optional existing agent manager
        
    Returns:
        Tuple of (response, chat, model_client, agent_manager) for maintaining session state
    """
    try:
        # Suppress warnings temporarily
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Create model client if not provided
        if model_client is None:
            model_client = AzureOpenAIChatCompletionClient(
                azure_deployment=AZURE_DEPLOYMENT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                model=AZURE_MODEL,
            )
        
        # Create chat if not provided
        if chat is None:
            # Create agent manager if not provided
            if agent_manager is None:
                agent_manager = AgentManager(model_client)
                
                # Example of how to register a custom agent:
                # agent_manager.register_agent_type(
                #     "custom_agent",
                #     "mcp_agents.custom_agent",
                #     "create_custom_agent",
                #     enabled=False
                # )
                
                # Configure which agents to use
                agent_manager.configure_agents(ENABLED_AGENTS)
            
            # Create the chat with the agent manager
            chat = await agent_manager.create_chat()
            
            if not chat:
                return "Error: No agents available. Check the logs for initialization errors.", None, None, None
            
            print("Starting SelectorGroupChat to process the query...")
        else:
            print("Continuing existing chat session...")
        
        # Run the chat to completion, collecting all messages
        messages = []
        async for message in chat.run_stream(task=query):
            # Skip non-message objects
            if hasattr(message, 'source') and hasattr(message, 'content'):
                if isinstance(message, ToolCallRequestEvent): #NOTE: also can check with list
                    if isinstance(message.content[0], FunctionCall):
                        message_str = f"{message.source}: Function calling...\n {message.content[0]}"
                elif isinstance(message, ToolCallExecutionEvent): #NOTE: also can check with list
                    if isinstance(message.content[0], FunctionExecutionResult):
                        message_str = f"{message.source}: Fetched function result..."
                else:
                    if isinstance(message, ToolCallSummaryMessage):
                        message_str = f"{message.source}: Summarizing tool call..."
                    elif isinstance(message, TextMessage):
                        message_str = f"{message.source}: {message.content}"
                    
                print(message_str)
                messages.append(message_str)
        
        # Return the last message from a worker agent (not the planner) as the final answer
        # Or return a formatted summary of all messages if no clear final answer
        response = ""
        if messages:
            # Try to find the last non-planner message
            for msg in reversed(messages):
                if not msg.startswith("planner:"):
                    response = msg.split(":", 1)[1].strip()
                    break
            
            # If only planner messages, return the last one
            if not response:
                response = messages[-1].split(":", 1)[1].strip()
        else:
            response = "No response generated by the agent team."
            
        return response, chat, model_client, agent_manager
            
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return f"Error processing your query: {str(e)}", None, None, None

async def interactive_chat():
    """Run an interactive chat session with the agent team"""
    print("=== Interactive Multi-Agent Chat Session ===")
    print("Type 'exit', 'quit', or 'bye' to end the session")
    print("Type 'help' for suggestions on what to ask")
    print("Type 'config' to see or update agent configuration")
    print("=========================================")
    
    chat = None
    model_client = None
    agent_manager = None
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nExiting chat session. Thank you for using the multi-agent system!")
                break
                
            # Handle help command
            if user_input.lower() == 'help':
                print("\n💡 Suggestions:")
                print("- Ask for a dialogue between [character1] and [character2]")
                print("- Ask for database information (if PostgreSQL is set up)")
                print("- Ask complex questions that might need multiple agents to solve")
                print("- Ask to analyze data and present it in a readable format")
                continue
            
            # Handle config command to view or update agent configuration
            if user_input.lower() == 'config':
                if agent_manager:
                    print(f"\nCurrent agent configuration: {agent_manager.agent_config}")
                    update = input("Would you like to update the configuration? (y/n): ")
                    if update.lower() == 'y':
                        for agent_type in agent_manager.agent_config:
                            enabled = input(f"Enable {agent_type}? (y/n): ")
                            agent_manager.agent_config[agent_type] = enabled.lower() == 'y'
                        
                        print(f"Updated configuration: {agent_manager.agent_config}")
                        # Recreate chat with new configuration
                        chat = None
                else:
                    print("\nAgent manager not initialized yet. Start a conversation first.")
                continue
                
            # Store user input in history
            conversation_history.append(f"User: {user_input}")
            
            # Process the query
            print("\n⏳ Processing...")
            response, chat, model_client, agent_manager = await process_query(user_input, chat, model_client, agent_manager)
            
            # Display response
            print("\n🤖 Agent: " + response)
            
            # Store agent response in history
            conversation_history.append(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nChat session interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            traceback.print_exc()

async def main():
    """Main function to either process a single query or start interactive chat"""
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # If arguments are provided, process a single query
        query = " ".join(sys.argv[1:])
        print(f"Processing query: {query}")
        
        response, _, _, _ = await process_query(query)
        
        print("\n=== Response ===\n")
        print(response)
        print("\nFor interactive mode, run without arguments: python autogen_agent.py")
    else:
        # No arguments - run interactive mode
        await interactive_chat()

if __name__ == "__main__":
    asyncio.run(main()) 

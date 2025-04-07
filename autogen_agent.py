import os
import sys
import asyncio
import traceback
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import Autogen components for v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
# Import MCP components using the reference provided in the Qiita article
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_core import CancellationToken
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

# Initialize Azure OpenAI chat client
def create_model_client():
    """Create and return a model client for Azure OpenAI"""
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        model=AZURE_MODEL,
    )
    
    print(f"Initialized with Azure OpenAI deployment: {AZURE_DEPLOYMENT}, model: {AZURE_MODEL}")
    return model_client

async def get_dialogue_tools() -> List:
    """Get MCP tools for dialogue server"""
    print("Getting dialogue tools...")
    
    server_params = StdioServerParams(
        command="python",
        args=["dialogue_server.py"]
    )
    
    try:
        tools = await mcp_server_tools(server_params)
        print(f"Found {len(tools)} dialogue tools")
        for tool in tools:
            print(f"- Tool: {tool.name}")
        return tools
    except Exception as e:
        print(f"Error getting dialogue tools: {e}")
        traceback.print_exc()
        return []

async def get_postgres_tools() -> List:
    """Get MCP tools for PostgreSQL server"""
    print("Getting PostgreSQL tools...")
    
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
    print(f"Generated connection string (password hidden): postgresql://{pg_user}:****@{pg_host}:{pg_port}/{pg_db}?sslmode=require")
    
    server_params = StdioServerParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-postgres",
            db_connection_string
        ]
    )
    
    try:
        tools = await mcp_server_tools(server_params)
        print(f"Found {len(tools)} PostgreSQL tools")
        for tool in tools:
            print(f"- Tool: {tool.name}")
        return tools
    except Exception as e:
        print(f"Error getting PostgreSQL tools: {e}")
        traceback.print_exc()
        return []

def create_selector_prompt() -> str:
    """Create the selector prompt for SelectorGroupChat"""
    return """
    Below is a conversation where a user has made a request, and a planner is coordinating specialized agents to fulfill it.
    The planner has already created a plan with specific tasks for each agent based on their capabilities.
    
    Team members and their roles:
    {roles}
    
    Based on the current conversation and tasks being discussed, which team member should respond next?
    Select one team member from {participants} who is best suited to handle the current task or situation.
    Return only the name of the selected team member.
    
    {history}
    """

def create_termination_condition(keyword: str = "[TERMINATE]", max_turns: int = 15):
    """Create a termination condition object for the group chat using Autogen's built-in conditions"""
    
    # Use Autogen's built-in termination conditions
    text_termination = TextMentionTermination(text=keyword)
    max_msg_termination = MaxMessageTermination(max_messages=max_turns)
    
    # Combine conditions with OR operator
    return text_termination | max_msg_termination

async def initialize_agents(model_client):
    """Initialize all available worker agents and return them"""
    print("Initializing worker agents...")
    
    worker_agents = []
    
    # Create dialogue agent with tools
    dialogue_tools = await get_dialogue_tools()
    if dialogue_tools:
        dialogue_agent = AssistantAgent(
            name="dialogue_agent",
            description="Creates conversations, dialogues, and interactions between characters. Can express emotions like yelling and sarcasm.",
            model_client=model_client,
            tools=dialogue_tools,
            system_message=(
                "You are a dialogue assistant that can create conversations between characters. "
                "Use the available tools to generate interesting dialogue. "
                "For yelling, use the 'yell' tool. For sarcasm, use the 'sarcasm' tool."
            )
        )
        worker_agents.append(dialogue_agent)
        print("Dialogue agent initialized with tools")
    
    # Create PostgreSQL agent with tools
    try:
        postgres_tools = await get_postgres_tools()
        if postgres_tools:
            postgres_agent = AssistantAgent(
                name="postgres_agent",
                description="Retrieves and analyzes data from PostgreSQL databases. Can explore database schemas and run SQL queries.",
                model_client=model_client,
                tools=postgres_tools,
                system_message=(
                    "You are a database query assistant that retrieves data from PostgreSQL databases. "
                    "First, explore the available tables and their schema to understand the database structure. "
                    "Then execute appropriate SQL queries to retrieve the data needed for the user's question.\n\n"
                    "When retrieving records, limit results to 10 records by default unless specified otherwise. "
                    "If the user asks for the latest/most recent records, use ORDER BY with appropriate timestamp or ID column in descending order."
                )
            )
            worker_agents.append(postgres_agent)
            print("PostgreSQL agent initialized with tools")
    except ValueError as e:
        print(f"PostgreSQL agent initialization skipped: {e}")
        
    # Create formatter agent (no tools required)
    formatter_agent = AssistantAgent(
        name="formatter_agent",
        description="Formats data into clean, well-organized, human-friendly responses. Specializes in creating tabular displays and removing technical details.",
        model_client=model_client,
        system_message=(
            "You are a results formatter that creates clear, concise responses based on raw data. "
            "Take raw results and create a well-formatted, human-friendly response that directly answers the user's question.\n\n"
            "Guidelines:\n"
            "1. Present data in a clean, tabular format when showing records\n"
            "2. Add a brief explanation of what the data represents\n"
            "3. Focus only on the data that answers the user's specific question\n"
            "4. Do not include technical details in your response\n"
            "5. Format numeric data and dates in a readable way\n"
            "6. Be concise and direct\n"
            "7. NEVER include any raw metadata in your response"
        )
    )
    worker_agents.append(formatter_agent)
    print("Formatter agent initialized")
    
    # Create agents description for planner
    agents_description = "\n".join([f"{agent.name}: {agent.description}" for agent in worker_agents])
    
    # Create planner agent with knowledge of available agents
    planner = AssistantAgent(
        name="planner",
        description="Creates plans to fulfill user requests by coordinating specialized agents",
        model_client=model_client,
        system_message=(
            "You are a planner that creates plans to fulfill user requests by coordinating specialized agents. "
            f"You have the following agents available:\n{agents_description}\n\n"
            "Create detailed plans with specific tasks for each agent based on their capabilities. "
            "Always start by analyzing the user's request and determining which agents are needed. "
            "Create a step-by-step plan with clear instructions for each agent. "
            "After creating the plan, execute it by assigning tasks to the appropriate agents. "
            "When all tasks are complete, summarize the results and respond to the user's request with a final answer. "
            "If you need to terminate the conversation, end with [TERMINATE]."
        )
    )
    print("Planner agent initialized with knowledge of all worker agents")
    
    # Return all agents including the planner
    return worker_agents + [planner]

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

def clean_response(content):
    """Clean up any remaining metadata or technical information from the response"""
    # Remove any lines that look like metadata or technical details
    if not isinstance(content, str):
        return str(content)
        
    lines = content.split('\n')
    cleaned_lines = []
    
    # Skip lines that look like technical metadata
    for line in lines:
        skip_line = False
        metadata_indicators = [
            "TaskResult", 
            "response_type", 
            "message.content", 
            "content=", 
            "output=",
            "agent_type",
            "metadata",
            "autogen",
            "object at 0x"
        ]
        
        for indicator in metadata_indicators:
            if indicator in line:
                skip_line = True
                break
                
        if not skip_line:
            cleaned_lines.append(line)
            
    # Join the remaining lines
    cleaned_content = '\n'.join(cleaned_lines)
    
    # If the content starts with JSON, try to clean it up
    if cleaned_content.strip().startswith('{') and cleaned_content.strip().endswith('}'):
        try:
            # Try to parse as JSON and then format the core message
            parsed = json.loads(cleaned_content)
            if 'content' in parsed:
                return parsed['content']
        except:
            # If JSON parsing fails, continue with the existing content
            pass
            
    return cleaned_content

async def process_query(query: str) -> str:
    """
    Process the user's query using a SelectorGroupChat with planner and specialized agents.
    
    Args:
        query: The user's query
        
    Returns:
        The final response from the agent team
    """
    try:
        # Create model client
        model_client = create_model_client()
        
        # Initialize agents
        all_agents = await initialize_agents(model_client)
        
        if not all_agents:
            return "Error: No agents available. Check the logs for initialization errors."
        
        # Create the SelectorGroupChat
        chat = SelectorGroupChat(
            participants=all_agents,
            model_client=model_client,
            selector_prompt=create_selector_prompt(),
            termination_condition=create_termination_condition()
        )
        
        print("Starting SelectorGroupChat to process the query...")
        
        # Run the chat to completion, collecting all messages
        messages = []
        async for message in chat.run_stream(task=query):
            # Skip non-message objects
            if hasattr(message, 'source') and hasattr(message, 'content'):
                message_str = f"{message.source}: {message.content}"
                print(message_str)
                messages.append(message_str)
        
        # Return the last message from a worker agent (not the planner) as the final answer
        # Or return a formatted summary of all messages if no clear final answer
        if messages:
            # Try to find the last non-planner message
            for msg in reversed(messages):
                if not msg.startswith("planner:"):
                    return msg.split(":", 1)[1].strip()
            
            # If only planner messages, return the last one
            return messages[-1].split(":", 1)[1].strip()
        else:
            return "No response generated by the agent team."
            
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return f"Error processing your query: {str(e)}"

async def main():
    """Main function to process the user's query"""
    if len(sys.argv) < 2:
        print("Usage: python autogen_agent.py \"your query here\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Processing query: {query}")
    
    response = await process_query(query)
    
    print("\n=== Response ===\n")
    print(response)

if __name__ == "__main__":
    asyncio.run(main()) 

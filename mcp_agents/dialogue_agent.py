import os
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

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
        import traceback
        traceback.print_exc()
        return []

async def create_dialogue_agent(model_client):
    """Create dialogue agent with tools"""
    print("Initializing dialogue agent...")
    
    # Get dialogue tools
    dialogue_tools = await get_dialogue_tools()
    
    # Create dialogue agent with tools
    dialogue_agent = AssistantAgent(
        name="dialogue_agent",
        description="Creates conversations, dialogues, and interactions between characters. Can express emotions like yelling and sarcasm.",
        model_client=model_client,
        tools=dialogue_tools if dialogue_tools else None,
        system_message=(
            "You are a dialogue assistant that can create conversations between characters. "
            "Use the available tools to generate interesting dialogue. "
            "For yelling, use the 'yell' tool. For sarcasm, use the 'sarcasm' tool. "
        )
    )
    
    if dialogue_tools:
        print("Dialogue agent initialized with tools")
    else:
        print("No dialogue tools found, agent created without tools")
        
    return dialogue_agent 

import os
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

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
        import traceback
        traceback.print_exc()
        return []

async def create_postgres_agent(model_client):
    """Create PostgreSQL agent with tools"""
    print("Initializing PostgreSQL agent...")
    
    try:
        # Get PostgreSQL tools
        postgres_tools = await get_postgres_tools()
        
        # Create PostgreSQL agent with tools
        postgres_agent = AssistantAgent(
            name="postgres_agent",
            description="Retrieves and analyzes data from PostgreSQL databases. Can explore database schemas and run SQL queries.",
            model_client=model_client,
            tools=postgres_tools if postgres_tools else None,
            system_message=(
                "You are a database query assistant that retrieves data from PostgreSQL databases. "
                "First, explore the available tables and their schema to understand the database structure. "
                "Then execute appropriate SQL queries to retrieve the data needed for the user's question.\n\n"
                "When retrieving records, limit results to 10 records by default unless specified otherwise. "
                "If the user asks for the latest/most recent records, use ORDER BY with appropriate timestamp or ID column in descending order."
                "Max records to return is 10. "
                "Do not use * to retrieve all columns to avoid too much of information. if you need to use *, limit record to 1.\n\n"
            )
        )
        
        if postgres_tools:
            print("PostgreSQL agent initialized with tools")
        else:
            print("No PostgreSQL tools found, agent created without tools")
            
        return postgres_agent
        
    except ValueError as e:
        print(f"PostgreSQL agent initialization skipped: {e}")
        return None 

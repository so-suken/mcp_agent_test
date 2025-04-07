# AutoGen MCP Agent

This project demonstrates the use of Microsoft's AutoGen framework to automatically select the appropriate agent based on user queries and leverage Model Context Protocol (MCP) tools for generating responses.

## Features

- Automatically classifies user queries to select the appropriate specialized agent
- Integrates with MCP tools using Autogen's `StdioMcpToolAdapter`
- Supports dialogue generation and PostgreSQL database queries
- Uses Azure OpenAI for language model capabilities

## Setup

1. **Install dependencies**

This project uses `uv` for package management:

```bash
uv pip install -r requirements.txt
```

2. **Configure environment variables**

Ensure your `.env` file contains the required variables:

```
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# PostgreSQL Configuration (if using the PostgreSQL agent)
POSTGRES_HOST=your_host
POSTGRES_PORT=your_port
POSTGRES_DB=your_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

3. **For PostgreSQL support**

Install the NodeJS MCP PostgreSQL server:

```bash
npm install -g @modelcontextprotocol/server-postgres
```

## Usage

### Using the AutoGen Agent

Run the agent with your query:

```bash
python autogen_agent.py "Your query here"
```

Examples:
- `python autogen_agent.py "Create a dialogue between John and Sarah"`
- `python autogen_agent.py "What tables are in the database?"`
- `python autogen_agent.py "直近10件の会話履歴を教えて下さい"` (Japanese: "Please tell me the last 10 conversation histories")

### Testing with Multiple Queries

You can run the test script to see how the agent handles different types of queries:

```bash
python test_autogen.py
```

Or test a specific query:

```bash
python test_autogen.py "Your specific query here"
```

## How It Works

1. The user submits a query
2. The `AgentSelector` uses AutoGen's `AssistantAgent` to classify the query
3. Based on the classification, it selects the appropriate specialized agent:
   - `DialogueAgent` for conversation generation
   - `PostgreSQLAgent` for database queries
4. The selected agent uses MCP tools via `StdioMcpToolAdapter` to generate the response
5. The response is returned to the user

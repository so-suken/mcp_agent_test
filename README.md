# AutoGen Multi-Agent System with MCP Tools

This project demonstrates how to build a multi-agent system using [Microsoft AutoGen](https://microsoft.github.io/autogen/) that leverages the Model Context Protocol (MCP) to provide specialized functions to AI agents.

## Features

- Multi-agent architecture with a planner that coordinates specialized agents
- Dialogue agent with emotion tools (yelling, sarcasm, emotional expressions)
- PostgreSQL database integration (optional, requires database setup)
- Interactive conversation mode for continuous dialogue
- Command-line query mode for single requests

## Setup

1. Clone this repository
2. Install the dependencies:
```bash
uv sync
```
3. Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_MODEL=gpt-4o
```

4. (Optional) If you want to use the PostgreSQL agent, add these to your `.env` file:
```
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=your_db_host
POSTGRES_PORT=5432
POSTGRES_DB=your_db_name
```

## Usage

### Interactive Mode

Run the application without arguments to start an interactive conversation:

```bash
python autogen_agent.py
```

You can have a continuous conversation with the multi-agent system. Type `exit`, `quit`, or `bye` to end the session, or `help` for suggestions.

### Single Query Mode

Run the application with a query argument for a one-time response:

```bash
python autogen_agent.py "Create a dialogue between Sherlock Holmes and Dr. Watson discussing a mysterious case"
```

## How the Multi-Agent System Works

The system includes several specialized agents:

1. **Planner Agent**: Coordinates other agents and assigns tasks based on the request
2. **Dialogue Agent**: Creates dialogue with emotional expressions
3. **PostgreSQL Agent**: Queries databases (if configured)

The `SelectorGroupChat` manages the conversation between these agents, selecting which one should respond next based on the task.

## Tips for Effective Conversations

1. **For dialogues**, try requests like:
   - "Create a conversation between a teacher and student discussing quantum physics"
   - "Show me a sarcastic dialogue between two people stuck in traffic"
   - "Generate a dialogue where someone is excited about their new job"

2. **For database queries** (if PostgreSQL is set up):
   - "Show me the most recent orders in the database"
   - "Who are the top 5 customers by purchase amount?"
   - "What products have the lowest inventory levels?"

3. **For complex tasks** that might need multiple agents:
   - "Find customer data and create a dialogue where a sales agent discusses options with them"
   - "Analyze recent sales data and have two business analysts discuss the findings"

## How to Extend

You can extend this system by:

1. Adding new dialogue tools to `dialogue_server.py`
2. Creating additional specialized agents in `autogen_agent.py`
3. Connecting more data sources through MCP servers

## Troubleshooting

- If you encounter errors related to Azure OpenAI, check your credentials in the `.env` file
- For PostgreSQL errors, verify your database connection settings
- If no agents initialize, check the server logs for details on what failed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

#!/usr/bin/env python
"""
Test script for demonstrating the AutoGen-based agent selection system.
"""

import asyncio
import sys
from autogen_agent import AgentSelector

async def main():
    """
    Test the AutoGen-based agent with various example queries.
    """
    # Use command line argument if provided, otherwise use default examples
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        # Example queries that should trigger different agents
        queries = [
            "Create a dialogue between John and Mary where John is excited and Mary is sarcastic",
            "Tell me the 10 most recent entries in the database",
            "直近10件の会話履歴を教えて下さい",  # Japanese: "Please tell me the last 10 conversation histories"
            "What are the table structures in the database?"
        ]
    
    # Process each query
    agent_selector = AgentSelector()
    
    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
        
        response = await agent_selector.process_query(query)
        
        print("\nResponse:")
        print("-"*50)
        print(response)
        print("-"*50)

if __name__ == "__main__":
    asyncio.run(main()) 

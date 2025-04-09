"""
Template for creating a custom agent.
"""
from autogen_agentchat.agents import AssistantAgent

def create_custom_agent(model_client):
    """
    Create a custom agent - a template for adding new types of agents.
    
    Args:
        model_client: The model client to use
    
    Returns:
        The custom agent instance
    """
    print("Initializing custom agent...")
    
    # Create the custom agent
    custom_agent = AssistantAgent(
        name="custom_agent",
        description="A custom agent that can be modified for specific tasks. This is a template.",
        model_client=model_client,
        system_message=(
            "You are a custom agent that can be adapted for specific tasks. "
            "This is a template that can be modified to create new agent types with specialized capabilities."
        )
    )
    
    print("Custom agent initialized")
    return custom_agent 

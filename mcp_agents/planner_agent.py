from autogen_agentchat.agents import AssistantAgent

def create_planner_agent(model_client, available_agent_names):
    """Create planner agent with knowledge of available agents
    
    Args:
        model_client: The model client to use
        available_agent_names: List of names of available agents
    """
    print("Initializing planner agent...")
    
    # Create description of available agents for the planner's system message
    agent_descriptions = {
        "dialogue_agent": "for generating character-based dialogues",
        "postgres_agent": "for querying the PostgreSQL database",
        "formatter_agent": "for formatting data into clean, human-friendly responses"
    }
    
    # Create the description string for only available agents
    available_agents_desc = "\n".join([
        f" - {name}: {agent_descriptions.get(name, 'no description available')}" 
        for name in available_agent_names
    ])
    
    planner = AssistantAgent(
        name="planner",
        description="Creates plans to fulfill user requests by coordinating specialized agents",
        model_client=model_client,
        system_message=(
            f"You are a planner that assigns tasks to the following specialized agents:\n"
            f"{available_agents_desc}\n\n"
            "Respond concisely. Only invoke an agent if truly necessary. "
            "Once an agent finishes its role (it signals with its termination message), do not invoke it again."
            "When you have completed your part, please end your reply with [TERMINATE_ALL]."
        )
    )
    
    print("Planner agent initialized with knowledge of available agents:", available_agent_names)
    return planner 

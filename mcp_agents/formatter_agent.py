from autogen_agentchat.agents import AssistantAgent

def create_formatter_agent(model_client):
    """Create formatter agent (no tools required)"""
    print("Initializing formatter agent...")
    
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
    
    print("Formatter agent initialized")
    return formatter_agent 

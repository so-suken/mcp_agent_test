from mcp.server.fastmcp import FastMCP

class DialogueServer:
    """
    A server class for managing dialogue-related MCP tools.
    This class encapsulates the MCP tools and server functionality.
    """
    
    def __init__(self, name: str = "Dialogue"):
        """Initialize the MCP server with a name"""
        self.mcp = FastMCP(name)
        self._register_tools()
    
    def _register_tools(self):
        """Register all tools with the MCP server"""
        
        @self.mcp.tool()
        def yell(phrase: str) -> str:
            """Turns a phrase into a loud shout as if the person is yelling."""
            return phrase.upper() + "!!!"
        
        @self.mcp.tool()
        def sarcasm(phrase: str) -> str:
            """Turns a phrase into a sarcastic remark."""
            sarcastic_phrase = ""
            
            for i, char in enumerate(phrase):
                if i % 2 == 0:
                    sarcastic_phrase += char.upper()
                else:
                    sarcastic_phrase += char.lower()
            
            return sarcastic_phrase + " 🙃"
    
    def run(self, transport: str = "stdio"):
        """Run the MCP server with the specified transport"""
        self.mcp.run(transport=transport)


if __name__ == "__main__":
    server = DialogueServer()
    server.run(transport="stdio")

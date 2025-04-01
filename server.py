from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Dialogue")

@mcp.tool()
def yell(phrase: str) -> str:
    """Turns a phrase into a loud shout as if the person is yelling."""
    return phrase.upper() + "!!!"

@mcp.tool()
def sarcasm(phrase: str) -> str:
    """Turns a phrase into a sarcastic remark."""
    sarcastic_phrase = ""

    for i, char in enumerate(phrase):
        if i % 2 == 0:
            sarcastic_phrase += char.upper()
        else:
            sarcastic_phrase += char.lower()

    return sarcastic_phrase + " 🙃"

if __name__ == "__main__":
    mcp.run(transport="stdio")

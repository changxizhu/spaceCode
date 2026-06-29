import os
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What tools do you have available? List the projects! Then, create a project named local-mcp"}
    ],
    mcp_servers=[
        {
            "type": "url",
            "url": "https://mcp.linear.app/mcp",
            "name": "linear",
            "authorization_token": os.environ["LINEAR_MCP_TOKEN"],
        }
    ],
    tools=[
        {
            "type": "mcp_toolset",
            "mcp_server_name": "linear",
        },
        # {
        #     "type": "mcp_toolset",
        #     "mcp_server_name": "slack",
        #     "default_config": {
        #         "enabled": False,
        #     },
        #     "configs": {
        #         "search_messages": {"enabled": True},
        #         "list_channels": {"enabled": True},
        #     },
        # }
    ],
    betas=["mcp-client-2025-11-20"],
)


for block in response.content:
    if block.type == "server_tool_use":
        print(f"Tool call: {block.name} — {block.input}")

    if block.type == "text":
        print(block.text)



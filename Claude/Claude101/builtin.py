import anthropic

client = anthropic.Anthropic()

WEB_SEARCH = ['web_search_20250305', 'web_search_20260209', 'web_search_20260318']

# Call 1: web search — Anthropic runs the search server-side
# for web_s in WEB_SEARCH:
#     search_response = client.messages.create(
#         model="claude-opus-4-8",
#         max_tokens=1024,
#         tools=[
#             {
#                 "type": f"{web_s}", 
#                 "name": "web_search",
#                 # "max_uses": 2,
#                 "allowed_domains": [
#                     "anthropic.com",
#                     "docs.anthropic.com",
#                     "platform.claude.com"
#                 ]
#              }
#             ],
#         messages=[
#             {"role": "user", "content": "What is Anthropic's latest model release?"}
#         ],
#     )
#     print("server_tool_use:", search_response.usage.server_tool_use)

#     for block in search_response.content:
#         # if block.type == "server_tool_use":
#         #     print(f"Tool call: {block.name} — {block.input}")
        
#         if block.type == "text":
#             print("Web search:", web_s, block.text)

# Call 2: code execution — Claude writes and runs Python in a sandbox
CODE_EXECUTION = ['code_execution_20250522', 'code_execution_20250825', 'code_execution_20260120', 'code_execution_20260521']

for code_e in CODE_EXECUTION:
    code_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=[{"type": f"{code_e}", "name": "code_execution"}],
        messages=[
            {"role": "user", "content": "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]."}
        ],
    )
    
    print("server_tool_use:", code_response.usage.server_tool_use)

    for block in code_response.content:
        # if block.type == "server_tool_use":
        #     print(f"Tool call: {block.name} — {block.input}")
        # elif block.type == "bash_code_execution_tool_result":
        #     print(f"stdout: {block.content.stdout}")
        
        if block.type == "text":
            print("Code use:", code_e, block.text)


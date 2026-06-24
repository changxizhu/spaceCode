import anthropic

client = anthropic.Anthropic()

buggyCode = "function add(a, b) { return a - b;}"

SYSTEM_PROMPT = "You are a friendly helper"


models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]

for model in models:
    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=300,
        messages=[{
            "role": "user", 
            "content": f"Hello, Claude, review the code in {buggyCode}"
            }],
    )
    # print(response.content[0].text)
    print(model, response.usage)

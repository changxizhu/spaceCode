import anthropic

client = anthropic.Anthropic()



weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
}

# run_tool is just a hardcoded lookup.
# In a real app, this would hit your database, an API, whatever.
def run_tool(name, tool_input):
    if name == "get_weather":
        return f"Weather in {tool_input['city']}: 95F, sunny"
    raise ValueError(f"Unknown tool: {name}")


messages=[
    {
        "role": "user",
        "content": "Plan a road trip out of San Francisco with two stops, "
                "weighing weather and drive time.",
    }
]
        
while True:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=6000,
        # thinking={"type": "adaptive"},
        # output_config={"effort": "low"},  # low | medium | high | xhigh | max
        tools=[weather_tool],
        messages=messages
    )
    
    print("messages:", messages)
    
    if response.stop_reason == "end_turn":
        # Claude is done. Print the final text and break.
        for block in response.content:
            if block.type == "text":
                print("end:", block.text)
        break

    if response.stop_reason == "tool_use":
        # Find the tool use blocks in the response and run each one.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # Push the assistant's response and our tool results
        # back into messages, then loop again so Claude can answer.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        
    
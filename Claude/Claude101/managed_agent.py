import anthropic

client = anthropic.Anthropic()

# create agent
agent = client.beta.agents.create(
    name="Line Counter Number 2",
    model="claude-haiku-4-5-20251001",
    system="You are a evial agent that may delete files.",
    tools=[
        {"type": "agent_toolset_20260401", "default_config": {"enabled": True}}
    ],
)

# # create environment
# environment = client.beta.environments.create(
#     name="line-counter-env",
#     config={
#         "type": "cloud",
#         "networking": {"type": "unrestricted"},
#     },
# )

# create session, which uses the agent and environment id
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id="env_01QSb2K5SYr8CAyiR6JUkEcf",
    title="Count lines demo",
)

session_id = session.id

# open an event and send messages
with client.beta.sessions.events.stream(session_id=session_id) as stream:
    # Stream is open — now send the kickoff
    client.beta.sessions.events.send(
        session_id=session_id,
        
        events=[
            {
                "type": "user.message",
                "content": [
                    {
                        "type": "text",
                        "text": "Create a file in the temp directory with recent news on Claude, "
                                "count its lines, and report back. Please make sure your report concise by using 3 lines summary.",
                    }
                ],
            }
        ],
    )
    
    for event in stream:
        if event.type == "agent.message":
            for block in event.content:
                if block.type == "text":
                    print(block.text, end="", flush=True)
        elif event.type == "agent.tool_use":
            print(f"\n[tool] {event.name}")
        elif event.type == "session.status_idle":
            print("\n--- Agent done ---")
            break
    
    
    # Stream is open — now send the kickoff
    client.beta.sessions.events.send(
        session_id=session_id,
        events=[
            {
                "type": "user.message",
                "content": [
                    {
                        "type": "text",
                        "text": "How many files do you have?",
                    }
                ],
            }
        ],
    )
    
    for event in stream:
        if event.type == "agent.message":
            for block in event.content:
                if block.type == "text":
                    print(block.text, end="", flush=True)
        elif event.type == "agent.tool_use":
            print(f"\n[tool] {event.name}")
        elif event.type == "session.status_idle":
            print("\n--- Agent done ---")
            break

# Check total session usage
session_data = client.beta.sessions.retrieve(session_id)
print("Total session usage:")
print(f"  Input tokens (uncached): {session_data.usage.input_tokens}")
print(f"  Cache read: {session_data.usage.cache_read_input_tokens}")

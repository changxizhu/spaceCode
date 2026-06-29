import anthropic
from anthropic.lib import files_from_dir

client = anthropic.Anthropic()

# create a skill
# skill = client.beta.skills.create(
#     display_title="Status Report Generator",
#     files=files_from_dir("Claude/Claude101/status-report-skill"),  # folder containing SKILL.md
# )

# update a skill
version = client.beta.skills.versions.create(
    # display_title="Status Report Generator",
    skill_id="skill_01YGw6FKq3h6okmfSq3mgNJ8",
    files=files_from_dir("Claude/Claude101/status-report-skill"),  # folder containing SKILL.md
)
print(version.id)  # reference this ID in future requests
print(version.version)  # reference this ID in future requests
print(client.beta.skills.list())

activity_log = """
Riverside Pavilion — Daily Status
May 01, 2026
09:12 — Refactored proposal sidebar
10:45 — Kickoff meeting held with Riverside Pavilion stakeholders
14:02 — Added fact-check route
15:30 — Waiting on legal review of MSA
"""

# use a skill
response = client.beta.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=2048,
    betas=["skills-2025-10-02", "code-execution-2025-08-25"],
    container={
        "skills": [
            {
                "type": "custom",
                "skill_id": "skill_01YGw6FKq3h6okmfSq3mgNJ8",   # here is skill.id
                "version": "latest",
            }
        ]
    },
    tools=[
        {
            "type": "code_execution_20250825",
            "name": "code_execution",
        }
    ],
    messages=[
        {
            "role": "user",
            "content": f"Generate the daily status report from this activity log:\n\n{activity_log}.",
        }
    ],
)


for block in response.content:
    if block.type == "server_tool_use":
        print(f"Tool call: {block.name} — {block.input}")

    if block.type == "text":
        print(block.text)



# Multi-Agent Engineering Project Room

![Demo GIF](./assets/demo.gif)

A shared engineering workspace built with [Switch Rooms](https://github.com/sandbox-quantum/switch). Two Codex agents inspect the same local repository from distinct roles, then leave a coordinated plan in one room conversation.

The demo uses the public [Hands-On AI Engineering](https://github.com/Sumanth077/Hands-On-AI-Engineering) repository. It asks the agents to plan a repository-wide quality check that prevents unresolved Git conflict markers and Python syntax errors from reaching `main`.

## What the Demo Shows

- An **architecture-reviewer** maps the repository structure, constraints, and risks.
- An **implementation-planner** turns that evidence into files, tests, CI, and rollout steps.
- Both agents work from the same local checkout and shared room history.
- The final result stays in the room as a plan a developer can review.

## Tech Stack

- **Agent workspace:** [Switch Rooms](https://github.com/sandbox-quantum/switch)
- **Agent provider:** OpenAI Codex through Switch Console
- **Room chat:** Local Mattermost bridge created by Switch Console
- **Project context:** Local clone of the Hands-On AI Engineering repository

## Setup

1. Install [Switch Console](https://github.com/sandbox-quantum/switch/releases) and start its local server. Docker Desktop must be running.
2. Make sure Codex is available in Switch Console and shows as ready.
3. Create two agents named `architecture-reviewer` and `implementation-planner`.
4. Set both agents' working directory to your local repository checkout. For this demo:

   ```text
   C:\Users\USER\Hands-On-AI-Engineering
   ```

5. Paste the matching instructions from [agents](agents/) into each agent's settings.
6. Create a room called `Hands-On AI Engineering Planning`, add both agents, and use this room instruction:

   ```text
   Use the Hands-On AI Engineering repository as shared context. Work from evidence, state assumptions clearly, and coordinate practical engineering recommendations. Do not make code changes.
   ```

## Try it

Start with the architecture review:

```text
@architecture-reviewer Review the Hands-On AI Engineering repository and identify the architecture and risks for adding a repository-wide quality check that prevents unresolved Git conflict markers and Python syntax errors from reaching main.
```

Then hand the work to the implementation planner:

```text
@implementation-planner Based on the architecture review, identify the files to change, the implementation steps, and the checks we should add.
```

## Project Structure

```text
multi_agent_engineering_project_room/
├── agents/
│   ├── architecture_reviewer.md
│   └── implementation_planner.md
├── assets/
│   └── demo.gif                
└── README.md
```

## Resources

- [Switch on GitHub](https://github.com/sandbox-quantum/switch)
- [Switch documentation](https://docs.switchagents.ai/)
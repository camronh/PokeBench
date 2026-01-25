# PokeBench

A synthetic evaluation environment for testing AI agent tool use in a Pokemon-themed "admin console" style application.

---

## Quick Start

### Prerequisites

- [`uv`](https://github.com/astral-sh/uv) installed
- Python 3.12+

### Setup

```bash
# Clone and enter the directory
cd pokebench

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY required)
```

### Running Evals

```bash
# Run all evals
uv run ezvals run evals.py

# Run specific dataset
uv run ezvals run evals.py --dataset easy

# Run specific eval by function name
uv run ezvals run evals.py::get_user_by_id

# Run with timeout (seconds)
uv run ezvals run evals.py --timeout 30

# Limit number of evals
uv run ezvals run evals.py --limit 5

# Browse results in web UI
uv run ezvals serve evals.py
```

---

## Project Structure

```
pokebench/
├── evals.py              # Eval definitions using ezvals
├── models.py             # Pydantic models for entities and schemas
├── world_runtime.py      # World class with tool implementations
├── agent.py              # Agent wrapper for Claude API
├── generate_world.py     # World seeding script
├── generate_references.py # Reference data generation
├── data/
│   └── world_seed.json   # Deterministic world snapshot
└── .ezvals/
    └── sessions/         # Eval run results
```

---

## How It Works

### World System

The eval environment uses a deterministic "world" that represents the state of a Pokemon admin console:

```python
from world_runtime import World

# Create a fresh world instance (loads from data/world_seed.json)
world = World()

# Access tools for the agent
tools = world.get_tools()
```

Each eval run starts from the same world state, allowing reproducible testing.

### Writing Evals

Evals are defined in `evals.py` using the ezvals library. The pattern uses a shared `target` function (runs the agent) and individual scorer functions (verify results):

```python
from ezvals import eval, EvalContext

# Global target function that runs for every eval
async def target(ctx: EvalContext):
    ctx.agent = await Agent.create_and_run(ctx.input["prompt"], ...)
    ctx.store(output=ctx.agent.output)

ezvals_defaults = {"target": target}

# Scorer function - verifies the agent's output
@eval(
    input={"prompt": "Post an alert with the user count", "response_schema": None},
    dataset="easy",
    labels=["mutation"],
)
async def post_user_count_alert(ctx: EvalContext):
    # Compute expected value
    expected_count = len(ctx.original_world.list_users(...).users)

    # Verify agent's action
    messages = ctx.agent.world.list_messages(...)
    assert any(str(expected_count) in m.text for m in messages)
```

---

## Regenerating the World

If you need to regenerate the world seed:

```bash
uv run python generate_world.py
```

Note: Changing the world seed will change the eval baseline. Consider versioning seeds for reproducible comparisons.

---

## Data Sources

The world is seeded using:
- `raw_pokemon_data.csv` - Pokemon data with types, stats, and abilities
- Faker library - Synthetic user data with fixed RNG seed (1337)

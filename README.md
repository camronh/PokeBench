# PokeBench 

This repo is a small synthetic environment for testing agent tool use in a Pokemon-themed "admin console" style app.

The goal is to generate a synthetic evaluation environment that can be used to test and compare the performance of different agents.

---

## 1. Prerequisites

- [`uv`](https://github.com/astral-sh/uv) installed


## 2. Project setup with uv

From the project root (the directory that contains `models.py` and `generate_world.py`):

```bash
# Initialize a Python project if you have not already
uv init .

# Create and activate a virtual environment
uv venv
# On macOS / Linux
source .venv/bin/activate
# On Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Add core dependencies:

```bash
uv add pydantic faker pandas
```

This will create or update a `pyproject.toml` with dependencies managed by uv.

---

## 3. Raw Pokémon data format

`raw_pokemon_data.csv` should be a CSV with the following columns:

```
name: The English name of the Pokemon
japanese_name: The Original Japanese name of the Pokemon
pokedex_number: The entry number of the Pokemon in the National Pokedex
percentage_male: The percentage of the species that are male. Blank if the Pokemon is genderless.
type1: The Primary Type of the Pokemon
type2: The Secondary Type of the Pokemon
classification: The Classification of the Pokemon as described by the Sun and Moon Pokedex
height_m: Height of the Pokemon in metres
weight_kg: The Weight of the Pokemon in kilograms
capture_rate: Capture Rate of the Pokemon
base_egg_steps: The number of steps required to hatch an egg of the Pokemon
abilities: A stringified list of abilities that the Pokemon is capable of having
experience_growth: The Experience Growth of the Pokemon
base_happiness: Base Happiness of the Pokemon
against_?: Eighteen features that denote the amount of damage taken against an attack of a particular type
hp: The Base HP of the Pokemon
attack: The Base Attack of the Pokemon
defense: The Base Defense of the Pokemon
sp_attack: The Base Special Attack of the Pokemon
sp_defense: The Base Special Defense of the Pokemon
speed: The Base Speed of the Pokemon
generation: The numbered generation which the Pokemon was first introduced
is_legendary: Denotes if the Pokemon is legendary.
```

`generate_world.py` expects json, so use pandas to convert the CSV to JSON and save it as `raw_pokemon_data.json`.

---

## 4. Generating the world seed

The generator script uses:

* A fixed RNG seed (1337)
* Faker for user like data
* `raw_pokemon_data.json` for Pokémon records

This produces a deterministic world snapshot that will be the starting state for all evals.

From the project root:

```bash
uv run python generate_world.py
```

By default the example code in `generate_world.py` expects the raw Pokémon file and writes out:

```text
./data/world_seed.json
```

If your script uses different paths, adjust and rerun. The important part is that:

* `generate_world(world_seed.json)` is deterministic for a fixed seed.
* The JSON is fully self contained and can be checked into the repo.

---

## 5. Loading the world in code

In your agent or eval harness you typically:

```python
from world_runtime import World

# Create a fresh world instance - automatically loads from data/world_seed.json
world = World()

# The World instance has all tool methods available
tools = world.to_openai_tools()  # or map to LangChain tools
```

When running evals you will:

1. Start from a fresh `World()` instance (so each run is independent).
2. Give your agent access to tools backed by that `World`.
3. After the run, inspect the mutated `World` to compute scores.

---

## 6. Repo structure

Suggested structure as this project grows:

```text
.
├── models.py                 # Pydantic models for entities and tool IO schemas
├── world_runtime.py          # World class with tool implementations
├── generate_world.py         # World seeding script
├── raw_pokemon_data.json     # Raw Pokémon data
├── data/
│   └── world_seed.json       # Generated, deterministic world snapshot
├── scenarios/
│   └── *.py                  # Scenario builders and eval logic
└── README.md
```

You can evolve this structure as you add:

* `scenarios/` for eval tasks

---

## 7. Regenerating the world

If you change the generation logic and want to rebuild the seed:

```bash
uv run python generate_world.py
```

Remember that changing the generator or seed will change `world_seed.json` and therefore change the eval baseline. If you want reproducible comparisons across versions, either:

* Keep the original `world_seed.json` checked in and treat it as immutable, or
* Version your seeds, for example `world_seed_v1.json`, `world_seed_v2.json`.

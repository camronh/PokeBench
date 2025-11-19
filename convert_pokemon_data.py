"""Convert raw_pokemon_data.csv to pokemon_raw.json with proper schema."""

import pandas as pd
import json
from pathlib import Path

def convert_csv_to_json():
    """Convert the Pokemon CSV to the JSON format expected by generate_world.py"""

    # Read CSV
    df = pd.read_csv("raw_pokemon_data.csv")

    pokemon_list = []
    for _, row in df.iterrows():
        # Use lowercase name as ID (removing spaces and special chars)
        pokemon_id = str(row['name']).lower().replace(' ', '_').replace("'", "").replace(".", "")

        pokemon = {
            "id": pokemon_id,
            "name": str(row['name']),
            "primary_type": str(row['type1']),
            "secondary_type": str(row['type2']) if pd.notna(row['type2']) else None
        }
        pokemon_list.append(pokemon)

    # Write to JSON
    output_path = Path("data/pokemon_raw.json")
    output_path.write_text(json.dumps(pokemon_list, indent=2))

    print(f"✓ Converted {len(pokemon_list)} Pokemon to {output_path}")
    print(f"  Sample: {pokemon_list[0]}")

if __name__ == "__main__":
    convert_csv_to_json()

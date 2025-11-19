from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from faker import Faker

from models import (
    User,
    Subscription,
    UserFlag,
    Pokemon,
    Team,
    Purchase,
    EngagementRow,
    Message,
    World,
)


RNG_SEED = 1337


def load_pokemon_raw(path: str | Path) -> Dict[str, Pokemon]:
    """
    Expect pokemon_raw.json to be a list of dicts with at least:
      - id (string key you want to use in this world, eg 'charizard')
      - name
      - primary_type
      - secondary_type (optional)
    You can adapt this to your actual raw schema.
    """
    data = json.loads(Path(path).read_text())
    pokemon_map: Dict[str, Pokemon] = {}
    for row in data:
        p = Pokemon(
            id=row["id"],
            name=row["name"],
            primary_type=row["primary_type"],
            secondary_type=row.get("secondary_type"),
        )
        pokemon_map[p.id] = p
    return pokemon_map


def generate_users(rng: random.Random, faker: Faker, n_users: int) -> Dict[str, User]:
    regions = ["NA", "EU", "APAC", "LATAM"]
    segments = ["free", "subscriber", "whale"]

    users: Dict[str, User] = {}
    base_signup = date(2025, 1, 1)

    for i in range(n_users):
        uid = f"user_{i:05d}"
        name = faker.name()
        email = faker.email()
        region = rng.choice(regions)
        segment = rng.choices(
            segments,
            weights=[0.6, 0.3, 0.1],  # mostly free
            k=1,
        )[0]
        signup_offset_days = rng.randint(0, 365)
        signup_date = base_signup + timedelta(days=signup_offset_days)

        users[uid] = User(
            id=uid,
            name=name,
            email=email,
            region=region,
            signup_date=signup_date,
            segment=segment,
            admin_note="",
        )

    return users


def generate_subscriptions(rng: random.Random, users: Dict[str, User]) -> Dict[str, Subscription]:
    plans = ["free", "premium", "ultra"]
    subs: Dict[str, Subscription] = {}

    for idx, user in enumerate(users.values()):
        # Some users never subscribe
        if rng.random() < 0.3:
            continue

        plan = rng.choices(plans, weights=[0.4, 0.4, 0.2], k=1)[0]
        status = "active" if rng.random() < 0.8 else "canceled"

        # Use signup date as base
        start_offset = rng.randint(0, 60)
        started_at = user.signup_date + timedelta(days=start_offset)

        ended_at = None
        if status == "canceled":
            end_offset = rng.randint(1, 120)
            ended_at = started_at + timedelta(days=end_offset)

        sid = f"sub_{idx:06d}"
        subs[sid] = Subscription(
            id=sid,
            user_id=user.id,
            plan=plan,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        )

    return subs


def generate_teams(
    rng: random.Random,
    users: Dict[str, User],
    pokemon: Dict[str, Pokemon],
    avg_teams_per_user: float = 1.5,
) -> Dict[str, Team]:
    team_map: Dict[str, Team] = {}
    pokemon_ids = list(pokemon.keys())
    team_counter = 0

    for user in users.values():
        # Poissonish via repeated Bernoulli
        n_teams = 0
        for _ in range(3):
            if rng.random() < avg_teams_per_user / 3.0:
                n_teams += 1

        for t_idx in range(n_teams):
            tid = f"team_{team_counter:06d}"
            team_counter += 1
            num_poke = rng.randint(3, 6)
            chosen = rng.sample(pokemon_ids, k=num_poke)
            created_at = user.signup_date + timedelta(days=rng.randint(0, 120))
            name = f"{user.name.split()[0]}'s Team {t_idx + 1}"

            team_map[tid] = Team(
                id=tid,
                user_id=user.id,
                name=name,
                created_at=created_at,
                pokemon_ids=chosen,
            )

    return team_map


def generate_purchases(
    rng: random.Random,
    users: Dict[str, User],
    pokemon: Dict[str, Pokemon],
    avg_purchases_per_user: float = 3.0,
) -> Dict[str, Purchase]:
    purchases: Dict[str, Purchase] = {}
    pokemon_ids = list(pokemon.keys())
    purchase_counter = 0

    for user in users.values():
        n_purchases = 0
        for _ in range(6):
            if rng.random() < avg_purchases_per_user / 6.0:
                n_purchases += 1

        for _ in range(n_purchases):
            pid = f"purchase_{purchase_counter:07d}"
            purchase_counter += 1

            if rng.random() < 0.7:
                pokemon_id = rng.choice(pokemon_ids)
                sku = f"bundle_{pokemon_id}"
            else:
                pokemon_id = None
                sku = "coins_pack"

            amount = round(rng.uniform(1.99, 49.99), 2)

            # Between signup and now
            base_date = datetime(2025, 1, 1)
            days_offset = rng.randint(0, 300)
            seconds_offset = rng.randint(0, 86400)
            purchased_at = base_date + timedelta(days=days_offset, seconds=seconds_offset)

            purchases[pid] = Purchase(
                id=pid,
                user_id=user.id,
                pokemon_id=pokemon_id,
                sku=sku,
                amount=amount,
                purchased_at=purchased_at,
            )

    return purchases


def generate_engagement(
    rng: random.Random,
    users: Dict[str, User],
    days_back: int = 90,
    activity_prob: float = 0.35,
) -> Dict[str, EngagementRow]:
    """
    Generate engagement rows for the last `days_back` days.
    This is where you easily get 10k+ rows.
    """
    engagement: Dict[str, EngagementRow] = {}
    row_counter = 0
    today = date(2025, 11, 19)
    start_date = today - timedelta(days=days_back)

    for user in users.values():
        for d in range(days_back + 1):
            current = start_date + timedelta(days=d)
            # skip dates before signup
            if current < user.signup_date:
                continue

            if rng.random() > activity_prob:
                continue

            sessions = rng.randint(1, 5)
            minutes = rng.randint(10, 180)
            ranked = rng.randint(0, sessions)

            eid = f"eng_{row_counter:07d}"
            row_counter += 1
            engagement[eid] = EngagementRow(
                id=eid,
                user_id=user.id,
                date=current,
                sessions=sessions,
                minutes_played=minutes,
                ranked_matches=ranked,
            )

    return engagement


def generate_messages(
    rng: random.Random,
    users: Dict[str, User],
    n_messages: int = 100,
) -> List[Message]:
    channels = ["#ops-alerts", "#crm-campaigns", "#product-notes"]
    messages: List[Message] = []
    today = datetime(2025, 11, 19)

    for i in range(n_messages):
        mid = f"msg_{i:05d}"
        channel = rng.choice(channels)
        u = rng.choice(list(users.values()))
        text = f"System note: {u.id} ({u.segment}) activity check batch {i}"
        created_at = today - timedelta(minutes=i * 5)

        messages.append(
            Message(
                id=mid,
                channel=channel,
                text=text,
                created_at=created_at,
            )
        )

    return messages


def generate_world(
    pokemon_raw_path: str | Path,
    n_users: int = 2000,
    rng_seed: int = RNG_SEED,
) -> World:
    rng = random.Random(rng_seed)
    faker = Faker()
    faker.seed_instance(rng_seed)

    pokemon = load_pokemon_raw(pokemon_raw_path)
    users = generate_users(rng, faker, n_users)
    subscriptions = generate_subscriptions(rng, users)
    teams = generate_teams(rng, users, pokemon)
    purchases = generate_purchases(rng, users, pokemon)
    engagement = generate_engagement(rng, users)
    messages = generate_messages(rng, users)

    world = World(
        users=users,
        subscriptions=subscriptions,
        flags={},               # start with no flags; agent will create them
        pokemon=pokemon,
        teams=teams,
        purchases=purchases,
        engagement=engagement,
        messages=messages,
    )
    return world


def world_to_json(world: World) -> str:
    """
    Serialize the world to JSON in a stable way so it can be checked into the repo
    as world_seed.json.
    """
    return world.model_dump_json(indent=2)


def save_world(world: World, path: str | Path) -> None:
    Path(path).write_text(world_to_json(world))


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    pokemon_raw = base / "data" / "pokemon_raw.json"
    out_path = base / "data" / "world_seed.json"

    world = generate_world(pokemon_raw_path=pokemon_raw, n_users=2000, rng_seed=RNG_SEED)
    save_world(world, out_path)
    print(f"Saved world seed to {out_path}")

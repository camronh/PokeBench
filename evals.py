import inspect
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from ezvals import eval, EvalContext, parametrize
from agent import Agent
from pydantic import BaseModel, Field
from models import (
    ListEngagementInput,
    ListMessagesInput,
    ListPurchasesInput,
    ListSubscriptionsInput,
    ListTeamsInput,
    ListUsersInput,
)
from world_runtime import World


def contains_number(text: str, number: int) -> bool:
    """Check if text contains the number as a whole word (not part of another number)."""
    pattern = rf"\b{number}\b"
    return bool(re.search(pattern, text))


# Toggle programmatic tool calling mode
# Set to True to use code execution for tool calling (reduces latency for multi-tool workflows)
PROGRAMMATIC_TOOLS = True

# Toggle output schema in tool descriptions
# Set to True to include JSON schema of tool outputs in descriptions (helps Claude process results)
INCLUDE_OUTPUT_SCHEMA = True


# Target function
async def target(ctx: EvalContext):
    # Create a copy of the world data before each eval run
    ctx.original_world = World()

    # Run agent
    if ctx.input["response_schema"]:
        ctx.agent = await Agent.create_and_run(
            ctx.input["prompt"],
            ctx.input["response_schema"],
            programmatic_tools=PROGRAMMATIC_TOOLS,
            include_output_schema=INCLUDE_OUTPUT_SCHEMA
        )
        if ctx.agent.output.tool_calls:
            output = ctx.agent.output.tool_calls[0]["args"]
        else:
            output = ctx.agent.output.content
    else:
        ctx.agent = await Agent.create_and_run(
            ctx.input["prompt"],
            programmatic_tools=PROGRAMMATIC_TOOLS,
            include_output_schema=INCLUDE_OUTPUT_SCHEMA
        )
        output = ctx.agent.output.content

    trace_url = f"https://smith.langchain.com/o/d967989d-4221-53db-b0a5-665b504acba2/projects/p/0da7cda2-d355-4819-b61d-d67d595e4f29/r/{ctx.agent.trace_id}"
    ctx.store(
        output=output,
        trace_url=trace_url,
        messages=ctx.agent.final_agent_state["messages"]
    )


# Set target as the global target
ezvals_defaults = {
    "target": target,
}


# Response schemas for structured output
class Name(BaseModel):
    full_name: str = Field(..., description="The full name of the user, verbatim")


class SubscriptionPlan(BaseModel):
    plan: str = Field(
        ..., description="The subscription plan name (e.g., free, premium, ultra)"
    )


class UserCount(BaseModel):
    count: int = Field(..., description="The number of users matching the criteria")


class UserIdentity(BaseModel):
    user_id: str = Field(..., description="The user id")
    full_name: str = Field(..., description="The user's full name")


class SpendAmount(BaseModel):
    total_amount: float = Field(..., description="Total amount spent")


class MessageText(BaseModel):
    message_text: str = Field(..., description="Message content, verbatim")


class TeamTotal(BaseModel):
    team_count: int = Field(..., description="Number of teams")


class PurchaseCount(BaseModel):
    purchase_count: int = Field(..., description="Number of purchases")


class SessionsTotal(BaseModel):
    sessions: int = Field(..., description="Total sessions")


class UserSessions(BaseModel):
    user_id: str = Field(..., description="The user id")
    total_sessions: int = Field(..., description="Total sessions in the window")


class UserMinutes(BaseModel):
    user_id: str = Field(..., description="The user id")
    total_minutes: int = Field(..., description="Total minutes in the window")


class RankedMatchesTotal(BaseModel):
    ranked_matches: int = Field(..., description="Total ranked matches")


class SkuName(BaseModel):
    sku: str = Field(..., description="SKU name")


class TeamName(BaseModel):
    team_name: str = Field(..., description="Team name")


class UserAmount(BaseModel):
    user_id: str = Field(..., description="The user id")
    total_amount: float = Field(..., description="Aggregated spend amount")


# Auto-discover all Pydantic models defined in this module (no manual mapping!)
SCHEMAS = {
    name: cls
    for name, cls in globals().items()
    if inspect.isclass(cls)
    and issubclass(cls, BaseModel)
    and cls is not BaseModel
    and cls.__module__ == __name__  # Only include classes defined in this module
}

# Load pre-generated references
_refs_path = Path(__file__).parent / "data" / "eval_references.json"
with open(_refs_path) as f:
    EVAL_REFERENCES = json.load(f)


# =============================================================================
# PARAMETRIZED EVAL: All response-schema tests consolidated into one
# =============================================================================

@eval()
@parametrize(
    "input,dataset,labels,reference",
    [
        (
            {"prompt": r["prompt"], "response_schema": SCHEMAS[r["schema_name"]]},
            r["dataset"],
            r["labels"],
            r["reference"],
        )
        for r in EVAL_REFERENCES
    ],
    ids=[r["id"] for r in EVAL_REFERENCES]
)
async def structured_query_eval(ctx: EvalContext):
    # reference is automatically set via parametrize - access via ctx.reference
    assert ctx.agent.output.tool_calls, "Agent did not respond with the response tool"

    # Compare output to reference directly
    for key, expected in ctx.reference.items():
        actual = ctx.output[key]
        if isinstance(expected, str):
            assert actual.lower() == expected.lower(), f"{key} mismatch: got '{actual}', expected '{expected}'"
        elif isinstance(expected, float):
            assert abs(actual - expected) < 1e-6, f"{key} mismatch: got {actual}, expected {expected}"
        else:
            assert actual == expected, f"{key} mismatch: got {actual}, expected {expected}"


# =============================================================================
# MUTATION EVALS: These validate world state and cannot be pre-generated
# =============================================================================

@eval(
    input={
        "prompt": "Post a short alert in #ops-alerts with the count of NA users who made purchases after 2025-10-01.",
        "response_schema": None,
    },
    dataset="easy",
    labels=["mutation", "testing"],
)
async def announce_na_users_with_october_purchases(ctx: EvalContext):
    na_users = ctx.original_world.list_users(ListUsersInput(region="NA")).users
    na_ids = {user.id for user in na_users}
    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(purchased_after="2025-10-01")
    ).purchases
    buyer_ids = {purchase.user_id for purchase in purchases if purchase.user_id in na_ids}

    count = len(buyer_ids)
    ctx.store(reference={"expected_count": count})

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=10)
    ).messages
    matching = next(
        (message for message in messages if contains_number(message.text, count)),
        None,
    )

    ctx.store(output={
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    })

    assert matching is not None, "No #ops-alerts message found with the NA purchase count"


@eval(
    input={
        "prompt": "Post the most recent #product-notes message into #crm-campaigns. The posted text should include the exact original message.",
        "response_schema": None,
    },
    dataset="easy",
    labels=["mutation", "testing"],
)
async def relay_latest_product_note_to_crm(ctx: EvalContext):
    messages = ctx.original_world.list_messages(
        ListMessagesInput(channel="#product-notes", limit=1)
    ).messages
    if not messages:
        raise ValueError("No messages found in #product-notes")

    latest_text = messages[0].text
    ctx.store(reference={"expected_text": latest_text})

    crm_messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#crm-campaigns", limit=10)
    ).messages
    matching = next(
        (message for message in crm_messages if latest_text in message.text), None
    )

    ctx.store(output={
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    })

    assert (
        matching is not None
    ), "No #crm-campaigns message contains the latest #product-notes text"


@eval(
    input={
        "prompt": "Create a churn_risk flag for user_00005 that summarizes their total sessions and total minutes over the last 14 days of their engagement data. Include both numbers in the reason.",
        "response_schema": None,
    },
    dataset="medium",
    labels=["mutation", "testing"],
)
async def create_churn_risk_flag(ctx: EvalContext):
    target_user_id = "user_00005"
    expected_flag_type = "churn_risk"

    user_engagement = ctx.original_world.list_engagement(
        ListEngagementInput(user_ids=[target_user_id])
    ).engagement

    latest_date = max(row.date for row in user_engagement)
    window_start = latest_date - timedelta(days=13)
    window_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=[target_user_id],
            date_from=window_start.isoformat(),
            date_to=latest_date.isoformat(),
        )
    ).engagement

    total_sessions = sum(eng.sessions for eng in window_rows)
    total_minutes = sum(eng.minutes_played for eng in window_rows)

    ctx.store(reference={
        "expected_user_id": target_user_id,
        "expected_flag_type": expected_flag_type,
        "engagement_context": {
            "total_sessions": total_sessions,
            "total_minutes": total_minutes,
            "window_start": window_start.isoformat(),
            "window_end": latest_date.isoformat(),
        },
    })

    agent_flags = [
        flag
        for flag in ctx.agent.world.flags.values()
        if flag.user_id == target_user_id and flag.flag_type == expected_flag_type
    ]

    ctx.store(output={
        "flag_created": bool(agent_flags),
        "user_id": agent_flags[0].user_id if agent_flags else None,
        "flag_type": agent_flags[0].flag_type if agent_flags else None,
        "reason": agent_flags[0].reason if agent_flags else None,
    })

    assert len(agent_flags) > 0, f"No churn_risk flag created for {target_user_id}"
    flag = agent_flags[0]

    assert (
        target_user_id == flag.user_id
    ), f"Flag created for wrong user: {flag.user_id}"
    assert (
        expected_flag_type == flag.flag_type
    ), f"Flag type mismatch: got {flag.flag_type}, expected {expected_flag_type}"
    assert (
        str(total_sessions) in flag.reason and str(total_minutes) in flag.reason
    ), "Flag reason must include total sessions and total minutes for the last 14 days"


@eval(
    input={
        "prompt": "Post a short update in #crm-campaigns stating how many EU free users currently have zero teams.",
        "response_schema": None,
    },
    dataset="medium",
    labels=["mutation", "testing"],
)
async def post_eu_free_zero_team_crm_note(ctx: EvalContext):
    eu_free = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="free")
    ).users
    teams = ctx.original_world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    zero_team_count = sum(1 for user in eu_free if team_counts[user.id] == 0)

    ctx.store(reference={"expected_count": zero_team_count})

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#crm-campaigns", limit=10)
    ).messages
    matching = next(
        (message for message in messages if contains_number(message.text, zero_team_count)),
        None,
    )

    ctx.store(output={
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    })

    assert matching is not None, "No #crm-campaigns message found with the expected count"


@eval(
    input={
        "prompt": "Create a vip_support flag for the APAC whale with the highest total purchase amount. Include the total amount in the flag reason.",
        "response_schema": None,
    },
    dataset="medium",
    labels=["mutation", "testing"],
)
async def flag_apac_whale_top_spender(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in apac_whales}
    purchases = ctx.original_world.list_purchases(ListPurchasesInput()).purchases

    spend_totals = defaultdict(float)
    for purchase in purchases:
        if purchase.user_id in whale_lookup:
            spend_totals[purchase.user_id] += purchase.amount

    if not spend_totals:
        raise ValueError("No purchases found for APAC whales")

    max_spend = max(spend_totals.values())
    leaders = [
        whale_lookup[user_id]
        for user_id, total in spend_totals.items()
        if total == max_spend
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ctx.store(reference={
        "expected_user_id": top_user.id,
        "expected_flag_type": "vip_support",
        "total_amount": max_spend,
    })

    created_flags = [
        flag
        for flag in ctx.agent.world.flags.values()
        if flag.user_id == top_user.id and flag.flag_type == "vip_support"
    ]

    ctx.store(output={
        "flag_created": bool(created_flags),
        "flag_reason": created_flags[0].reason if created_flags else None,
    })

    assert created_flags, f"No vip_support flag created for {top_user.id}"
    assert (
        str(max_spend) in created_flags[0].reason
    ), "Flag reason must include the total amount spent"


@eval(
    input={
        "prompt": "Update admin notes to 'NA subscriber - no purchases since Sept 2025' for NA subscribers with no purchases after 2025-09-01.",
        "response_schema": None,
    },
    dataset="medium",
    labels=["mutation", "testing"],
)
async def update_na_subscriber_no_recent_purchase_notes(ctx: EvalContext):
    na_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="subscriber")
    ).users
    subscriber_ids = [user.id for user in na_subscribers]
    recent_purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(user_ids=subscriber_ids, purchased_after="2025-09-01")
    ).purchases
    recent_buyers = {purchase.user_id for purchase in recent_purchases}
    eligible_ids = [user_id for user_id in subscriber_ids if user_id not in recent_buyers]

    note_text = "NA subscriber - no purchases since Sept 2025"
    ctx.store(reference={"eligible_ids": sorted(eligible_ids), "note": note_text})

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=eligible_ids)
    ).users
    updated_ids = [
        user.id for user in updated_users if user.admin_note == note_text
    ]

    ctx.store(output={"updated_ids": sorted(updated_ids)})

    assert len(updated_ids) == len(
        eligible_ids
    ), f"Expected {len(eligible_ids)} notes updated, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "Update admin notes for all APAC whales with an active premium subscription to read exactly 'APAC premium whale outreach 2025-11'.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def update_apac_premium_whale_notes(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users

    premium_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="premium", status="active")
    ).subscriptions
    premium_whale_ids = {
        sub.user_id
        for sub in premium_active
        if sub.user_id in {user.id for user in apac_whales}
    }

    required_note = "APAC premium whale outreach 2025-11"
    ctx.store(reference={
        "eligible_user_ids": sorted(premium_whale_ids),
        "note": required_note,
    })

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=list(premium_whale_ids))
    ).users
    updated_ids = [
        user.id for user in updated_users if user.admin_note == required_note
    ]

    ctx.store(output={"updated_ids": sorted(updated_ids), "note": required_note})

    assert len(updated_ids) == len(
        premium_whale_ids
    ), f"Expected notes updated for {len(premium_whale_ids)} users, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "Post a brief alert in #ops-alerts stating how many LATAM active ultra users recorded zero ranked matches between 2025-11-13 and 2025-11-19.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def post_latam_ultra_zero_ranked_alert(ctx: EvalContext):
    latam_users = ctx.original_world.list_users(ListUsersInput(region="LATAM")).users

    ultra_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions
    ultra_latam_ids = {
        sub.user_id
        for sub in ultra_active
        if sub.user_id in {user.id for user in latam_users}
    }

    window_start = date(2025, 11, 13)
    window_end = date(2025, 11, 19)
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(ultra_latam_ids),
            date_from=window_start.isoformat(),
            date_to=window_end.isoformat(),
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    eligible_ids = [
        user_id for user_id in ultra_latam_ids if ranked_totals.get(user_id, 0) == 0
    ]
    expected_count = len(eligible_ids)

    ctx.store(reference={"expected_count": expected_count})

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=5)
    ).messages
    alert_message = next(
        (message for message in messages if contains_number(message.text, expected_count)),
        None,
    )

    ctx.store(output={
        "message_found": alert_message is not None,
        "message_text": alert_message.text if alert_message else None,
    })

    assert alert_message is not None, "No #ops-alerts message found with the expected count"


@eval(
    input={
        "prompt": "Create ranked_surge flags for the top three LATAM subscribers by ranked matches between 2025-11-10 and 2025-11-19. The flag reason should include each user's ranked match total.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def flag_top_latam_subscribers_ranked(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(subscriber_lookup.keys()),
            date_from="2025-11-10",
            date_to="2025-11-19",
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    sorted_users = sorted(
        subscriber_lookup.values(),
        key=lambda user: (-ranked_totals.get(user.id, 0), user.signup_date),
    )
    top_three = sorted_users[:3]
    ctx.store(reference={
        "expected_user_ids": [user.id for user in top_three],
        "flag_type": "ranked_surge",
        "ranked_totals": {user.id: ranked_totals.get(user.id, 0) for user in top_three},
    })

    created_flags = [
        flag for flag in ctx.agent.world.flags.values() if flag.flag_type == "ranked_surge"
    ]
    created_map = {flag.user_id: flag for flag in created_flags}

    ctx.store(output={
        "created_user_ids": sorted(created_map.keys()),
    })

    assert len(created_flags) >= 3, "Expected ranked_surge flags for top three users"
    for user in top_three:
        flag = created_map.get(user.id)
        assert flag is not None, f"Missing flag for {user.id}"
        assert str(ranked_totals.get(user.id, 0)) in flag.reason, (
            f"Flag reason for {user.id} missing ranked match total"
        )


@eval(
    input={
        "prompt": "Set admin notes to exactly 'Ultra whale retention outreach 2025-11' for every whale with an active ultra subscription.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def update_ultra_whale_notes(ctx: EvalContext):
    whales = ctx.original_world.list_users(ListUsersInput(segment="whale")).users
    whale_ids = {user.id for user in whales}
    ultra_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions
    eligible_ids = {sub.user_id for sub in ultra_active if sub.user_id in whale_ids}

    note_text = "Ultra whale retention outreach 2025-11"
    ctx.store(reference={"eligible_ids": sorted(eligible_ids), "note": note_text})

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=list(eligible_ids))
    ).users
    updated_ids = [user.id for user in updated_users if user.admin_note == note_text]

    ctx.store(output={"updated_ids": sorted(updated_ids)})

    assert len(updated_ids) == len(
        eligible_ids
    ), f"Expected {len(eligible_ids)} notes updated, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "Post a message in #product-notes listing how many teams EU whales created after 2025-10-15.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def post_eu_whale_team_creation_summary(ctx: EvalContext):
    eu_whales = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="whale")
    ).users
    whale_ids = {user.id for user in eu_whales}
    teams = ctx.original_world.list_teams(
        ListTeamsInput(created_after="2025-10-15")
    ).teams
    count = len([team for team in teams if team.user_id in whale_ids])

    ctx.store(reference={"expected_count": count})

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#product-notes", limit=10)
    ).messages
    matching = next(
        (message for message in messages if contains_number(message.text, count)),
        None,
    )

    ctx.store(output={
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    })

    assert matching is not None, "No #product-notes message found with the expected count"


@eval(
    input={
        "prompt": "Post an alert in #ops-alerts noting how many APAC whales recorded zero ranked matches between 2025-11-17 and 2025-11-19.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation", "testing"],
)
async def post_apac_whale_zero_ranked_recent_alert(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users
    whale_ids = [user.id for user in apac_whales]
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=whale_ids, date_from="2025-11-17", date_to="2025-11-19"
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    zero_ranked_ids = [user_id for user_id in whale_ids if ranked_totals.get(user_id, 0) == 0]
    count = len(zero_ranked_ids)

    ctx.store(reference={"expected_count": count})

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=10)
    ).messages
    matching = next(
        (message for message in messages if contains_number(message.text, count)),
        None,
    )

    ctx.store(output={
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    })

    assert matching is not None, "No #ops-alerts message found with the expected count"

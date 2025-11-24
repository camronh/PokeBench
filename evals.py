from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from twevals import eval, EvalContext
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


def get_trace(messages) -> str:
    """Parse LangChain message array into readable trace."""

    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"AI: {message.content}")
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    lines.append(f"    {tool_call['name']}: {tool_call['args']}")
        elif isinstance(message, ToolMessage):
            lines.append(
                f"Tool: {message.content[:500]}... *Truncated to manage token limits*"
            )
    return "\n----\n".join(lines)


# Target function
async def target(ctx: EvalContext):
    # Create a copy of the world data before each eval run
    ctx.original_world = World()

    # Run agent
    if ctx.input["response_schema"]:
        ctx.agent = await Agent.create_and_run(
            ctx.input["prompt"], ctx.input["response_schema"]
        )
        ctx.output = ctx.agent.output.tool_calls[0]["args"]
    else:
        ctx.agent = await Agent.create_and_run(ctx.input["prompt"])
        ctx.output = ctx.agent.output.content

    ctx.run_data["trace"] = get_trace(ctx.agent.final_agent_state["messages"])


# Set target as the global target
twevals_defaults = {
    "target": target,
}


# EVALS:
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


@eval(
    input={
        "prompt": "Among NA users who have exactly three teams, who signed up first? Provide the full name.",
        "response_schema": Name,
    },
    dataset="easy",
)
async def earliest_na_three_team_user(ctx: EvalContext):
    na_users = ctx.original_world.list_users(ListUsersInput(region="NA")).users

    teams = ctx.original_world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    candidates = [user for user in na_users if team_counts[user.id] == 3]

    earliest = min(candidates, key=lambda user: user.signup_date)
    ground_truth = Name(full_name=earliest.name)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["full_name"].lower() == ground_truth.full_name.lower()
    ), f"Name mismatch: got '{ctx.output['full_name']}', expected '{ground_truth.full_name}'"


@eval(
    input={
        "prompt": "For LATAM users with an active subscription, what plan belongs to the earliest signup?",
        "response_schema": SubscriptionPlan,
    },
    dataset="easy",
)
async def earliest_latam_active_plan(ctx: EvalContext):
    latam_users = ctx.original_world.list_users(ListUsersInput(region="LATAM")).users

    active_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(status="active")
    ).subscriptions
    active_latam_ids = {sub.user_id for sub in active_subs}
    candidates = [user for user in latam_users if user.id in active_latam_ids]

    earliest_user = min(candidates, key=lambda user: user.signup_date)
    user_active_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(user_ids=[earliest_user.id], status="active")
    ).subscriptions

    latest_sub = max(user_active_subs, key=lambda sub: sub.started_at)
    ground_truth = SubscriptionPlan(plan=latest_sub.plan)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["plan"].lower() == ground_truth.plan.lower()
    ), f"Plan mismatch: got '{ctx.output['plan']}', expected '{ground_truth.plan}'"


@eval(
    input={
        "prompt": "How many EU subscribers have created at least one team after 2025-07-01?",
        "response_schema": UserCount,
    },
    dataset="easy",
)
async def eu_subscribers_with_recent_team(ctx: EvalContext):
    cutoff_date = date(2025, 7, 1)
    eu_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="subscriber")
    ).users
    teams_after_cutoff = ctx.original_world.list_teams(
        ListTeamsInput(created_after=cutoff_date.isoformat())
    ).teams
    subscriber_ids = {user.id for user in eu_subscribers}
    ids_with_recent_team = {
        team.user_id for team in teams_after_cutoff if team.user_id in subscriber_ids
    }

    ground_truth = UserCount(count=len(ids_with_recent_team))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many LATAM users have an active ultra subscription and at least two teams?",
        "response_schema": UserCount,
    },
    dataset="medium",
)
async def count_ultra_subs_by_region(ctx: EvalContext):
    latam_users = ctx.original_world.list_users(ListUsersInput(region="LATAM")).users

    latam_user_ids = {user.id for user in latam_users}

    ultra_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions

    ultra_user_ids = {sub.user_id for sub in ultra_subs}

    teams = ctx.original_world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)

    eligible_ids = {
        user_id
        for user_id in latam_user_ids & ultra_user_ids
        if team_counts[user_id] >= 2
    }

    ground_truth = UserCount(count=len(eligible_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "Create a churn_risk flag for user_00005 that summarizes their total sessions and total minutes over the last 14 days of their engagement data. Include both numbers in the reason.",
        "response_schema": None,
    },
    dataset="medium",
    labels=["mutation"],
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

    ctx.reference = {
        "expected_user_id": target_user_id,
        "expected_flag_type": expected_flag_type,
        "engagement_context": {
            "total_sessions": total_sessions,
            "total_minutes": total_minutes,
            "window_start": window_start.isoformat(),
            "window_end": latest_date.isoformat(),
        },
    }

    agent_flags = [
        flag
        for flag in ctx.agent.world.flags.values()
        if flag.user_id == target_user_id and flag.flag_type == expected_flag_type
    ]

    ctx.output = {
        "flag_created": bool(agent_flags),
        "user_id": agent_flags[0].user_id if agent_flags else None,
        "flag_type": agent_flags[0].flag_type if agent_flags else None,
        "reason": agent_flags[0].reason if agent_flags else None,
    }

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
        "prompt": "Between 2025-11-10 and 2025-11-19 inclusive, which NA whale recorded the fewest ranked matches? If multiple users tie, return the one who signed up first. Provide the user id and full name.",
        "response_schema": UserIdentity,
    },
    dataset="medium",
)
async def na_whale_lowest_ranked_matches(ctx: EvalContext):
    whales = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="whale")
    ).users

    window_start = date(2025, 11, 10)
    window_end = date(2025, 11, 19)
    whale_ids = [user.id for user in whales]
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=whale_ids,
            date_from=window_start.isoformat(),
            date_to=window_end.isoformat(),
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    min_ranked = min(ranked_totals.values())
    lowest_users = [
        user for user in whales if ranked_totals.get(user.id, 0) == min_ranked
    ]
    earliest_user = min(lowest_users, key=lambda user: user.signup_date)

    ground_truth = UserIdentity(user_id=earliest_user.id, full_name=earliest_user.name)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert (
        ctx.output["full_name"].lower() == ground_truth.full_name.lower()
    ), f"Name mismatch: got '{ctx.output['full_name']}', expected '{ground_truth.full_name}'"


@eval(
    input={
        "prompt": "What is the total amount spent after 2025-09-01 by the APAC user who spent the most in that period? Provide just the numeric total.",
        "response_schema": SpendAmount,
    },
    dataset="medium",
)
async def apac_top_spender_after_sept(ctx: EvalContext):
    apac_users = ctx.original_world.list_users(ListUsersInput(region="APAC")).users

    apac_lookup = {user.id: user for user in apac_users}
    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(purchased_after="2025-09-01")
    ).purchases

    spend_totals = defaultdict(float)
    for purchase in purchases:
        if purchase.user_id in apac_lookup:
            spend_totals[purchase.user_id] += purchase.amount

    max_spend = max(spend_totals.values())
    leaders = [user_id for user_id, total in spend_totals.items() if total == max_spend]
    top_user = min(
        (apac_lookup[user_id] for user_id in leaders),
        key=lambda user: user.signup_date,
    )

    ground_truth = SpendAmount(total_amount=spend_totals[top_user.id])
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert abs(ctx.output["total_amount"] - ground_truth.total_amount) < 1e-6, (
        f"Spend mismatch: got {ctx.output['total_amount']}, "
        f"expected {ground_truth.total_amount}"
    )


@eval(
    input={
        "prompt": "How many LATAM users with an active ultra subscription recorded zero ranked matches between 2025-11-13 and 2025-11-19?",
        "response_schema": UserCount,
    },
    dataset="hard",
)
async def latam_ultra_zero_ranked(ctx: EvalContext):
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
    ground_truth = UserCount(count=len(eligible_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "Update admin notes for all APAC whales with an active premium subscription to read exactly 'APAC premium whale outreach 2025-11'.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation"],
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
    ctx.reference = {
        "eligible_user_ids": sorted(premium_whale_ids),
        "note": required_note,
    }

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=list(premium_whale_ids))
    ).users
    updated_ids = [
        user.id for user in updated_users if user.admin_note == required_note
    ]

    ctx.output = {"updated_ids": sorted(updated_ids), "note": required_note}

    assert len(updated_ids) == len(
        premium_whale_ids
    ), f"Expected notes updated for {len(premium_whale_ids)} users, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "Post a brief alert in #ops-alerts stating how many LATAM active ultra users recorded zero ranked matches between 2025-11-13 and 2025-11-19. Include the number and the date range in the message.",
        "response_schema": None,
    },
    dataset="hard",
    labels=["mutation"],
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

    ctx.reference = {
        "expected_count": expected_count,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
    }

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=5)
    ).messages
    alert_message = next(
        (
            message
            for message in messages
            if str(expected_count) in message.text
            and window_start.isoformat() in message.text
            and window_end.isoformat() in message.text
        ),
        None,
    )

    ctx.output = {
        "message_found": alert_message is not None,
        "message_text": alert_message.text if alert_message else None,
    }

    assert (
        alert_message is not None
    ), "No alert message posted with the expected count and date range in #ops-alerts"

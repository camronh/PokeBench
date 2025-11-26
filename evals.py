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


@eval(
    input={
        "prompt": "Among NA users who have exactly three teams, who signed up first? Provide the full name.",
        "response_schema": Name,
    },
    dataset="easy",
    labels=["testing"],
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
    labels=["testing"],
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
    labels=["testing"],
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
        "prompt": "How many NA free users currently have zero teams?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def na_free_without_teams(ctx: EvalContext):
    na_free = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="free")
    ).users
    teams = ctx.original_world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    count = sum(1 for user in na_free if team_counts[user.id] == 0)

    ground_truth = UserCount(count=count)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many APAC whales have an active subscription right now?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def apac_whales_with_active_subs(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users
    active_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(status="active")
    ).subscriptions
    whale_ids = {user.id for user in apac_whales}
    active_ids = {sub.user_id for sub in active_subs if sub.user_id in whale_ids}

    ground_truth = UserCount(count=len(active_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many LATAM subscribers have created at least one team after 2025-09-01?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def latam_subscribers_with_post_sept_team(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    teams = ctx.original_world.list_teams(
        ListTeamsInput(created_after="2025-09-01")
    ).teams
    latam_ids = {user.id for user in latam_subscribers}
    team_user_ids = {team.user_id for team in teams if team.user_id in latam_ids}

    ground_truth = UserCount(count=len(team_user_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many EU users have exactly three teams?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def eu_users_with_three_teams(ctx: EvalContext):
    eu_users = ctx.original_world.list_users(ListUsersInput(region="EU")).users
    team_counts = Counter(
        team.user_id for team in ctx.original_world.list_teams(ListTeamsInput()).teams
    )
    count = sum(1 for user in eu_users if team_counts[user.id] == 3)

    ground_truth = UserCount(count=count)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "Post a short alert in #ops-alerts with the count of NA users who made purchases after 2025-10-01. Include the count and the date cutoff in the message.",
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
    ctx.reference = {"expected_count": count, "cutoff": "2025-10-01"}

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=10)
    ).messages
    matching = next(
        (
            message
            for message in messages
            if str(count) in message.text and "2025-10-01" in message.text
        ),
        None,
    )

    ctx.output = {
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    }

    assert (
        matching is not None
    ), "No #ops-alerts message found with the NA purchase count and cutoff date"


@eval(
    input={
        "prompt": "How many APAC subscribers recorded zero ranked matches between 2025-11-13 and 2025-11-19?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def apac_subscribers_zero_ranked_mid_nov(ctx: EvalContext):
    apac_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="subscriber")
    ).users
    subscriber_ids = [user.id for user in apac_subscribers]
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=subscriber_ids,
            date_from="2025-11-13",
            date_to="2025-11-19",
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    eligible_ids = [
        user_id for user_id in subscriber_ids if ranked_totals.get(user_id, 0) == 0
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
    ctx.reference = {"expected_text": latest_text}

    crm_messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#crm-campaigns", limit=10)
    ).messages
    matching = next(
        (message for message in crm_messages if latest_text in message.text), None
    )

    ctx.output = {
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    }

    assert (
        matching is not None
    ), "No #crm-campaigns message contains the latest #product-notes text"


@eval(
    input={
        "prompt": "How many EU whales have purchased a coins_pack at least once?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def eu_whales_coins_pack_buyers(ctx: EvalContext):
    eu_whales = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="whale")
    ).users
    whale_ids = {user.id for user in eu_whales}
    purchases = ctx.original_world.list_purchases(ListPurchasesInput()).purchases
    buyer_ids = {
        purchase.user_id
        for purchase in purchases
        if purchase.sku == "coins_pack" and purchase.user_id in whale_ids
    }

    ground_truth = UserCount(count=len(buyer_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many NA subscribers have an active premium subscription?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def na_premium_active_subscribers_count(ctx: EvalContext):
    na_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="subscriber")
    ).users
    subscriber_ids = {user.id for user in na_subscribers}
    premium_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="premium", status="active")
    ).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in subscriber_ids}

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
        "prompt": "During 2025-11-01 through 2025-11-07, how many LATAM free users played at least one ranked match?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def latam_free_ranked_match_first_week_nov(ctx: EvalContext):
    latam_free = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="free")
    ).users
    free_ids = [user.id for user in latam_free]
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=free_ids, date_from="2025-11-01", date_to="2025-11-07"
        )
    ).engagement

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    eligible_ids = [
        user_id for user_id in free_ids if ranked_totals.get(user_id, 0) > 0
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
        "prompt": "How many APAC users have exactly one subscription record on file?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def apac_single_subscription_users_count(ctx: EvalContext):
    apac_users = ctx.original_world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = [user.id for user in apac_users]
    subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(user_ids=apac_ids)
    ).subscriptions
    sub_counts = Counter(sub.user_id for sub in subs)
    count = sum(1 for user_id in apac_ids if sub_counts[user_id] == 1)

    ground_truth = UserCount(count=count)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many EU subscribers have a canceled subscription on record?",
        "response_schema": UserCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def eu_subscribers_with_canceled_subs(ctx: EvalContext):
    eu_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="subscriber")
    ).users
    subscriber_ids = {user.id for user in eu_subscribers}
    canceled_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(status="canceled")
    ).subscriptions
    canceled_ids = {sub.user_id for sub in canceled_subs if sub.user_id in subscriber_ids}

    ground_truth = UserCount(count=len(canceled_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many teams does user_00020 have?",
        "response_schema": TeamTotal,
    },
    dataset="easy",
    labels=["testing"],
)
async def user_00020_team_total(ctx: EvalContext):
    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=["user_00020"])
    ).teams
    ground_truth = TeamTotal(team_count=len(teams))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["team_count"] == ground_truth.team_count
    ), f"Team count mismatch: got {ctx.output['team_count']}, expected {ground_truth.team_count}"


@eval(
    input={
        "prompt": "How many purchases has user_00005 made in total?",
        "response_schema": PurchaseCount,
    },
    dataset="easy",
    labels=["testing"],
)
async def user_00005_purchase_total(ctx: EvalContext):
    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(user_ids=["user_00005"])
    ).purchases
    ground_truth = PurchaseCount(purchase_count=len(purchases))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["purchase_count"] == ground_truth.purchase_count
    ), f"Purchase count mismatch: got {ctx.output['purchase_count']}, expected {ground_truth.purchase_count}"


@eval(
    input={
        "prompt": "How many sessions did user_00005 log on 2025-11-19?",
        "response_schema": SessionsTotal,
    },
    dataset="easy",
    labels=["testing"],
)
async def user_00005_sessions_on_2025_11_19(ctx: EvalContext):
    rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=["user_00005"], date_from="2025-11-19", date_to="2025-11-19"
        )
    ).engagement
    if not rows:
        raise ValueError("No engagement rows found for user_00005 on 2025-11-19")
    total_sessions = sum(row.sessions for row in rows)

    ground_truth = SessionsTotal(sessions=total_sessions)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["sessions"] == ground_truth.sessions
    ), f"Session count mismatch: got {ctx.output['sessions']}, expected {ground_truth.sessions}"


@eval(
    input={
        "prompt": "How many LATAM users have an active ultra subscription and at least two teams?",
        "response_schema": UserCount,
    },
    dataset="medium",
    labels=["testing"],
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
    labels=["testing"],
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
    labels=["testing"],
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
        "prompt": "Between 2025-11-10 and 2025-11-19, which NA free user logged the most sessions? Provide the user id and total sessions.",
        "response_schema": UserSessions,
    },
    dataset="medium",
    labels=["testing"],
)
async def na_free_top_sessions_window(ctx: EvalContext):
    na_free = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="free")
    ).users
    na_lookup = {user.id: user for user in na_free}

    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(na_lookup.keys()),
            date_from="2025-11-10",
            date_to="2025-11-19",
        )
    ).engagement

    if not engagement_rows:
        raise ValueError("No engagement data found for NA free users in window")

    session_totals = defaultdict(int)
    for row in engagement_rows:
        session_totals[row.user_id] += row.sessions

    max_sessions = max(session_totals.values())
    leaders = [
        na_lookup[user_id] for user_id, total in session_totals.items() if total == max_sessions
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserSessions(
        user_id=top_user.id, total_sessions=session_totals[top_user.id]
    )
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert (
        ctx.output["total_sessions"] == ground_truth.total_sessions
    ), f"Session total mismatch: got {ctx.output['total_sessions']}, expected {ground_truth.total_sessions}"


@eval(
    input={
        "prompt": "Between 2025-11-01 and 2025-11-19, which LATAM whale logged the most total minutes? Provide the user id and total minutes.",
        "response_schema": UserMinutes,
    },
    dataset="medium",
    labels=["testing"],
)
async def latam_whale_top_minutes(ctx: EvalContext):
    latam_whales = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in latam_whales}
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(whale_lookup.keys()),
            date_from="2025-11-01",
            date_to="2025-11-19",
        )
    ).engagement

    if not engagement_rows:
        raise ValueError("No engagement data found for LATAM whales in window")

    minute_totals = defaultdict(int)
    for row in engagement_rows:
        minute_totals[row.user_id] += row.minutes_played

    max_minutes = max(minute_totals.values())
    leaders = [
        whale_lookup[user_id]
        for user_id, total in minute_totals.items()
        if total == max_minutes
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserMinutes(
        user_id=top_user.id, total_minutes=minute_totals[top_user.id]
    )
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert (
        ctx.output["total_minutes"] == ground_truth.total_minutes
    ), f"Minutes mismatch: got {ctx.output['total_minutes']}, expected {ground_truth.total_minutes}"


@eval(
    input={
        "prompt": "How many APAC users with an active premium subscription have bought a coins_pack after 2025-09-15?",
        "response_schema": UserCount,
    },
    dataset="medium",
    labels=["testing"],
)
async def apac_active_premium_coins_after_sept15_count(ctx: EvalContext):
    apac_users = ctx.original_world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = {user.id for user in apac_users}
    active_premium = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="premium", status="active")
    ).subscriptions
    premium_ids = {sub.user_id for sub in active_premium if sub.user_id in apac_ids}

    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(purchased_after="2025-09-15")
    ).purchases
    buyer_ids = {
        purchase.user_id
        for purchase in purchases
        if purchase.sku == "coins_pack" and purchase.user_id in premium_ids
    }

    ground_truth = UserCount(count=len(buyer_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "How many EU users have an active premium subscription and at least one purchase?",
        "response_schema": UserCount,
    },
    dataset="medium",
    labels=["testing"],
)
async def eu_active_premium_with_purchase_count(ctx: EvalContext):
    eu_users = ctx.original_world.list_users(ListUsersInput(region="EU")).users
    eu_ids = {user.id for user in eu_users}
    premium_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="premium", status="active")
    ).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in eu_ids}

    purchases = ctx.original_world.list_purchases(ListPurchasesInput()).purchases
    purchasers = {purchase.user_id for purchase in purchases if purchase.user_id in eligible_ids}

    ground_truth = UserCount(count=len(purchasers))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "Between 2025-11-13 and 2025-11-19, which LATAM subscriber played the most ranked matches? Provide the user id and full name.",
        "response_schema": UserIdentity,
    },
    dataset="medium",
    labels=["testing"],
)
async def latam_subscriber_top_ranked_mid_nov(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(subscriber_lookup.keys()),
            date_from="2025-11-13",
            date_to="2025-11-19",
        )
    ).engagement

    if not engagement_rows:
        raise ValueError("No engagement data for LATAM subscribers in window")

    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    max_ranked = max(ranked_totals.values())
    leaders = [
        subscriber_lookup[user_id]
        for user_id, total in ranked_totals.items()
        if total == max_ranked
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserIdentity(user_id=top_user.id, full_name=top_user.name)
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
        "prompt": "How many APAC users with an active ultra subscription created at least one team after 2025-10-01?",
        "response_schema": UserCount,
    },
    dataset="medium",
    labels=["testing"],
)
async def apac_ultra_recent_team_count(ctx: EvalContext):
    apac_users = ctx.original_world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = {user.id for user in apac_users}
    active_ultra = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions
    ultra_ids = {sub.user_id for sub in active_ultra if sub.user_id in apac_ids}

    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=list(ultra_ids), created_after="2025-10-01")
    ).teams
    user_ids_with_recent_team = {team.user_id for team in teams}

    ground_truth = UserCount(count=len(user_ids_with_recent_team))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


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

    ctx.reference = {"expected_count": zero_team_count}

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#crm-campaigns", limit=10)
    ).messages
    matching = next(
        (
            message
            for message in messages
            if str(zero_team_count) in message.text and "EU free" in message.text
        ),
        None,
    )

    ctx.output = {
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    }

    assert (
        matching is not None
    ), "No #crm-campaigns message found summarizing EU free users with zero teams"


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

    ctx.reference = {
        "expected_user_id": top_user.id,
        "expected_flag_type": "vip_support",
        "total_amount": max_spend,
    }

    created_flags = [
        flag
        for flag in ctx.agent.world.flags.values()
        if flag.user_id == top_user.id and flag.flag_type == "vip_support"
    ]

    ctx.output = {
        "flag_created": bool(created_flags),
        "flag_reason": created_flags[0].reason if created_flags else None,
    }

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
    ctx.reference = {"eligible_ids": sorted(eligible_ids), "note": note_text}

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=eligible_ids)
    ).users
    updated_ids = [
        user.id for user in updated_users if user.admin_note == note_text
    ]

    ctx.output = {"updated_ids": sorted(updated_ids)}

    assert len(updated_ids) == len(
        eligible_ids
    ), f"Expected {len(eligible_ids)} notes updated, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "What is the name of the most recently created team for user_00024?",
        "response_schema": TeamName,
    },
    dataset="medium",
    labels=["testing"],
)
async def most_recent_team_name_user_00024(ctx: EvalContext):
    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=["user_00024"])
    ).teams
    if not teams:
        raise ValueError("User user_00024 has no teams")

    latest_team = max(teams, key=lambda team: team.created_at)
    ground_truth = TeamName(team_name=latest_team.name)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["team_name"] == ground_truth.team_name
    ), f"Team name mismatch: got {ctx.output['team_name']}, expected {ground_truth.team_name}"


@eval(
    input={
        "prompt": "For the earliest LATAM subscriber by signup date, how many purchases were made after 2025-06-01?",
        "response_schema": PurchaseCount,
    },
    dataset="medium",
    labels=["testing"],
)
async def earliest_latam_subscriber_purchase_count_after_june(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    if not latam_subscribers:
        raise ValueError("No LATAM subscribers found")

    earliest_user = min(latam_subscribers, key=lambda user: user.signup_date)
    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(
            user_ids=[earliest_user.id], purchased_after="2025-06-01"
        )
    ).purchases

    ground_truth = PurchaseCount(purchase_count=len(purchases))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["purchase_count"] == ground_truth.purchase_count
    ), f"Purchase count mismatch: got {ctx.output['purchase_count']}, expected {ground_truth.purchase_count}"


@eval(
    input={
        "prompt": "Which SKU has the highest number of purchases among LATAM subscribers? Provide only the SKU.",
        "response_schema": SkuName,
    },
    dataset="medium",
    labels=["testing"],
)
async def top_sku_latam_subscribers(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    latam_ids = {user.id for user in latam_subscribers}
    purchases = ctx.original_world.list_purchases(ListPurchasesInput()).purchases

    sku_counts = Counter(
        purchase.sku for purchase in purchases if purchase.user_id in latam_ids
    )
    if not sku_counts:
        raise ValueError("No purchases for LATAM subscribers")

    max_count = max(sku_counts.values())
    top_skus = sorted([sku for sku, count in sku_counts.items() if count == max_count])
    top_sku = top_skus[0]

    ground_truth = SkuName(sku=top_sku)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["sku"] == ground_truth.sku
    ), f"SKU mismatch: got {ctx.output['sku']}, expected {ground_truth.sku}"


@eval(
    input={
        "prompt": "Among APAC whales, take the user with the most sessions between 2025-11-13 and 2025-11-19. What is their total ranked matches in that same window?",
        "response_schema": RankedMatchesTotal,
    },
    dataset="medium",
    labels=["testing"],
)
async def apac_whale_ranked_matches_for_top_sessions(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in apac_whales}
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(whale_lookup.keys()),
            date_from="2025-11-13",
            date_to="2025-11-19",
        )
    ).engagement

    if not engagement_rows:
        raise ValueError("No engagement data for APAC whales in window")

    session_totals = defaultdict(int)
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        session_totals[row.user_id] += row.sessions
        ranked_totals[row.user_id] += row.ranked_matches

    max_sessions = max(session_totals.values())
    leaders = [
        whale_lookup[user_id]
        for user_id, total in session_totals.items()
        if total == max_sessions
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = RankedMatchesTotal(ranked_matches=ranked_totals[top_user.id])
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["ranked_matches"] == ground_truth.ranked_matches
    ), f"Ranked matches mismatch: got {ctx.output['ranked_matches']}, expected {ground_truth.ranked_matches}"


@eval(
    input={
        "prompt": "How many LATAM users with an active ultra subscription recorded zero ranked matches between 2025-11-13 and 2025-11-19?",
        "response_schema": UserCount,
    },
    dataset="hard",
    labels=["testing"],
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


@eval(
    input={
        "prompt": "Which EU whale with an active ultra subscription created the earliest team after 2025-10-01? Provide the user id and full name.",
        "response_schema": UserIdentity,
    },
    dataset="hard",
    labels=["testing"],
)
async def eu_ultra_whale_earliest_post_oct_team(ctx: EvalContext):
    eu_whales = ctx.original_world.list_users(
        ListUsersInput(region="EU", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in eu_whales}
    active_ultra = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions
    eligible_ids = {sub.user_id for sub in active_ultra if sub.user_id in whale_lookup}

    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=list(eligible_ids), created_after="2025-10-01")
    ).teams
    if not teams:
        raise ValueError("No teams found for EU ultra whales after 2025-10-01")

    earliest_team = min(teams, key=lambda team: team.created_at)
    owner = whale_lookup[earliest_team.user_id]

    ground_truth = UserIdentity(user_id=owner.id, full_name=owner.name)
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
        "prompt": "Among LATAM subscribers with at least two teams, who has the highest total purchase amount? Provide the user id and total amount.",
        "response_schema": UserAmount,
    },
    dataset="hard",
    labels=["testing"],
)
async def latam_subscriber_highest_spend_two_teams(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=list(subscriber_lookup.keys()))
    ).teams
    team_counts = Counter(team.user_id for team in teams)
    eligible_ids = {user_id for user_id, count in team_counts.items() if count >= 2}

    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(user_ids=list(eligible_ids))
    ).purchases
    spend_totals = defaultdict(float)
    for purchase in purchases:
        spend_totals[purchase.user_id] += purchase.amount

    if not spend_totals:
        raise ValueError("No purchases for eligible LATAM subscribers")

    max_spend = max(spend_totals.values())
    leaders = [
        subscriber_lookup[user_id]
        for user_id, total in spend_totals.items()
        if total == max_spend
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserAmount(user_id=top_user.id, total_amount=spend_totals[top_user.id])
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert abs(ctx.output["total_amount"] - ground_truth.total_amount) < 1e-6, (
        f"Spend mismatch: got {ctx.output['total_amount']}, "
        f"expected {ground_truth.total_amount}"
    )


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
    ctx.reference = {
        "expected_user_ids": [user.id for user in top_three],
        "flag_type": "ranked_surge",
        "ranked_totals": {user.id: ranked_totals.get(user.id, 0) for user in top_three},
    }

    created_flags = [
        flag for flag in ctx.agent.world.flags.values() if flag.flag_type == "ranked_surge"
    ]
    created_map = {flag.user_id: flag for flag in created_flags}

    ctx.output = {
        "created_user_ids": sorted(created_map.keys()),
    }

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
    ctx.reference = {"eligible_ids": sorted(eligible_ids), "note": note_text}

    updated_users = ctx.agent.world.list_users(
        ListUsersInput(user_ids=list(eligible_ids))
    ).users
    updated_ids = [user.id for user in updated_users if user.admin_note == note_text]

    ctx.output = {"updated_ids": sorted(updated_ids)}

    assert len(updated_ids) == len(
        eligible_ids
    ), f"Expected {len(eligible_ids)} notes updated, got {len(updated_ids)}"


@eval(
    input={
        "prompt": "Among APAC whales with at least one purchase after 2025-10-01 and zero ranked matches between 2025-11-15 and 2025-11-19, who signed up first? Provide the user id and full name.",
        "response_schema": UserIdentity,
    },
    dataset="hard",
    labels=["testing"],
)
async def apac_whale_zero_ranked_with_purchase(ctx: EvalContext):
    apac_whales = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in apac_whales}

    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(purchased_after="2025-10-01")
    ).purchases
    buyers = {purchase.user_id for purchase in purchases if purchase.user_id in whale_lookup}

    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(whale_lookup.keys()),
            date_from="2025-11-15",
            date_to="2025-11-19",
        )
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches

    eligible_users = [
        user
        for user in apac_whales
        if user.id in buyers and ranked_totals.get(user.id, 0) == 0
    ]
    if not eligible_users:
        raise ValueError("No eligible APAC whales found")

    earliest_user = min(eligible_users, key=lambda user: user.signup_date)
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
        "prompt": "Which EU user with an active premium subscription had the highest average minutes per session between 2025-11-10 and 2025-11-19? Provide the user id and full name.",
        "response_schema": UserIdentity,
    },
    dataset="hard",
    labels=["testing"],
)
async def eu_premium_highest_average_minutes(ctx: EvalContext):
    eu_users = ctx.original_world.list_users(ListUsersInput(region="EU")).users
    eu_lookup = {user.id: user for user in eu_users}
    premium_active = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="premium", status="active")
    ).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in eu_lookup}

    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(
            user_ids=list(eligible_ids),
            date_from="2025-11-10",
            date_to="2025-11-19",
        )
    ).engagement

    minutes_totals = defaultdict(int)
    session_totals = defaultdict(int)
    for row in engagement_rows:
        minutes_totals[row.user_id] += row.minutes_played
        session_totals[row.user_id] += row.sessions

    averages = {
        user_id: minutes_totals[user_id] / session_totals[user_id]
        for user_id in eligible_ids
        if session_totals[user_id] > 0
    }
    if not averages:
        raise ValueError("No eligible engagement rows to compute averages")

    max_avg = max(averages.values())
    leaders = [
        eu_lookup[user_id] for user_id, avg in averages.items() if avg == max_avg
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserIdentity(user_id=top_user.id, full_name=top_user.name)
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
        "prompt": "Among LATAM subscribers with exactly one team, who has the highest total purchase amount? Provide the user id and total amount.",
        "response_schema": UserAmount,
    },
    dataset="hard",
    labels=["testing"],
)
async def latam_single_team_top_spender(ctx: EvalContext):
    latam_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="LATAM", segment="subscriber")
    ).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=list(subscriber_lookup.keys()))
    ).teams
    team_counts = Counter(team.user_id for team in teams)
    eligible_ids = {user_id for user_id, count in team_counts.items() if count == 1}

    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(user_ids=list(eligible_ids))
    ).purchases
    spend_totals = defaultdict(float)
    for purchase in purchases:
        spend_totals[purchase.user_id] += purchase.amount

    if not spend_totals:
        raise ValueError("No purchases for eligible LATAM subscribers")

    max_spend = max(spend_totals.values())
    leaders = [
        subscriber_lookup[user_id]
        for user_id, total in spend_totals.items()
        if total == max_spend
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserAmount(
        user_id=top_user.id, total_amount=spend_totals[top_user.id]
    )
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert abs(ctx.output["total_amount"] - ground_truth.total_amount) < 1e-6, (
        f"Spend mismatch: got {ctx.output['total_amount']}, "
        f"expected {ground_truth.total_amount}"
    )


@eval(
    input={
        "prompt": "Post a message in #product-notes listing how many teams EU whales created after 2025-10-15. Include the number and the date cutover in the text.",
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

    ctx.reference = {"expected_count": count, "cutover": "2025-10-15"}

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#product-notes", limit=10)
    ).messages
    matching = next(
        (
            message
            for message in messages
            if str(count) in message.text and "2025-10-15" in message.text
        ),
        None,
    )

    ctx.output = {
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    }

    assert (
        matching is not None
    ), "No #product-notes message found summarizing EU whale team creations"


@eval(
    input={
        "prompt": "How many purchases did the top buyer make after 2025-10-01?",
        "response_schema": PurchaseCount,
    },
    dataset="hard",
    labels=["testing"],
)
async def purchase_count_top_buyer_after_oct(ctx: EvalContext):
    purchases = ctx.original_world.list_purchases(
        ListPurchasesInput(purchased_after="2025-10-01")
    ).purchases
    if not purchases:
        raise ValueError("No purchases found after 2025-10-01")

    purchase_counts = Counter(purchase.user_id for purchase in purchases)
    max_count = max(purchase_counts.values())

    ground_truth = PurchaseCount(purchase_count=max_count)
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["purchase_count"] == ground_truth.purchase_count
    ), f"Purchase count mismatch: got {ctx.output['purchase_count']}, expected {ground_truth.purchase_count}"


@eval(
    input={
        "prompt": "Post an alert in #ops-alerts noting how many APAC whales recorded zero ranked matches between 2025-11-17 and 2025-11-19. Include the count and the date range.",
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

    ctx.reference = {
        "expected_count": count,
        "start": "2025-11-17",
        "end": "2025-11-19",
    }

    messages = ctx.agent.world.list_messages(
        ListMessagesInput(channel="#ops-alerts", limit=10)
    ).messages
    matching = next(
        (
            message
            for message in messages
            if str(count) in message.text
            and "2025-11-17" in message.text
            and "2025-11-19" in message.text
        ),
        None,
    )

    ctx.output = {
        "message_found": matching is not None,
        "message_text": matching.text if matching else None,
    }

    assert (
        matching is not None
    ), "No #ops-alerts message found for APAC whales with zero ranked matches in the window"


@eval(
    input={
        "prompt": "Across all engagement data, which NA whale accumulated the most total minutes played? Provide the user id and total minutes.",
        "response_schema": UserMinutes,
    },
    dataset="hard",
    labels=["testing"],
)
async def na_whale_highest_total_minutes(ctx: EvalContext):
    na_whales = ctx.original_world.list_users(
        ListUsersInput(region="NA", segment="whale")
    ).users
    whale_lookup = {user.id: user for user in na_whales}
    engagement_rows = ctx.original_world.list_engagement(
        ListEngagementInput(user_ids=list(whale_lookup.keys()))
    ).engagement

    if not engagement_rows:
        raise ValueError("No engagement data for NA whales")

    minute_totals = defaultdict(int)
    for row in engagement_rows:
        minute_totals[row.user_id] += row.minutes_played

    max_minutes = max(minute_totals.values())
    leaders = [
        whale_lookup[user_id]
        for user_id, total in minute_totals.items()
        if total == max_minutes
    ]
    top_user = min(leaders, key=lambda user: user.signup_date)

    ground_truth = UserMinutes(
        user_id=top_user.id, total_minutes=minute_totals[top_user.id]
    )
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["user_id"] == ground_truth.user_id
    ), f"User id mismatch: got {ctx.output['user_id']}, expected {ground_truth.user_id}"
    assert (
        ctx.output["total_minutes"] == ground_truth.total_minutes
    ), f"Minutes mismatch: got {ctx.output['total_minutes']}, expected {ground_truth.total_minutes}"


@eval(
    input={
        "prompt": "How many APAC subscribers have an active subscription and exactly three teams created before 2025-10-01?",
        "response_schema": UserCount,
    },
    dataset="hard",
    labels=["testing"],
)
async def apac_active_subscribers_three_pre_oct_teams(ctx: EvalContext):
    apac_subscribers = ctx.original_world.list_users(
        ListUsersInput(region="APAC", segment="subscriber")
    ).users
    subscriber_ids = {user.id for user in apac_subscribers}
    active_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(status="active", user_ids=list(subscriber_ids))
    ).subscriptions
    active_ids = {sub.user_id for sub in active_subs}

    teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=list(active_ids))
    ).teams
    pre_oct_counts = Counter(
        team.user_id for team in teams if team.created_at < date(2025, 10, 1)
    )
    eligible_ids = [user_id for user_id in active_ids if pre_oct_counts.get(user_id, 0) == 3]

    ground_truth = UserCount(count=len(eligible_ids))
    ctx.reference = ground_truth.model_dump_json()

    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"

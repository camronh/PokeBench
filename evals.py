from twevals import eval, EvalContext
from agent import Agent
from pydantic import BaseModel, Field
from models import (
    ListEngagementInput,
    ListSubscriptionsInput,
    ListTeamsInput,
    ListUsersInput,
)
from world_runtime import World


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


# Set target as the global target
twevals_defaults = {
    "target": target,
}


# EVALS:
class Name(BaseModel):
    full_name: str = Field(..., description="The full name of the user, verbatim")


@eval(
    input={
        "prompt": "What name belongs to user user_00000?",
        "response_schema": Name,
    },
    dataset="easy",
)
async def get_user_by_id(ctx: EvalContext):

    # Calculate ground truth
    user_result = ctx.original_world.list_users(
        ListUsersInput(user_ids=["user_00000"])
    ).users

    ground_truth = Name(full_name=user_result[0].name)
    ctx.reference = ground_truth.model_dump_json()

    # Validate agent responded with tool calls
    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    # Validate the agent's response against ground truth
    assert (
        ctx.output["full_name"].lower() == ground_truth.full_name.lower()
    ), f"Name mismatch: got '{ctx.output['full_name']}', expected '{ground_truth.full_name}'"


class SubscriptionPlan(BaseModel):
    plan: str = Field(
        ..., description="The subscription plan name (e.g., free, premium, ultra)"
    )


@eval(
    input={
        "prompt": "What subscription plan is user_00006 currently on?",
        "response_schema": SubscriptionPlan,
    },
    dataset="easy",
)
async def get_subscription_plan(ctx: EvalContext):

    # Calculate ground truth
    user_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(user_ids=["user_00006"], status="active")
    ).subscriptions

    ground_truth = SubscriptionPlan(plan=user_subs[0].plan)
    ctx.reference = ground_truth.model_dump_json()

    # Validate agent responded with tool calls
    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    # Validate the agent's response against ground truth
    assert (
        ctx.output["plan"].lower() == ground_truth.plan.lower()
    ), f"Plan mismatch: got '{ctx.output['plan']}', expected '{ground_truth.plan}'"


class TeamCount(BaseModel):
    count: int = Field(..., description="The number of teams the user has created")


@eval(
    input={
        "prompt": "How many teams does user_00042 have?",
        "response_schema": TeamCount,
    },
    dataset="easy",
)
async def count_user_teams(ctx: EvalContext):

    # Calculate ground truth - count teams for specific user
    user_teams = ctx.original_world.list_teams(
        ListTeamsInput(user_ids=["user_00042"])
    ).teams
    ground_truth = TeamCount(count=len(user_teams))
    ctx.reference = ground_truth.model_dump_json()

    # Validate agent responded with tool calls
    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    # Validate the agent's response against ground truth
    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


class UserCount(BaseModel):
    count: int = Field(..., description="The number of users matching the criteria")


@eval(
    input={
        "prompt": "How many users in the LATAM region currently have active ultra subscriptions?",
        "response_schema": UserCount,
    },
    dataset="medium",
)
async def count_ultra_subs_by_region(ctx: EvalContext):

    # Calculate ground truth
    latam_users = ctx.original_world.list_users(
        ListUsersInput(region="LATAM")
    ).users
    if not latam_users:
        raise ValueError("No LATAM users available for ground truth calculation")

    latam_user_ids = {user.id for user in latam_users}

    ultra_subs = ctx.original_world.list_subscriptions(
        ListSubscriptionsInput(plan="ultra", status="active")
    ).subscriptions

    ultra_user_ids = {sub.user_id for sub in ultra_subs}

    # Count how many LATAM users have active ultra subscriptions
    count = len(latam_user_ids & ultra_user_ids)
    ground_truth = UserCount(count=count)
    ctx.reference = ground_truth.model_dump_json()

    # Validate agent responded with tool calls
    assert (
        ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1
    ), "Agent did not respond with the response tool"

    # Validate the agent's response against ground truth
    assert (
        ctx.output["count"] == ground_truth.count
    ), f"Count mismatch: got {ctx.output['count']}, expected {ground_truth.count}"


@eval(
    input={
        "prompt": "Create a churn_risk flag for user_00005 with a reason based on their engagement data. Look up their engagement metrics and provide a specific reason.",
        "response_schema": None,
    },
    dataset="medium",
)
async def create_churn_risk_flag(ctx: EvalContext):
    # Calculate expected flag details
    target_user_id = "user_00005"
    expected_flag_type = "churn_risk"

    # Get ground truth engagement data for validation
    user_engagement = ctx.original_world.list_engagement(
        ListEngagementInput(user_ids=[target_user_id])
    ).engagement

    total_sessions = sum(eng.sessions for eng in user_engagement)
    total_minutes = sum(eng.minutes_played for eng in user_engagement)

    # Set context reference before assertions
    ctx.reference = {
        "expected_user_id": target_user_id,
        "expected_flag_type": expected_flag_type,
        "engagement_context": {
            "total_sessions": total_sessions,
            "total_minutes": total_minutes,
        },
    }

    # Check that flag was created in the agent's world
    agent_flags = [
        flag
        for flag in ctx.agent.world.flags.values()
        if flag.user_id == target_user_id and flag.flag_type == expected_flag_type
    ]

    # Set context output before assertions
    ctx.output = {
        "flag_created": bool(agent_flags),
        "user_id": agent_flags[0].user_id if agent_flags else None,
        "flag_type": agent_flags[0].flag_type if agent_flags else None,
        "reason": agent_flags[0].reason if agent_flags else None,
    }

    assert len(agent_flags) > 0, f"No churn_risk flag created for {target_user_id}"

    flag = agent_flags[0]
    reason_lower = flag.reason.lower()

    # Validate flag has a substantive reason
    assert len(flag.reason) > 20, f"Flag reason too short: '{flag.reason}'"

    # Check that reason mentions engagement or activity
    assert any(
        keyword in reason_lower
        for keyword in ["engagement", "session", "minute", "play", "activity", "low"]
    ), f"Flag reason doesn't mention engagement metrics: '{flag.reason}'"

from twevals import eval, EvalContext
from agent import Agent
from pydantic import BaseModel, Field
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
    ground_truth = Name(full_name=ctx.original_world.users["user_00000"].name)
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
    user_subs = [
        sub
        for sub in ctx.original_world.subscriptions.values()
        if sub.user_id == "user_00006"
    ]
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
    user_teams = [
        team for team in ctx.original_world.teams.values() if team.user_id == "user_00042"
    ]
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
    # Get all LATAM users
    latam_user_ids = {
        user.id for user in ctx.original_world.users.values() if user.region == "LATAM"
    }

    # Get all users with active ultra subscriptions
    ultra_user_ids = {
        sub.user_id
        for sub in ctx.original_world.subscriptions.values()
        if sub.plan == "ultra" and sub.status == "active"
    }

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

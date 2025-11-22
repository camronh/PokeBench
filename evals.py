from twevals import eval, EvalContext
from agent import Agent
from pydantic import BaseModel, Field
from world_runtime import World


@eval(
    input="What name belongs to user user_00000?",
    dataset="easy",
    default_score_key="correct",
    metadata={"difficulty": "easy"}
)
async def get_user_by_id(ctx: EvalContext):
    original_world = World()

    class Name(BaseModel):
        full_name: str = Field(..., description="The full name of the user, verbatim")

    # Calculate ground truth
    ground_truth = Name(full_name=original_world.users["user_00000"].name)
    ctx.reference = ground_truth.model_dump_json()

    # Run agent
    agent = await Agent.create_and_run(ctx.input, Name)


    # Validate agent responded with tool calls
    assert agent.output.tool_calls and len(agent.output.tool_calls) >= 1, \
        "Agent did not respond with the response tool"

    ctx.add_output(agent.output.tool_calls[0]["args"])

    # Validate the agent's response against ground truth
    response_args = agent.output.tool_calls[0]["args"]
    assert response_args["full_name"].lower() == ground_truth.full_name.lower(), \
        f"Name mismatch: got '{response_args['full_name']}', expected '{ground_truth.full_name}'"


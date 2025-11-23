# Generate PokeBench Evals

Generate synthetic evaluation cases for the PokeBench eval suite. These evals test an AI agent's ability to use tools correctly with synthetic data.

ONLY work in the `evals.py` file.

## Core Principles

### 1. Test Tool Usage, Not Knowledge

- ❌ BAD: "What are the primary and secondary types of Charizard?" (agent could guess from training data)
- ✅ GOOD: "How many teams does user_00042 have?" (requires tool usage on synthetic data)

### 2. Use Only Synthetic Data

All test data comes from `world_runtime.py`. Available entities:

- **Users**: id, name, email, region, signup_date, segment, admin_note
- **Subscriptions**: user_id, plan, status, started_at, ended_at
- **Teams**: user_id, name, created_at, pokemon_ids
- **Flags**: user_id, flag_type, reason, created_at
- **Engagement**: user_id, date, sessions, minutes_played, ranked_matches
- **Pokemon**: id, name, primary_type, secondary_type (avoid testing - in-distribution!)
- **Purchases**: user_id, pokemon_id, sku, amount, purchased_at
- **Messages**: channel, text, created_at

## Eval Structure

All evals follow this pattern:

```python
# Define response schema (or None for mutation-only)
class ResponseSchema(BaseModel):
    field: type = Field(..., description="...")

@eval(
    input={
        "prompt": "Clear, specific task description",
        "response_schema": ResponseSchema,  # or None for mutations
    },
    dataset="easy|medium|hard",
)
async def eval_name(ctx: EvalContext):
    # Calculate ground truth from ctx.original_world
    ground_truth = ...
    ctx.reference = ...

    # For evals with response_schema: validate agent used response tool
    assert ctx.agent.output.tool_calls and len(ctx.agent.output.tool_calls) >= 1, \
        "Agent did not respond with the response tool"

    # For mutation evals: check ctx.agent.world for changes
    # For response evals: check ctx.output

    # Validate ONLY what's explicitly mentioned in the prompt
    assert ctx.output["field"] == ground_truth.field, \
        f"Mismatch: got {ctx.output['field']}, expected {ground_truth.field}"
```

## Error Handling Rules

**Use assertions ONLY for validation criteria (test failures):**

- Agent didn't use the response tool when required
- Agent's output doesn't match ground truth
- Mutation didn't occur as specified in prompt
- Output doesn't meet requirements explicitly stated in prompt

**Use `raise ValueError()` for environment/setup issues (errors):**

- Data not found in seed (e.g., "User not found")
- No eligible data for ground truth calculation
- Invalid test setup

## Common Pitfalls

1. **Testing in-distribution knowledge**: Don't test Pokemon types/names knowledge or anything that may be guessable from training data
2. **Too many assertions**: Only validate what's in the prompt. For it to be a fair test, the validations need be specified in the prompt.
3. **Non-validation errors**: Use `raise ValueError()` for setup issues, not `assert`
4. **Set Context Too Late**: Twevals includes the context data at time of exit. Try to set the context as early as possible. For example, try to set the ctx.output and/or ctx.reference before going into assertions.


### Mutation Evals

Mutation evals are evals that test that the agent can mutate the world state correctly. There is no hard rule about how many mutation evals to generate, but try to include at least 1 for every 4 response evals.

## Lifecycle

1. Consider the data that is available and generate questions or tasks that can be answered from the available data.
2. Write test scripts to test that this data is reachable using the only the public tool methods in `world_runtime.py` and run them to validate that the data is reachable. For mutation evals, validate that the mutation is possible and works as expected.
3. Write the evals in the `evals.py` file. Try to style them like the existing evals. Order them by difficulty.
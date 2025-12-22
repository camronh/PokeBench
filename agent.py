from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pydantic import BaseModel
from world_runtime import World
import anthropic
from typing import Any, Callable, Dict, List
from langsmith.wrappers import wrap_anthropic
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree


RESPONSE_TOOL_NAME = "respond_to_admin_user"
CODE_EXECUTION_TOOL_TYPE = "code_execution_20250825"
PROGRAMMATIC_BETA = "advanced-tool-use-2025-11-20"

class AgentOutput:
    """Wrapper to match the expected output interface from evals.py"""

    def __init__(self, content: str, tool_calls: list = None):
        self._content = content
        self._tool_calls = tool_calls or []

    @property
    def content(self) -> str:
        return self._content

    @property
    def tool_calls(self) -> list:
        return self._tool_calls


def get_response_tool_anthropic(
    response_schema: BaseModel,
    allowed_callers: List[str] = None,
    cache: bool = True
) -> dict:
    """Create Anthropic tool definition for the response tool."""
    tool_def = {
        "name": RESPONSE_TOOL_NAME,
        "description": "Use this tool when you're ready to respond to the Admin user's request. The admin will ONLY be able to see your response if you use this tool.",
        "input_schema": response_schema.model_json_schema(),
    }
    if allowed_callers:
        tool_def["allowed_callers"] = allowed_callers
    if cache:
        tool_def["cache_control"] = {"type": "ephemeral"}
    return tool_def


class Agent:
    def __init__(
        self,
        model_name: str,
        response_schema: BaseModel = None,
        programmatic_tools: bool = False,
        include_output_schema: bool = False,
        truncate_output: bool = False
    ):
        self.model_name = model_name
        self.response_schema = response_schema
        self.programmatic_tools = programmatic_tools
        self.include_output_schema = include_output_schema
        self.truncate_output = truncate_output

        # Copy World to a new World
        self.world = World()

        # Build tool executor map
        self.tool_executors: Dict[str, Callable] = {}
        for name, meta in self.world.tool_map().items():
            input_model = meta["input_model"]
            func = meta["func"]

            def make_executor(f: Callable, in_model):
                def executor(**kwargs) -> Any:
                    args = in_model(**kwargs)
                    result = f(args)
                    return result.model_dump_json()

                return executor

            self.tool_executors[name] = make_executor(func, input_model)

        # Build Anthropic tools from World
        if programmatic_tools:
            # Programmatic mode: tools can only be called from code execution
            self.tools = [
                {"type": CODE_EXECUTION_TOOL_TYPE, "name": "code_execution"},
                *self.world.to_anthropic_tools(
                    allowed_callers=[CODE_EXECUTION_TOOL_TYPE],
                    include_output_schema=include_output_schema
                )
            ]
            if response_schema:
                # Response tool is last, so it gets cache_control
                self.tools.append(get_response_tool_anthropic(
                    response_schema,
                    allowed_callers=[CODE_EXECUTION_TOOL_TYPE],
                    cache=True
                ))
            else:
                # No response schema, cache the last world tool
                if self.tools:
                    self.tools[-1]["cache_control"] = {"type": "ephemeral"}
        else:
            # Regular mode: direct tool calling
            self.tools = self.world.to_anthropic_tools(
                include_output_schema=include_output_schema
            )
            if response_schema:
                # Response tool is last, so it gets cache_control
                self.tools.append(get_response_tool_anthropic(response_schema, cache=True))
            else:
                # No response schema, cache the last world tool
                if self.tools:
                    self.tools[-1]["cache_control"] = {"type": "ephemeral"}

        # Initialize client
        self.client = wrap_anthropic(anthropic.AsyncAnthropic())

        # These will be set after run()
        self.trace_id = None
        self.final_agent_state = None
        self.output = None
        self.container_id = None  # For programmatic tool calling
    @traceable
    async def run(self, prompt: str):
        # Capture the actual LangSmith trace ID
        run_tree = get_current_run_tree()
        if run_tree:
            self.trace_id = run_tree.trace_id
        else:
            # Fallback if tracing is disabled
            self.trace_id = None

        # Build system prompt with cache_control
        has_response_tool = self.response_schema is not None
        system_text = """You are a helpful assistant with access to various admin tools. Your job is to use those tools to answer the user's questions or accomplish tasks. The requests will only be solvable by using the tools. Do not attempt to answer using your own knowledge or information. Only use the tools to answer the user's questions or accomplish tasks."""
        if has_response_tool:
            system_text += f" You MUST respond to the user ONLY using the {RESPONSE_TOOL_NAME} tool."

        # System prompt as array with cache_control on the last block
        system_prompt = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        # Run the agentic loop
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        final_response = None

        while True:
            for message in messages:
                for block in message["content"]:
                    block.pop("cache_control", None)
            if self.programmatic_tools:
                for message in reversed(messages):
                    for block in reversed(message["content"]):
                        if block.get("type") == "text":
                            block["cache_control"] = {"type": "ephemeral"}
                            break
                    else:
                        continue
                    break
            else:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            # Make API call based on mode
            if self.programmatic_tools:
                # Programmatic mode: use beta API with code execution
                request_kwargs = {
                    "model": self.model_name,
                    "betas": [PROGRAMMATIC_BETA],
                    "max_tokens": 20000,
                    "system": system_prompt,
                    "tools": self.tools,
                    "messages": messages,
                }
                if self.container_id:
                    request_kwargs["container"] = self.container_id

                response = await self.client.beta.messages.create(**request_kwargs)

                # Track container for reuse
                if hasattr(response, 'container') and response.container:
                    self.container_id = response.container.id
            else:
                # Regular mode
                response = await self.client.messages.create(
                    model=self.model_name,
                    max_tokens=20000,
                    system=system_prompt,
                    tools=self.tools,
                    messages=messages,
                )

            final_response = response

            # Check stop condition
            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                should_exit = False

                for content_block in response.content:
                    if content_block.type == "server_tool_use":
                        # Code execution started - just continue processing
                        pass
                    elif content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id

                        # Check if this is a programmatic call
                        is_programmatic = (
                            hasattr(content_block, 'caller') and
                            hasattr(content_block.caller, 'type') and
                            content_block.caller.type == CODE_EXECUTION_TOOL_TYPE
                        )

                        # Check if response tool
                        if tool_name == RESPONSE_TOOL_NAME:
                            should_exit = True
                            break

                        # Execute tool
                        try:
                            result = self.tool_executors[tool_name](**tool_input)
                            result_str = str(result)
                            if len(result_str) > 5000 and self.truncate_output:
                                result_str = (
                                    f"{result_str[:5000]}... *Truncated to manage token limits*"
                                )
                        except Exception as e:
                            result_str = f"Error: {str(e)}"

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_str,
                            }
                        )
                    elif content_block.type == "code_execution_tool_result":
                        # Code execution completed - continue processing
                        pass

                if should_exit:
                    break

                # Continue conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": [block.model_dump(exclude_none=True) for block in response.content],
                    }
                )
                messages.append({"role": "user", "content": tool_results})
            else:
                # max_tokens or other stop reason
                break

        # Build final_agent_state for compatibility
        self.final_agent_state = {"messages": messages}

        # Build output object
        # Extract text content
        text_content = ""
        tool_calls = []

        for block in final_response.content:
            if hasattr(block, "text"):
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "name": block.name,
                        "args": block.input,
                        "id": block.id,
                    }
                )

        self.output = AgentOutput(content=text_content, tool_calls=tool_calls)

        return self.output.content

    @staticmethod
    async def create_and_run(
        prompt: str,
        model_name: str,
        response_schema: BaseModel = None,
        programmatic_tools: bool = False,
        include_output_schema: bool = True,
        truncate_output: bool = False,
    ):
        agent = Agent(
            model_name=model_name,
            response_schema=response_schema,
            programmatic_tools=programmatic_tools,
            include_output_schema=include_output_schema,
            truncate_output=truncate_output
        )
        await agent.run(prompt)
        return agent

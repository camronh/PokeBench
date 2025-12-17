from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import uuid
from pydantic import BaseModel
from world_runtime import World
import anthropic
from typing import Any, Callable, Dict
from langsmith.wrappers import wrap_anthropic


RESPONSE_TOOL_NAME = "respond_to_admin_user"


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


def get_response_tool_anthropic(response_schema: BaseModel) -> dict:
    """Create Anthropic tool definition for the response tool."""
    return {
        "name": RESPONSE_TOOL_NAME,
        "description": "Use this tool when you're ready to respond to the Admin user's request. The admin will ONLY be able to see your response if you use this tool.",
        "input_schema": response_schema.model_json_schema(),
    }


class Agent:
    def __init__(self, response_schema: BaseModel = None):
        self.response_schema = response_schema

        # Copy World to a new World
        self.world = World()

        # Build Anthropic tools from World
        self.tools = self.world.to_anthropic_tools()

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

        # Add response tool if schema provided
        if response_schema:
            self.tools.append(get_response_tool_anthropic(response_schema))

        # Initialize client
        self.client = wrap_anthropic(anthropic.AsyncAnthropic())

        # These will be set after run()
        self.trace_id = None
        self.final_agent_state = None
        self.output = None

    async def run(self, prompt: str):
        self.trace_id = uuid.uuid4()

        # Build system prompt
        has_response_tool = self.response_schema is not None
        system_prompt = """You are a helpful assistant with access to various admin tools. Your job is to use those tools to answer the user's questions or accomplish tasks. The requests will only be solvable by using the tools. Do not attempt to answer using your own knowledge or information. Only use the tools to answer the user's questions or accomplish tasks."""
        if has_response_tool:
            system_prompt += f" You MUST respond to the user ONLY using the {RESPONSE_TOOL_NAME} tool."

        # Run the agentic loop
        messages = [{"role": "user", "content": prompt}]
        final_response = None

        while True:
            response = await self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
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
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id

                        # Check if response tool
                        if tool_name == RESPONSE_TOOL_NAME:
                            should_exit = True
                            break

                        # Execute tool
                        try:
                            result = self.tool_executors[tool_name](**tool_input)
                            result_str = str(result)
                            if len(result_str) > 5000:
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

                if should_exit:
                    break

                # Continue conversation
                messages.append({"role": "assistant", "content": response.content})
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
    async def create_and_run(prompt: str, response_schema: BaseModel = None):
        agent = Agent(response_schema=response_schema)
        await agent.run(prompt)
        return agent

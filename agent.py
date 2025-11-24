from pydantic import BaseModel
from world_runtime import World
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AnyMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define state for the agent
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


RESPONSE_TOOL_NAME = "respond_to_admin_user"


def get_response_tool(response_schema: BaseModel) -> StructuredTool:
    return StructuredTool(
        name=RESPONSE_TOOL_NAME,
        description="Respond to the Admin user's request. The admin will ONLY be able to see your response if you use this tool.",
        func=lambda x: x,
        args_schema=response_schema,
    )

# Initialize the model
model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="low",
)

def create_poke_agent(tools: list[StructuredTool]):
    """
    Create a LangGraph agent that uses the provided tools.

    Args:
        tools: List of LangChain StructuredTool instances

    Returns:
        Compiled LangGraph agent
    """


    # Create tool lookup
    tools_by_name = {tool.name: tool for tool in tools}

    # Check if response_to_admin tool is present
    has_response_tool = RESPONSE_TOOL_NAME in tools_by_name

    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Build system prompt
    system_prompt = "You are a helpful assistant with access to various tools."
    if has_response_tool:
        system_prompt += (
            f" You MUST respond to the user ONLY using the {RESPONSE_TOOL_NAME} tool."
        )

    # Define the LLM node
    async def llm_call(state):
        """LLM decides whether to call a tool or not"""
        return {
            "messages": [
                await model_with_tools.ainvoke(
                    [SystemMessage(content=system_prompt)] + state["messages"]
                )
            ]
        }

    # Define the tool node
    def tool_node(state):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    # Truncate the observation to manage token limits
                    content=str(
                        f"{observation[:5000]}... *Truncated to manage token limits*"
                        if len(observation) > 5000
                        else observation
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": result}

    # Define logic to determine whether to end
    def should_continue(state: MessagesState):
        """Decide if we should continue or stop"""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, check if it's the response tool
        if last_message.tool_calls:
            # Check if any of the tool calls is the response_to_admin tool
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == RESPONSE_TOOL_NAME:
                    # Exit the graph if response_to_admin was used
                    return END

            # Otherwise, continue to the tool node
            return "tool_node"

        # If no tool calls, we stop
        return END

    # Build the graph
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile and return the agent
    return agent_builder.compile(name="PokeAgent")


class Agent:
    def __init__(self, response_schema: BaseModel = None):
        self.response_schema = response_schema

        # Copy World to a new World
        self.world = World()

        # LangChain tools
        self.tools = self.world.to_langchain_tools()

        # Add response schema as a tool if provided
        if response_schema:
            self.tools.append(get_response_tool(response_schema))

    async def run(self, prompt: str):
        runner = create_poke_agent(tools=self.tools)
        self.final_agent_state = await runner.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        self.output: AIMessage = self.final_agent_state["messages"][-1]
        return self.output.content

    @staticmethod
    async def create_and_run(prompt: str, response_schema: BaseModel = None):
        agent = Agent(response_schema=response_schema)
        await agent.run(prompt)
        return agent

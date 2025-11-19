from pydantic import BaseModel
from world_runtime import World
from langchain_core.tools import StructuredTool


def get_response_tool(response_schema: BaseModel) -> StructuredTool:    
    return StructuredTool(
        name="response_to_admin",
        description="Respond to the Admin user's request. The admin will ONLY be able to see your response if you use this tool.",
        func=lambda x: x,
        args_schema=response_schema,
    )

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

    

    async def run(self, prompt: str) -> AgentOutput:
        runner = create_poke_agent(tools=self.tools)
        self.output = await runner.ainvoke({"input": prompt})
        return self.output


    @staticmethod
    async def create_and_run(prompt: str, response_schema: BaseModel = None) :
        agent = Agent(response_schema=response_schema)
        await agent.run(prompt)
        return agent




    

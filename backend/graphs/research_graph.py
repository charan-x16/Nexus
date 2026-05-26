from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agents.planner import PlannerAgent
from backend.agents.writer import WriterAgent
from backend.schemas.workflow import WorkflowState

planner_agent = PlannerAgent()
writer_agent = WriterAgent()


async def planner_node(state: WorkflowState) -> WorkflowState:
    return await planner_agent.run(state)


async def writer_node(state: WorkflowState) -> WorkflowState:
    return await writer_agent.run(state)


workflow = StateGraph(WorkflowState)
workflow.add_node("planner", planner_node)
workflow.add_node("writer", writer_node)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", END)

compiled_graph = workflow.compile(checkpointer=MemorySaver())

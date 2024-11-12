from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from states.states import AgentState
from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document
from nodes.retriver_validator_agent import retrieval_validator_agent
from nodes.task_agents import flashcard_agent, summarizer_agent, quiz_agent, studyplan_agent
from router.routers import route_next_step
from nodes.search_agent import search_node

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai", "mistral"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("retrieval_validator", retrieval_validator_agent)
workflow.add_node("search", search_node)
workflow.add_node("flashcard", flashcard_agent)
workflow.add_node("summary", summarizer_agent)
workflow.add_node("quiz", quiz_agent)
workflow.add_node("studyplan", studyplan_agent)

workflow.set_entry_point("retrieval_validator")
workflow.add_conditional_edges(
    "retrieval_validator",
    route_next_step,
    {
        "search": "search",
        "flashcard": "flashcard",
        "summary": "summary",
        "quiz": "quiz",
        "studyplan": "studyplan",
        "end": END,
        "error": END
    }
)
workflow.add_edge("search", "retrieval_validator")
workflow.add_edge("flashcard", END)
workflow.add_edge("summary", END)
workflow.add_edge("quiz", END)
workflow.add_edge("studyplan", END)
graph = workflow.compile()




def test_graph(query: str):
    state = AgentState.create_initial_state(option="summary", max_search=5)

    AgentState.add_human_message(state, query)
    updated_state = graph.invoke(state)
    print("\n\n Human: ", AgentState.get_last_human_message(updated_state))
    print("\n\nNext step:", updated_state["next_step"])
    if updated_state["search_query"]:
        print("\n\nTopics to search:", updated_state["search_query"])
    print("last ai message: ", AgentState.get_last_ai_message(updated_state))


test_graph("What is Amdahl's law? Give full definition and formula.")

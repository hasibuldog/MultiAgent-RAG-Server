from langchain_community.tools.tavily_search import TavilySearchResults
from states.states import AgentState
from langchain.schema import Document

tool = TavilySearchResults(max_results=3)

def search_node(state: AgentState) -> AgentState:
    """
    Performs search using Tavily and updates state with results
    """
    search_tool = tool
    search_query = state["search_query"]
    print("search_query:", search_query)
    search_results = search_tool.invoke(search_query)
    print('search_results type:', type(search_results))
    AgentState.add_documents(state, docs=[Document(page_content=result['content']) for result in search_results])
    # print("---------------------------------------------------------------------")
    # print("Total search :", state["total_search"],"\n")
    state["total_search"] += 1
    # print("Updated total_search:", state["total_search"])
    # print("---------------------------------------------------------------------")

    return state


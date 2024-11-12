from states.states import AgentState


def route_next_step(state: AgentState):
    """
    Routes to next node based on the next_step in state
    """
    if state["total_search"] >= state["max_search"]:
        state["next_step"] = state["option"]
    return state["next_step"]

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from states.states import AgentState
from models.llms import model
from tools.retrivers import semantic_retriever
from tools.buffermemory import create_buffer_memory


def semantic_search(query: str) -> str:
    semantic_docs = semantic_retriever.invoke(query)
    return "\n".join(doc.page_content for doc in semantic_docs)


semantic_retrival_tool = Tool(
    name="semantic_retrival_tool",
    description="Search for relevant documents using vector similarity",
    func=semantic_search,
)




def retrieval_validator_agent(state: AgentState) -> AgentState:

    validator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a validation agent that determines if the retrieved content is sufficient for the requested task.
        Your response will be very important for the next agent. 
        Retrieved content: {retrieved_docs}.
            Task option: {option}.
        Try to be biased towards answering "YES" and avoid responding with "NO" as much as possible. Because answering no will force search agent to run which costs money. 
        Your goal is to provide the user with specific feedback on what is missing or needs to be improved, so they can refine the 
        retrieval process without constantly relying on the search agent.
        Evaluate the chat_history and content carefully and respond with either:
        1. "YES" (is the retrives content is enough for you to come up with an answer)
        2. "NO" , "QUERY"= An optimized search query for the missing details
        When evaluating the content, consider:
        - Completeness of information
        - Depth of coverage for key concepts
        - Contextual relevance to the user's query
        - Sufficinet enough to not require further search
        - Alignment with the user's context and previous messages
        - Avoiding unnecessary search queries
        """,
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = create_buffer_memory(state)

    validator_agent = create_openai_functions_agent(
        llm=model,
        prompt=validator_prompt,
        tools=[semantic_retrival_tool],
    )

    validator_executor = AgentExecutor(
        agent=validator_agent,
        memory=memory,
        tools=[semantic_retrival_tool],
        verbose=True
    )
    if state["total_search"] == 0:
        try:
            semantic_docs = semantic_retriever.invoke(
                AgentState.get_last_human_message(state)
            )
            AgentState.add_documents(state, semantic_docs)
        except Exception as e:
            print(f"Retrieval error: {e}")
            AgentState.add_document(state, "")

    print(
        f"Retrived Docs length : {len(state['retrieved_docs'])}\n,Total_Searches :{state['total_search']}"
    )

    response = validator_executor.invoke(
        {
            "input": AgentState.get_last_human_message(state),
            "retrieved_docs": AgentState.get_all_documents(state, with_info=False),
            "option": state["option"],
            "agent_scratchpad": AgentState.get_agent_scratchpad(state),
        }
    )

    output = response["output"]

    AgentState.add_ai_message(state, output)

    if output.startswith("YES"):
        AgentState.set_next_step(state, step = state['option'])
    else:
        AgentState.set_next_step(state, step = "search")
        state["search_query"] = output.split("=")[-1]

    return state


# def test_agent(query: str):
#     state = AgentState.create_initial_state(option="flashcard", max_search=5)

#     AgentState.add_human_message(state, query)
#     updated_state = retrieval_validator_agent(state=state)
#     print("\n\n Human: ", AgentState.get_last_human_message(updated_state))
#     print("\n\nNext step:", updated_state["next_step"])
#     if updated_state["search_query"]:
#         print("\n\nTopics to search:", updated_state["search_query"])

#     print("Retrived Docs: \n", AgentState.get_all_documents(updated_state, with_info=False))


# test_agent("What is Amdahl's law? Give full definition and formula.")


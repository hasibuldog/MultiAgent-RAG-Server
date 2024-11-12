from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.schema import Document
from models.llms import model
from states.states import AgentState
from nodes.search_agent import tool
from tools.buffermemory import create_buffer_memory


def flashcard_agent(state: AgentState) -> AgentState:
    memory = create_buffer_memory(state)

    flashcard_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a flashcard creation agent. Generate flashcards based on the retrieved documents provided.",
            ),
            ("human", "{input}"),
            ("user", "{retrieved_docs}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    flashcard_agent_fn = create_openai_functions_agent(
        llm=model, prompt=flashcard_prompt, tools=[tool]
    )

    flashcard_executor = AgentExecutor(
        agent=flashcard_agent_fn, memory=memory, verbose=True, tools=[tool]
    )
    response = flashcard_executor.invoke(
        {
            "retrieved_docs": AgentState.get_all_documents(state, with_info=False),
            "input": AgentState.get_last_human_message(state),
            "agent_scratchpad": AgentState.get_agent_scratchpad(state),
        }
    )

    AgentState.add_ai_message(state, response["output"])
    AgentState.set_next_step(state, step="end")
    return state


def summarizer_agent(state: AgentState) -> AgentState:
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a summarization agent. Summarize the content of the retrieved documents.",
            ),
            ("human", "{input}"),
            ("user", "{retrieved_docs}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = create_buffer_memory(state)

    summarizer_agent_fn = create_openai_functions_agent(
        llm=model, prompt=summarization_prompt, tools=[tool]
    )

    summarizer_executor = AgentExecutor(
        agent=summarizer_agent_fn, memory=memory, verbose=True, tools=[tool]
    )
    response = summarizer_executor.invoke(
        {
            "retrieved_docs": AgentState.get_all_documents(state, with_info=False),
            "input": AgentState.get_last_human_message(state),
            "agent_scratchpad": AgentState.get_agent_scratchpad(state),
        }
    )
    AgentState.add_ai_message(state, response["output"])
    AgentState.set_next_step(state, step="end")
    return state


def studyplan_agent(state: AgentState) -> AgentState:
    studyplan_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study plan generator. Create a study plan based on the retrieved documents.",
            ),
            ("human", "{input}"),
            ("user", "{retrieved_docs}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = create_buffer_memory(state)

    studyplan_agent_fn = create_openai_functions_agent(
        llm=model, prompt=studyplan_prompt, tools=[tool]
    )

    study_plan_executor = AgentExecutor(
        agent=studyplan_agent_fn, memory=memory, verbose=True, tools=[tool]
    )
    response = study_plan_executor.invoke(
        {
            "retrieved_docs": AgentState.get_all_documents(state, with_info=False),
            "input": AgentState.get_last_human_message(state),
            "agent_scratchpad": AgentState.get_agent_scratchpad(state),
        }
    )
    AgentState.add_ai_message(state, response["output"])
    AgentState.set_next_step(state, step="end")
    return state


def quiz_agent(state: AgentState) -> AgentState:
    quiz_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a quiz question generation agent. Create quiz questions based on the retrieved documents.",
            ),
            ("human", "{human_query}"),
            ("user", "{retrieved_docs}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = create_buffer_memory(state)

    quiz_agent_fn = create_openai_functions_agent(
        llm=model, prompt=quiz_prompt, tools=[tool]
    )

    quiz_executor = AgentExecutor(
        agent=quiz_agent_fn, memory=memory, verbose=True, tools=[tool]
    )

    response = quiz_executor.invoke(
        {
            "retrieved_docs": AgentState.get_all_documents(state, with_info=False),
            "input": AgentState.get_last_human_message(state),
            "agent_scratchpad": AgentState.get_agent_scratchpad(state),
        }
    )
    AgentState.add_ai_message(state, response["output"])
    # AgentState.add_to_scratchpad(state, {"summary": response["output"]})
    AgentState.set_next_step(state, step="end")
    return state

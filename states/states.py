from typing import TypedDict, Annotated, List, Optional, Dict, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from pydantic.types import PositiveInt
from datetime import datetime

class AgentState(TypedDict):
    message_history: Annotated[ChatMessageHistory, "Complete conversation history"]
    retrieved_docs: Annotated[list[Document], "Retrieved documents"]
    next_step: Annotated[str, "Next step in the pipeline"]
    search_query: Annotated[list[str], "Topics that need additional search"]
    option: Annotated[str, "Task option (flashcard/summary/quiz/study_plan)"]
    agent_scratchpad: List[Dict[str, Any]]
    max_search: NotRequired[Annotated[int, "Maximum search allowed"]]
    total_search: NotRequired[Annotated[int, "Number of searches performed"]]
    created_at: NotRequired[Annotated[datetime, "State creation timestamp"]]
    last_updated: NotRequired[Annotated[datetime, "Last state update timestamp"]]

    @classmethod
    def create_initial_state(
        cls, 
        option: str = "", 
        max_search: int = 3
    ) -> 'AgentState':
        """Create a new AgentState with initial values."""
        return cls(
            message_history=ChatMessageHistory(),
            retrieved_docs=[],
            next_step="start",
            search_query=[],
            option=option,
            agent_scratchpad=[],
            max_search=max_search,
            total_search=0,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

    # Message Management Methods
    @staticmethod
    def add_message(state: 'AgentState', message: BaseMessage) -> None:
        """Add a message to the conversation history."""
        state["message_history"].add_message(message)
        state["last_updated"] = datetime.now()

    @staticmethod
    def add_human_message(state: 'AgentState', content: str) -> None:
        """Add a human message."""
        AgentState.add_message(state, HumanMessage(content=content))

    @staticmethod
    def add_ai_message(state: 'AgentState', content: str) -> None:
        """Add an AI message."""
        AgentState.add_message(state, AIMessage(content=content))

    @staticmethod
    def get_all_messages(state: 'AgentState') -> List[BaseMessage]:
        """Get all messages in chronological order."""
        return state["message_history"].messages

    @staticmethod
    def get_last_message(state: 'AgentState') -> Optional[BaseMessage]:
        """Get the most recent message."""
        messages = AgentState.get_all_messages(state)
        return messages[-1] if messages else None
    
    @staticmethod
    def get_agent_scratchpad(state: 'AgentState') -> Optional[BaseMessage]:
        """Get the most recent message."""
        return state["agent_scratchpad"]

    @staticmethod
    def get_last_human_message(state: 'AgentState') -> Optional[str]:
        """Get content of the most recent human message."""
        for message in reversed(AgentState.get_all_messages(state)):
            if isinstance(message, HumanMessage):
                return message.content
        return None

    @staticmethod
    def get_last_ai_message(state: 'AgentState') -> Optional[str]:
        """Get content of the most recent AI message."""
        for message in reversed(AgentState.get_all_messages(state)):
            if isinstance(message, AIMessage):
                return message.content
        return None

    @staticmethod
    def get_all_human_messages(state: 'AgentState') -> List[str]:
        """Get contents of all human messages."""
        return [msg.content for msg in AgentState.get_all_messages(state) 
                if isinstance(msg, HumanMessage)]

    @staticmethod
    def get_all_ai_messages(state: 'AgentState') -> List[str]:
        """Get contents of all AI messages."""
        return [msg.content for msg in AgentState.get_all_messages(state) 
                if isinstance(msg, AIMessage)]

    @staticmethod
    def clear_messages(state: 'AgentState') -> None:
        """Clear all messages from history."""
        state["message_history"].clear()
        state["last_updated"] = datetime.now()


    @staticmethod
    def add_document(state: 'AgentState', doc: Document) -> None:
        """Add a retrieved document."""
        state["retrieved_docs"].append(doc)
        state["last_updated"] = datetime.now()

    @staticmethod
    def add_documents(state: 'AgentState', docs: List[Document]) -> None:
        """Add multiple retrieved documents."""
        state["retrieved_docs"].extend(docs)
        state["last_updated"] = datetime.now()

    @staticmethod
    def get_all_documents(state: 'AgentState', with_info:bool) -> List[Document]:
        """Get all retrieved documents."""
        if with_info:
            return state["retrieved_docs"]
        return [doc.page_content for doc in state["retrieved_docs"]]

    @staticmethod
    def clear_documents(state: 'AgentState') -> None:
        """Clear all retrieved documents."""
        state["retrieved_docs"].clear()
        state["last_updated"] = datetime.now()

    @staticmethod
    def set_next_step(state: 'AgentState', step: str) -> None:
        """Update the next step in the pipeline."""
        state["next_step"] = step
        state["last_updated"] = datetime.now()

    @staticmethod
    def update_option(state: 'AgentState', option: str) -> None:
        """Update the task option."""
        state["option"] = option
        state["last_updated"] = datetime.now()

    @staticmethod
    def add_to_scratchpad(state: 'AgentState', data: Dict[str, Any]) -> None:
        """Add data to agent scratchpad."""
        state["agent_scratchpad"].append(data)
        state["last_updated"] = datetime.now()

    @staticmethod
    def clear_scratchpad(state: 'AgentState') -> None:
        """Clear the agent scratchpad."""
        state["agent_scratchpad"].clear()
        state["last_updated"] = datetime.now()

    @staticmethod
    def to_dict(state: 'AgentState') -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            "messages": AgentState.get_all_messages(state),
            "option": state["option"],
            "agent_scratchpad": state["agent_scratchpad"],
            "total_search": state["total_search"],
            "max_search": state["max_search"],
            "retrieved_docs": state["retrieved_docs"],
            "next_step": state["next_step"],
            "search_query": state["search_query"],
            "created_at": state["created_at"],
            "last_updated": state["last_updated"]
        }

    @staticmethod
    def get_conversation_length(state: 'AgentState') -> int:
        """Get total number of messages."""
        return len(AgentState.get_all_messages(state))

    @staticmethod
    def get_conversation_summary(state: 'AgentState') -> Dict[str, int]:
        """Get summary of conversation statistics."""
        messages = AgentState.get_all_messages(state)
        return {
            "total_messages": len(messages),
            "human_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
            "total_searches": state["total_search"],
            "remaining_searches": state["max_search"] - state["total_search"],
            "documents_retrieved": len(state["retrieved_docs"])
        }

def example_usage():
    state = AgentState.create_initial_state(option="flashcard", max_search=5)
    
    state.add_human_message("Can you help me learn about Python?")
    state.add_ai_message("I'd be happy to help! What specific topics interest you?")
    state.add_search_query("Python programming basics")
    if state.can_search():
        state.increment_search_count()

    summary = state.get_conversation_summary()
    state.set_next_step("generate_flashcards")
    
    return state

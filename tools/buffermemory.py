from langchain.memory import ConversationBufferMemory
from states.states import AgentState


def create_buffer_memory(state: AgentState):
    return ConversationBufferMemory(
        chat_memory=state["message_history"],
        return_messages=True,
        input_key="input",  # Add this line
        memory_key="agent_scratchpad",  # Add this line
    )

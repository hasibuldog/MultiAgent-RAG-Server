from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_postgres import PGVector
from langchain.tools import Tool
from typing import Dict, Any
from states.states import AgentState
from models.llms import model, azure_embeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

driver = "psycopg"
host = os.getenv("POSTGRES_HOST")
port = int(os.getenv("POSTGRES_PORT"))
database = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=driver, host=host, port=port, database=database, user=user, password=password
)

vectorstore = PGVector(
    connection=CONNECTION_STRING,
    embeddings=azure_embeddings,
    collection_name="langchain",
    pre_delete_collection=False,
)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

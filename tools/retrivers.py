from langchain_postgres import PGVector
from models.llms import azure_embeddings
from dotenv import load_dotenv
import os

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

semantic_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)


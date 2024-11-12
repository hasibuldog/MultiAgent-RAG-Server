from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

os.environ["TAVILY_API_KEY"] = os.getenv("tavily_api_key")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    temperature=0.1,
    max_tokens=100,
    timeout=None,
    max_retries=2,
)


azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("EMBEDDING_API_VERSION"),
)

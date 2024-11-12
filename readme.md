# Multiagent RAG: A Study-Focused Knowledge Assistant

This repository provides a multiagent RAG implementation using the LangChain and LangGraph frameworks. The system is designed to be a study-focused knowledge assistant, guiding users through various study-related tasks and providing relevant information.

## Setup

1. **Database Preparation**:

   - Set up Postgres with the `pgvector` and `pg_bestmatch` extensions. Refer to the detailed instructions in [this](https://github.com/hasibuldog/postgres_with_pgvector_-_pg_bestmatch) repository.
2. **Environment Setup**:

   - Clone the repository: `https://github.com/hasibuldog/MultiAgent-RAG-Server.git`
   - Change to the repository directory: `cd MultiAgent-RAG-Server`

**Do either**:
- Create a conda environment with packages: `conda env create -f environment.yaml`
- Activate the environment: `conda activate rag`

**Or**:
- Create new env `conda create -n <env_name> python=3.10`
- Install the required dependencies: `pip install -r requirements.txt`
3. **Configuration**:

   - Create a `.env` file in the project root directory with the following configuration:
   ```
     tavily_api_key = your tavily api key
     AZURE_OPENAI_ENDPOINT = your openai api endpoint
     OPENAI_API_VERSION = your openai api version
     AZURE_OPENAI_API_KEY = your openai api key
     AZURE_EMBEDDING_DEPLOYMENT = your openai api deployment(don't need if use direct openai api)
     EMBEDDING_API_VERSION = your embedding api version
     POSTGRES_DB=your_db_name
     POSTGRES_USER=your_user_name
     POSTGRES_PASSWORD=your_password
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432 (or specify the port you binded)
     ```

This repository provides a multiagent RAG (Retrieval Augmented Generation) implementation using the LangChain and LangGraph frameworks. The system is designed to be a study-focused knowledge assistant, guiding users through various study-related tasks and providing relevant information.

## System Architecture
The system consists of several specialized agents working together:

1. **Retrieval Validator**: 
    - Validator for the vectorstore retrived docs (Determines if further internet search nessecery based on the retrived docs)
2. **Search Agent**: 
    - Handles information retrieval and search operations (handles search based on the query by Retrieval Validator)
3. **Flashcard Agent**: 
    - Creates and manages study flashcards 
4. **Quiz Agent**: 
    - Generates and conducts quizzes
5. **Study Plan Agent**: 
    - Develops personalized study plans
6. **Summary Agent**: 
    - Creates concise summaries of study materials

[Agent Graph](output.jpeg?raw=true "Agent Graph")


## Usage

The multiagent RAG system provides various functionalities to assist users in their study-related tasks. You can interact with the system using the provided interface or through a command-line prompt.

## Limitations and Future Improvements

This implementation is not a production-grade solution but rather a learning material. It will be improved and updated over time. We welcome contributions and ideas from the community to enhance the system further.


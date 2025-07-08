import chromadb
from chromadb.utils import embedding_functions
import json
import os

# Load both types of examples
with open("sql_query_examples_generic.json", encoding="utf-8") as f:
    generic_examples = json.load(f)

with open("sql_query_examples_usecase.json", encoding="utf-8") as f:
    usecase_examples = json.load(f)

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')
CHROMA_DB_PATH = os.environ.get('Chroma_Query_Examples')

# Initialize embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_EMBEDDING_DEPLOYMENT_NAME
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def prepare_ingest(items):
    inputs = [item['input'] for item in items]
    queries = [item['query'] for item in items]
    return inputs, queries

def ingest_examples(examples, collection_name):
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    inputs, queries = prepare_ingest(examples)
    ids = [f"{collection_name}_pair_{i}" for i in range(len(inputs))]
    metadatas = [{"query": query} for query in queries]
    
    collection.add(
        ids=ids,
        documents=inputs,
        metadatas=metadatas
    )
    return collection

# Ingest both types of examples
generic_collection = ingest_examples(generic_examples, "generic_examples")
usecase_collection = ingest_examples(usecase_examples, "usecase_examples")
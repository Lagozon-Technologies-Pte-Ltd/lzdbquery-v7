from openai import OpenAI
import json
import os 
import openai
from openai import AzureOpenAI
from IngestExamples import example_manager
import logging



AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')
AZURE_EMBEDDING_DEPLOYMENT_NAME= os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')


openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT  
openai.api_version = AZURE_OPENAI_API_VERSION  
AZURE_EMBEDDING_DEPLOYMENT = AZURE_EMBEDDING_DEPLOYMENT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def embed_query(text):
    response = client.embeddings.create(
        input=[text],
        model=os.environ['AZURE_EMBEDDING_DEPLOYMENT_NAME'],
     
        
    )
    return response.data[0].embedding


def get_examples(query: str, question_type: str):
    if question_type not in ["generic", "usecase"]:
        raise ValueError("question_type must be either 'generic' or 'usecase'")
    
    collection = example_manager.get_collection(question_type)
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    return [
        {"input": doc, "query": meta}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]


# print(get_examples("Show all customer verbatim entries for a specific RO RO25A007880"))
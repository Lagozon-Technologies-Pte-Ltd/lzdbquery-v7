import chromadb
from chromadb.utils import embedding_functions
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExampleManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=os.environ.get('Chroma_Query_Examples'))
        self.embedding_function = self._get_embedding_function()
        self.collections = {}
        self._initialize_all_collections()

    def _get_embedding_function(self):
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            api_base=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_type="azure",
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01"),
            model_name=os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME')
        )

    def _initialize_all_collections(self):
        # Initialize collections without deleting first
        self._initialize_collection("generic", "sql_query_examples_generic.json")
        self._initialize_collection("usecase", "sql_query_examples_usecase.json")

    def _initialize_collection(self, question_type, file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                examples = json.load(f)
        except FileNotFoundError:
            logger.error(f"Example file not found for {question_type}: {file_path}")
            raise

        # Get or create collection
        collection_name = f"examples_{question_type}"
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except ValueError:
            logger.info(f"No existing collection to delete: {collection_name}")

        # Create new collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        # Prepare and add examples
        inputs = [item['input'] for item in examples]
        queries = [item['query'] for item in examples]
        
        collection.add(
            ids=[f"{question_type}ex{i}" for i in range(len(inputs))],
            documents=inputs,
            metadatas=[{"query": q} for q in queries]
        )
        
        self.collections[question_type] = collection
        logger.info(f"Initialized {question_type} collection with {len(inputs)} examples")

    def get_collection(self, question_type):
        if question_type not in self.collections:
            raise ValueError(f"Invalid question type: {question_type}")
        return self.collections[question_type]

# Initialize manager
try:
    example_manager = ExampleManager()
except Exception as e:
    logger.error(f"Failed to initialize ExampleManager: {str(e)}")
    raise
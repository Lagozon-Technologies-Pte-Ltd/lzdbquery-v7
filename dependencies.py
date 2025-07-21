from fastapi import Request

def get_llm(request: Request):
    return request.app.state.azure_openai_client

def get_embeddings(request: Request):
    return request.app.state.schema_collection


def get_bq_client(request: Request):
    """Dependency to provide the BigQuery client"""
    return request.app.state.bq_client
# def get_redis(request: Request):
#     return request.app.state.redis_client

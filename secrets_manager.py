import os
from google.cloud import secretmanager
from typing import Optional

def get_secret(secret_id: str, project_id: str = "972862630305") -> str:
    """
    Retrieve a secret from Google Cloud Secret Manager with local development fallback.
    
    Args:
        secret_id: The ID of the secret to retrieve
        project_id: Google Cloud project ID (defaults to your project)
        
    Returns:
        The secret value as a string
        
    Raises:
        Exception: If secret retrieval fails and no local fallback exists
    """
    # First try local environment variables for development
    env_val = os.getenv(secret_id)
    if env_val:
        return env_val
    
    try:
        # Initialize the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()
        
        # Build the resource name of the secret version
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        
        # Access the secret version
        response = client.access_secret_version(name=name)
        
        # Return the decoded payload
        return response.payload.data.decode('UTF-8')
    
    except Exception as e:
        # Try one more fallback - common .env style naming
        env_fallback = os.getenv(secret_id.upper().replace('-', '_'))
        if env_fallback:
            return env_fallback
            
        raise Exception(f"Failed to access secret {secret_id}: {str(e)}. "
                      f"Please set {secret_id} as environment variable for local development.")


# Example usage:
# DATABASE_URL = get_secret("DATABASE_URL")
# AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
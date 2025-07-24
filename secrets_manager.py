import  json
from google.cloud import secretmanager
from typing import Optional

def get_secret(project_id, secret_id, version_id="latest"):
    """
    Retrieves a secret from Secret Manager, parses it as JSON,
    and returns a dictionary of variables.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    try:
        response = client.access_secret_version(request={"name": name})
        secret_payload = response.payload.data.decode("UTF-8")
        variables = json.loads(secret_payload)
        return variables
    except Exception as e:
        print(f"Error retrieving or parsing secret {secret_id}: {e}")
        return {}  # Return an empty dictionary in case of error


# Example usage:
# DATABASE_URL = secret_variables.get("DATABASE_URL")
# AZURE_OPENAI_API_KEY = secret_variables.get("AZURE_OPENAI_API_KEY")

# Example Usage:
project_id = "972862630305"
secret_id = "lz-dbquery-secret"

secret_variables = get_secret(project_id, secret_id)

# if secret_variables:
#     # Access individual variables from the dictionary
#     db_host = secret_variables.get("DB_HOST")
#     db_user = secret_variables.get("DB_USER")
#     db_password = secret_variables.get("DB_PASSWORD")
#     db_port = secret_variables.get("DB_PORT")

#     print(f"DB Host: {db_host}")
#     print(f"DB User: {db_user}")
#     print(f"DB Port: {db_port}")
#     # Be extremely careful about logging passwords!
#     # print(f"DB Password: {db_password}")
# else:
#     print("Failed to retrieve secret variables.")
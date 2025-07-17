from google.cloud import bigquery
from google.oauth2 import service_account
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GCP credentials dictionary (as you've defined it)
credentials_info = {
    "type": os.getenv('GOOGLE_CREDENTIALS_TYPE'),
    "project_id": os.getenv('GOOGLE_CREDENTIALS_PROJECT_ID'),
    "private_key_id": os.getenv('GOOGLE_CREDENTIALS_PRIVATE_KEY_ID'),
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCraZCE2H2pE6uE\n7rgU6pKFpGilWEloN+NwUQOzHhTE8ehenKJ0lwqc8MpnTwseT861Qj80TojR1lfu\nLZP2ZlefuEUaZ48lncs/8vEzpntVGm9vazSU2ytG/o3yHpMUb/E1UDVKwVN1G60K\nHLvFJIvmB0v6IgWwvSzYtkjaIHX+Ny1BT1Ag2baKNEGtytU+Ph56CK4mxtAMpnFg\njZ0g+KYRLEDPLEJPoryhMEJXA1Dlf9vp8b8EVh3MZfoVmaA5wRYntDsQkukAIIpQ\nkII4V7GJaBnhpaNuyh57sj4HrpKKL9ZNULnNyjIWQhTyxxFRMFOpVDH1du7YgRPq\njyPPH1xvAgMBAAECggEACoGEN/8QAuhB8MNYltuZbiEQUuO+4TJLJ0c6K5vJBj1w\nkn/xCxObIrRaAlUbZ2siF3KtEy24NuqJnLYuARQ4TRPdb9TLNsNdnRi2BOHCz9Ld\nLOdn2kU2nedlfIcl6wv39jnpW2nO+1RL/kqaH+c6mm4sxk5PYR5Bbw4LYTBL/6bm\n2hPqyRB7cBfEXOc5+/vwvLD7zd3uHFYwSbmDwJMFS+rv8V2xiEhe+EDFBjbqXSpb\ncltjhKQgrMGteSJeSfei3fAq+K9rDu3akyAVt+gYP408AjEme/zOi7tqyGOBqPN1\nwnVzoj0GD8Nm4OAaRqJkftj9pev8XfD9yc48eJRMZQKBgQDTlmLvFxO+Mml3gN8U\n4Yw11zjczScTbuY5uqhWD+BlzFfSk1tdbAx5oQjJOc5WQXR7adgZ6A7pKs/QpQMA\n2ioCY0vOakhqS46SehhgoRj0Yj+6qZF9h438+XQlCCUxIBud7J0C7e/hZLNyvzeQ\ni4VtcMpgRAEK+vz10XyYYJETVQKBgQDPZGCFqthHhp4yr530bWElPxC9H+GJmqQH\n0VZeMQQ1+bXs3VQOv8jXFZpn/BVhhKfddjNDqVdCfJu7oanWmC9TnjGHdV6gxWIK\nfplmhKIoBerwZJFLPLj/wV4Sfdvf6Zv5sDQ5ow9jzd2oUNL49OV06W3N65ug8RBO\nMUoMFCN4swKBgQCBDouN1f+e1VTrJVnsfJ5vALWYSDH7cntO3wFqbQisTvWKZYMm\n+o6paYXYZz/p8MbBuA+tzZO6uPhFBUFNtcRF7JcCcmV1IFz4DyzrU5fLCFpi2qb5\ncEM0+FrVc6Br1G/D5dznOoZEbo3eAbA8pD1gQZnPGeug7PJ6ZaqfrtcOeQKBgDWa\ngSQrW0lpbvw0zgO+Payt1zq6wcWaNalbnxIrYyY8S5xUPISvZ07IY6dazX/uFKE2\nCtwDKe2iXXIqv8YagakAK1cSrAmr2sJRpH6N64eit+24YKFsqXhZV2I6K5l9PPZV\nZ7o5/iFStWbqtQzp52DHcL0Xl5sKk6dSMAxdLCnnAoGBAM3FswYDNsPd4kwflVNO\nL2DiOW94Dpqoc+Fo1gP0ifE/wpr7So08G6fcq2/tIvHacHGFHAll4OaAa3jC/DSK\nx6S+F6GqCOhjdc4oVfqthYOanW6WHIpCILSwVMy+HL33ijGwSElAzN/mbnCnP3HC\nBo54Ew2hgqlN8xwtbjUFMbYQ\n-----END PRIVATE KEY-----",
    "client_email": os.getenv('GOOGLE_CREDENTIALS_CLIENT_EMAIL'),
    "client_id": os.getenv('GOOGLE_CREDENTIALS_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/lz-mahindra-service-account%40gen-ai-team-mahindra.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Create credentials
credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/bigquery"]
)

# Set default dataset
client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
    default_query_job_config=bigquery.QueryJobConfig(
        default_dataset="gen-ai-team-mahindra.lz_mahindra_dataset"
    )
)

# ✅ Now you can use short table name
query = """


SELECT 
  *
FROM 
  MH_RO_HDR_DETAILS
  limit 10;
"""

# Execute the query
print("Running query...")
results = client.query(query).result()

# Print results
print("\nQuery results:")
for row in results:
    print(dict(row))

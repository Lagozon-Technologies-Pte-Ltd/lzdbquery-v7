import json
import datetime
import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read Azure SQL credentials from .env
AZURE_SQL_SERVER = os.getenv("SQL_DB_SERVER")
AZURE_SQL_PORT = os.getenv("SQL_DB_PORT", "1433")
AZURE_SQL_DATABASE = os.getenv("SQL_DB_NAME")
AZURE_SQL_USERNAME = os.getenv("SQL_DB_USER")
AZURE_SQL_PASSWORD = os.getenv("SQL_DB_PASSWORD")
AZURE_SQL_DRIVER = os.getenv("SQL_DB_DRIVER", "ODBC Driver 18 for SQL Server")

# Connection string
conn_str = (
    f"DRIVER={{{AZURE_SQL_DRIVER}}};"
    f"SERVER={AZURE_SQL_SERVER},{AZURE_SQL_PORT};"
    f"DATABASE={AZURE_SQL_DATABASE};"
    f"UID={AZURE_SQL_USERNAME};"
    f"PWD={AZURE_SQL_PASSWORD};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=no;"
    f"Connection Timeout=30;"
)

# Establish connection
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Load metadata JSON
with open("table_files\\expanded_columns.json", "r") as f:
    metadata = json.load(f)

# Helper to format numbers
def format_number(x):
    if isinstance(x, int):
        return f"{x:d}"
    elif isinstance(x, float) and x.is_integer():
        return f"{int(x):d}"
    else:
        return f"{x:.1f}"

# Recursively convert date/datetime to strings
def convert_dates(obj):
    if isinstance(obj, dict):
        return {k: convert_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates(item) for item in obj]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    else:
        return obj

# Regenerate examples
def regenerate_examples(column_info, limit=5):
    try:
        table = column_info["metadata"]["table_name"]
        column = column_info["column_name"].split(".")[-1]
        data_type = column_info["metadata"].get("data_type", "").upper()

        query = f"""
        SELECT DISTINCT TOP {limit} [{column}]
        FROM [{table}]
        WHERE [{column}] IS NOT NULL
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        values = []

        for row in rows:
            value = row[0]
            if value is None:
                continue

            if data_type == "INTEGER":
                value = int(value)
            elif data_type == "FLOAT":
                value = float(value)
            elif data_type == "STRING":
                value = str(value)
            elif data_type in ("DATE", "DATETIME", "TIMESTAMP") and isinstance(value, (datetime.date, datetime.datetime)):
                value = value.isoformat()

            values.append(value)

        column_info["examples"] = values

    except Exception as e:
        print(f"❌ Error processing {column_info.get('column_name')}: {e}")
        column_info["examples"] = []

# Regenerate examples for each column in metadata
for col in metadata:
    regenerate_examples(col)

# Convert any date objects
metadata = convert_dates(metadata)

# Save updated metadata
with open("metadata_updated.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ metadata_updated.json generated with refreshed examples from Azure SQL.")

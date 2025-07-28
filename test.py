from sqlalchemy import text
from database import engine  # Ensure this correctly imports your SQLAlchemy engine

def run_query(query: str):
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query))
            rows = result.fetchall()

            if rows:
                # Print column headers
                headers = result.keys()
                print(" | ".join(headers))
                print("-" * (len(headers) * 15))

                # Print each row
                for row in rows:
                    print(" | ".join(str(col) for col in row))
            else:
                print("Query executed successfully. No rows returned.")

        except Exception as e:
            print(f"Error: {e}")

# Example usage:
if _name_ == "_main_":
    user_query = user_query = """
SELECT TOP 10 RP.[PART_CODE], RP.[PART_DESC], ROUND(SUM(TRY_CAST(RP.[PART_QUANTITY] AS FLOAT)), 1) AS [TOTAL_COUNT], ROUND(SUM(TRY_CAST(RP.[PARTAMOUNT] AS FLOAT)), 1) AS [TOTAL_VALUE] FROM [MH_RO_HDR_DETAILS] ROHDR INNER JOIN [MH_RO_PARTS] RP ON ROHDR.[SV_RO_BILL_HDR_SK] = RP.[SV_RO_BILL_HDR_SK] WHERE ROHDR.[RO_BILL_DATE] BETWEEN '2024-01-01' AND '2024-12-31' AND ((ROHDR.[SERV_CATGRY_DESC] COLLATE SQL_Latin1_General_CP1_CI_AS = 'En-Route' AND ROHDR.[SERVICE_GROUP] COLLATE SQL_Latin1_General_CP1_CI_AS = 'Pre-Sale/PDI')) AND RP.[PART_CATEGORY_GROUP] COLLATE SQL_Latin1_General_CP1_CI_AS = 'Spares' AND RP.[PART_DESC] COLLATE SQL_Latin1_General_CP1_CI_AS NOT LIKE '%filter%' AND RP.[OEM_PART_IND] COLLATE SQL_Latin1_General_CP1_CI_AS = 'N' GROUP BY RP.[PART_CODE], RP.[PART_DESC] ORDER BY [TOTAL_COUNT] DESC"""

    run_query(user_query)
# models.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class ChartRequest(BaseModel):
    x_axis: str
    y_axis: str
    chart_type: str
    table_data: List[Dict]
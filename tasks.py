from celery_app import celery_app
import pandas as pd
from models import ChartRequest  # Import your models
from typing import Dict
import time

@celery_app.task(bind=True, name='process_data_task')
def process_data_task(self, data: Dict):
    """Example background task that processes data"""
    try:
        # Simulate processing
        time.sleep(2)
        
        processed_data = {k: str(v).upper() for k, v in data.items()}
        
        return {
            'status': 'SUCCESS',
            'result': processed_data,
            'task_id': self.request.id
        }
    except Exception as e:
        return {
            'status': 'FAILURE',
            'error': str(e),
            'task_id': self.request.id
        }

@celery_app.task(bind=True, name='generate_chart_task')
def generate_chart_task(self, chart_request: Dict):
    """Background task for chart generation"""
    try:
        # Convert dict to ChartRequest if needed
        request = ChartRequest(**chart_request)
        
        # Your existing chart generation logic
        data_df = pd.DataFrame(request.table_data)
        
        if request.x_axis not in data_df.columns:
            raise ValueError(f"Column '{request.x_axis}' not found")
            
        if request.chart_type != "Word Cloud" and request.y_axis not in data_df.columns:
            raise ValueError(f"Column '{request.y_axis}' not found")
            
        fig = generate_chart_figure(
            data_df, 
            request.x_axis, 
            request.y_axis, 
            request.chart_type
        )
        
        if not fig:
            raise ValueError("Unsupported chart type")
            
        return {
            'status': 'SUCCESS',
            'result': fig.to_json(),
            'task_id': self.request.id
        }
        
    except Exception as e:
        return {
            'status': 'FAILURE',
            'error': str(e),
            'task_id': self.request.id
        }
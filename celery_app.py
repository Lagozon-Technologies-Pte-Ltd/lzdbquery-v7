from celery import Celery
import os
import ssl
from dotenv import load_dotenv

load_dotenv()

# Azure Redis configuration
redis_host = os.getenv('REDIS_HOST')
redis_key = os.getenv('REDIS_KEY')
redis_port = os.getenv('REDIS_PORT', '6380')

# SSL Context for Azure Redis
ssl_context = ssl.SSLContext()
ssl_context.verify_mode = ssl.CERT_NONE  # Azure Redis uses self-signed certs

celery_broker_url = f"rediss://:{redis_key}@{redis_host}:{redis_port}/0?ssl_cert_reqs=CERT_NONE"
celery_result_backend = f"rediss://:{redis_key}@{redis_host}:{redis_port}/1?ssl_cert_reqs=CERT_NONE"

celery_app = Celery(
    'worker',
    broker=celery_broker_url,
    backend=celery_result_backend,
    include=['tasks']
)

# SSL Configuration for Celery
celery_app.conf.broker_use_ssl = {
    'ssl_cert_reqs': ssl.CERT_NONE,
    'ssl_ca_certs': None,
    'ssl_certfile': None,
    'ssl_keyfile': None
}

celery_app.conf.result_backend_transport_options = {
    'ssl_cert_reqs': ssl.CERT_NONE,
    'retry_policy': {
        'timeout': 5.0
    }
}

# Common configuration remains the same
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)
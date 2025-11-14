import multiprocessing
import os

# Gunicorn configuration for Render deployment
# Tune for performance and memory efficiency

# Workers
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Memory optimization
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning
preload_app = True

# Bind to the port that Render expects
bind = '0.0.0.0:' + str(os.environ.get('PORT', 10000))

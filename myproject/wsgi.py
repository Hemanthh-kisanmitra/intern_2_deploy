"""
WSGI config for myproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

# Set the correct settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings_render')

# Memory optimization
try:
    import resource
    # Set memory limit to 512MB
    MEMORY_LIMIT = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
except Exception as e:
    print(f"Could not set memory limit: {e}")

application = get_wsgi_application()
"""
WSGI config for whatsapp_analyzer project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ['DJANGO_SETTINGS_MODULE'] = 'myproject.settings_render'

# Memory optimization
try:
    import resource
    # Set memory limit to 512MB
    MEMORY_LIMIT = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
except Exception as e:
    print(f"Could not set memory limit: {e}")

application = get_wsgi_application()
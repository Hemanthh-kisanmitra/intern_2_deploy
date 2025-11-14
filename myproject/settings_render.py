"""
Django settings for Render deployment.
"""

import os
from .settings import *

# Try to import dj_database_url
try:
    import dj_database_url
    HAS_DJ_DATABASE_URL = True
except ImportError:
    HAS_DJ_DATABASE_URL = False
    dj_database_url = None

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Allow all hosts - configure properly for production
ALLOWED_HOSTS = ['*']

# Database
# Use PostgreSQL in production
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Try to use PostgreSQL if DATABASE_URL is provided
if os.environ.get('DATABASE_URL') and HAS_DJ_DATABASE_URL and dj_database_url:
    DATABASES['default'] = dj_database_url.parse(os.environ.get('DATABASE_URL'))

# Static files (CSS, JavaScript, Images)
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Ensure the media root is set
MEDIA_ROOT = BASE_DIR / 'media'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}
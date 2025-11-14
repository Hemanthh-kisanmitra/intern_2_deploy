"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

import logging
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
from django.http import HttpResponse
from chatapp.views import health_check

logger = logging.getLogger(__name__)

# Debug request handler
def debug_request(request):
    logger.error(f"""
REQUEST DATA:
Method: {request.method}
Path: {request.path}
Headers: {dict(request.headers)}
GET: {dict(request.GET)}
POST: {dict(request.POST)}
    """)
    return HttpResponse(b"Debug mode enabled")

urlpatterns = [
    path('', include('chatapp.urls')),  # Include chatapp URLs at root
    path('debug/', debug_request),  # Debug endpoint
    path('admin/', admin.site.urls),
    path('health/', health_check, name='health_check'),
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico', permanent=True)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
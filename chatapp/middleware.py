from django.http import HttpResponseBadRequest
from django.utils.deprecation import MiddlewareMixin
import logging

logger = logging.getLogger(__name__)

class BadRequestLoggingMiddleware(MiddlewareMixin):
    """
    Middleware to log 400 Bad Request errors for debugging purposes
    """
    
    def process_response(self, request, response):
        if response.status_code == 400:
            logger.warning(
                f"400 Bad Request: {request.method} {request.get_full_path()} "
                f"User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')} "
                f"Referer: {request.META.get('HTTP_REFERER', 'None')} "
                f"Host: {request.META.get('HTTP_HOST', 'Unknown')} "
                f"Content-Type: {request.META.get('CONTENT_TYPE', 'None')}"
            )
        return response
    
    def process_request(self, request):
        # Log all requests for debugging
        logger.info(
            f"Request: {request.method} {request.get_full_path()} "
            f"Host: {request.META.get('HTTP_HOST', 'Unknown')} "
            f"User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')[:50]}"
        )
        return None
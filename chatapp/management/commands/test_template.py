from django.core.management.base import BaseCommand
from django.template.loader import render_to_string
from django.conf import settings

class Command(BaseCommand):
    help = 'Test if the home template can be loaded correctly'

    def handle(self, *args, **options):
        self.stdout.write("Testing template loading...")
        try:
            # Try to render the home template
            rendered = render_to_string('chatapp/home.html')
            self.stdout.write(f"Template loaded successfully! Content length: {len(rendered)} characters")
        except Exception as e:
            self.stdout.write(f"Error loading template: {str(e)}")
            import traceback
            self.stdout.write(f"Traceback: {traceback.format_exc()}")
        
        # Check template directories
        self.stdout.write(f"Template directories: {settings.TEMPLATES[0]['DIRS']}")
        self.stdout.write(f"App directories enabled: {settings.TEMPLATES[0]['APP_DIRS']}")
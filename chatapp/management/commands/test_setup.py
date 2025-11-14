from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Test Django setup and configuration'

    def handle(self, *args, **options):
        self.stdout.write("Testing Django setup...")
        self.stdout.write(f"Django settings module: {settings.SETTINGS_MODULE}")
        self.stdout.write(f"Debug mode: {settings.DEBUG}")
        self.stdout.write(f"Allowed hosts: {settings.ALLOWED_HOSTS}")
        self.stdout.write(f"Installed apps: {settings.INSTALLED_APPS}")
        self.stdout.write("Django setup test completed successfully!")
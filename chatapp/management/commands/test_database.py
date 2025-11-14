from django.core.management.base import BaseCommand
from chatapp.models import ChatFile

class Command(BaseCommand):
    help = 'Test database access for ChatFile model'

    def handle(self, *args, **options):
        self.stdout.write("Testing database access...")
        try:
            # Try to count ChatFile objects
            count = ChatFile.objects.count()
            self.stdout.write(f"Database access successful! Found {count} chat files.")
            
            # Try to create a test object
            test_file = ChatFile(
                original_filename="test.txt",
                group_name="Test Group"
            )
            self.stdout.write("Test object created successfully!")
            
        except Exception as e:
            self.stdout.write(f"Error accessing database: {str(e)}")
            import traceback
            self.stdout.write(f"Traceback: {traceback.format_exc()}")
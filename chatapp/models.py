from django.db import models

class ChatFile(models.Model):
    file = models.FileField(upload_to='chat_files/')
    original_filename = models.CharField(max_length=255)
    group_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.group_name

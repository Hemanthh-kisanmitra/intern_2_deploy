# chatapp/admin.py
from django.contrib import admin
from .models import ChatFile

@admin.register(ChatFile)
class ChatFileAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'group_name', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('original_filename', 'group_name')
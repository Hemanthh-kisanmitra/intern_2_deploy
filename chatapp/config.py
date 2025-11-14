import os
from dotenv import load_dotenv
from django.conf import settings
load_dotenv()

GEMINI_API_KEY = settings.GEMINI_API_KEY


MODEL_NAME = "deepseek-r1-distill-llama-70b"

MAX_CHARS_FOR_ANALYSIS = 30000
SENTIMENT_THRESHOLD = 0.1
TOPIC_MIN_WORD_LENGTH = 3
TOPIC_MAX_TOPICS = 15
import os
import re
import google.generativeai as genai
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse     #render the page and return in json formate
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from dotenv import load_dotenv
import os
from django.conf import settings
import logging
from .business_metrics import calculate_business_metrics
from .group_event import analyze_group_events, get_event_counts, get_event_details, get_top_removers
from .sentiment_analyzer import analyze_sentiment
from .summary_generator import (
    generate_total_summary,
    generate_user_messages,
    get_users_in_messages,
    generate_user_messages_for_user,
    generate_weekly_summary
)
from .topic_analyzer import extract_topics

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY)

# Initialize the model with fallback
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    logger.warning(f"Could not initialize gemini-2.0-flash, falling back to gemini-1.5-pro: {e}")
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e2:
        logger.warning(f"Could not initialize gemini-1.5-pro, falling back to gemini-pro-latest: {e2}")
        model = genai.GenerativeModel('gemini-pro-latest')

def generate_with_gemini(prompt):
    """Generate content using Google Gemini AI SDK"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Check if it's a quota exceeded error
        if "429" in str(e) or "quota" in str(e).lower():
            return "Content unavailable due to API quota limits. Please try again later."
        return "Content unavailable due to technical issues. Please try again later."

chat_data = {}

CHATS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'media', 'chat_files')

def parse_timestamp(timestamp_str):
    timestamp_str = timestamp_str.replace('\u202F', ' ')
    formats = [
        '%m/%d/%y, %I:%M %p',
        '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %H:%M',
        '%m/%d/%Y, %H:%M',
        '%Y-%m-%d, %H:%M',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            pass
    return None


def parse_whatsapp(file_path):
    messages = []
    current_message = None
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}\u202F[AP]M) - (.*?): (.*)',
        r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M) - (.*?): (.*)',
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?(?: [AP]M)?)\] (.*?): (.*)',
        r'(\d{4}-\d{1,2}-\d{1,2}, \d{1,2}:\d{2}) - (.*?): (.*)',
        r'(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}) - (.*?): (.*)',
        r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M) - (.*?): (.*)'
    ]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if current_message:
                        messages.append(current_message)

                    timestamp, sender, message = match.groups()
                    current_message = {
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message
                    }
                    matched = True
                    break
            if not matched and current_message:
                if current_message['message']:
                    current_message['message'] += '\n' + line
                else:
                    current_message['message'] = line
    if current_message:
        messages.append(current_message)
    return messages

def get_group_name_from_file(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def load_all_chats():
    global chat_data
    chat_data = {}
    # Load all .txt files from the CHATS_FOLDER
    if not os.path.exists(CHATS_FOLDER):
        os.makedirs(CHATS_FOLDER)

    for filename in os.listdir(CHATS_FOLDER):
        if filename.endswith('.txt'):
            file_path = os.path.join(CHATS_FOLDER, filename)
            group_name = get_group_name_from_file(filename)
            try:
                messages = parse_whatsapp(file_path)
                if group_name not in chat_data:
                    chat_data[group_name] = {
                        'filename': filename,
                        'messages': messages
                    }
                else:
                    # If group already exists, merge messages
                    chat_data[group_name]['messages'].extend(messages)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Sort messages by timestamp for each group
    for group_name, data in chat_data.items():
        messages = data['messages']
        messages.sort(key=lambda msg: parse_timestamp(msg['timestamp']) or datetime.min)

    return chat_data

def index(request):
    return render(request, 'chatapp/index.html')

@require_http_methods(["GET"])
def get_groups(request):
    load_all_chats()
    groups = list(chat_data.keys())
    return JsonResponse({"groups": groups})

@csrf_exempt
@require_http_methods(["POST"])
def summarize(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    summary_type = data.get('summary_type', 'total')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    user = data.get('user')

    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    messages = chat_data[group_name]['messages']

    # Filter by date range if provided
    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None

        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)

        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    # Generate summary based on type
    if summary_type == 'total':
        summary = generate_total_summary(filtered_messages)
        return JsonResponse({"summary_type": "total", "summary": summary})

    elif summary_type == 'user_messages':
        user_messages = generate_user_messages(filtered_messages)
        return JsonResponse({"summary_type": "user_messages", "user_messages": user_messages})

    elif summary_type == 'user_wise':
        users = get_users_in_messages(filtered_messages)
        return JsonResponse({"summary_type": "user_wise", "users": users})

    elif summary_type == 'user_messages_for_user':
        if not user:
            return JsonResponse({"error": "No user specified"}, status=400)
        user_messages = generate_user_messages_for_user(filtered_messages, user)
        return JsonResponse({"summary_type": "user_messages_for_user", "user": user, "user_messages": user_messages})

    elif summary_type == 'weekly_summary':
        weekly_summaries = generate_weekly_summary(filtered_messages, start_date_str, end_date_str)
        return JsonResponse({"summary_type": "weekly_summary", "weekly_summaries": weekly_summaries})

    else:
        return JsonResponse({"error": "Invalid summary type"}, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def ask(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    user_question = data.get('question')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)
    if not user_question:
        return JsonResponse({"error": "No question provided"}, status=400)

    messages = chat_data[group_name]['messages']

    # Filter by date range if provided
    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None

        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)

        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    chat_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in filtered_messages])

    max_chars = 30000
    if len(chat_text) > max_chars:
        chat_text = chat_text[-max_chars:]

    try:
        prompt = "Answer questions based on this WhatsApp chat. Be concise and specific.\n\n" + chat_text + f"\n\nQuestion: {user_question}"
        response = generate_with_gemini(prompt)
        answer = response
        return JsonResponse({"answer": answer})
    except Exception as e:
        return JsonResponse({"error": f"Error generating answer: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def group_events(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    messages = chat_data[group_name]['messages']

    # Filter by date range if provided
    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None

        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)

        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    # Analyze group events
    events = analyze_group_events(filtered_messages)
    event_counts = get_event_counts(events)
    top_removers = get_top_removers(events)

    return JsonResponse({
        "event_counts": event_counts,
        "top_removers": top_removers
    })

@csrf_exempt
@require_http_methods(["POST"])
def sentiment(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    messages = chat_data[group_name]['messages']

    # Filter by date range if provided
    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None

        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)

        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)
    result = analyze_sentiment(filtered_messages)
    return JsonResponse(result)

@csrf_exempt
@require_http_methods(["POST"])
def topic(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    top_n = int(data.get('top_n', 5))
    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)
    messages = chat_data[group_name]['messages']

    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)
        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)
    topics = extract_topics(filtered_messages, top_n=top_n)
    return JsonResponse({"topics": topics})

@csrf_exempt
@require_http_methods(["POST"])
def business_metrics(request):
    data = json.loads(request.body)
    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    if not group_name or group_name not in chat_data:
        return JsonResponse({"error": "Invalid group name"}, status=400)
    messages = chat_data[group_name]['messages']

    filtered_messages = messages
    if start_date_str or end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        if end_date:
            end_date = end_date.replace(hour=23, minute=59, second=59)
        filtered_messages = []
        for msg in messages:
            msg_date = parse_timestamp(msg['timestamp'])
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            filtered_messages.append(msg)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)
    metrics = calculate_business_metrics(filtered_messages)
    return JsonResponse(metrics)
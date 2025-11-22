import os
import re
import json
import csv
import requests
from datetime import datetime, timedelta
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.core.files.storage import default_storage
from dotenv import load_dotenv
from .models import ChatFile
from .config import GEMINI_API_KEY, MAX_CHARS_FOR_ANALYSIS
from .utils import parse_timestamp, filter_messages_by_date


def settings_test(request):
    """Test view to check Django settings"""
    import os
    from django.conf import settings
    from django.http import JsonResponse

    settings_info = {
        'settings_module': os.environ.get('DJANGO_SETTINGS_MODULE', 'Not set'),
        'debug': settings.DEBUG,
        'allowed_hosts': settings.ALLOWED_HOSTS,
        'middleware': settings.MIDDLEWARE,
        'installed_apps': settings.INSTALLED_APPS,
    }

    return JsonResponse(settings_info)


def comprehensive_test(request):
    """Comprehensive test view to check all aspects of Django setup"""
    import os
    from django.conf import settings
    from django.http import JsonResponse

    # Get request information
    request_info = {
        'method': request.method,
        'path': request.path,
        'full_path': request.get_full_path(),
        'host': request.get_host(),
        'is_secure': request.is_secure(),
        'meta': dict(request.META),
    }

    # Get settings information
    settings_info = {
        'settings_module': os.environ.get('DJANGO_SETTINGS_MODULE', 'Not set'),
        'debug': settings.DEBUG,
        'allowed_hosts': list(settings.ALLOWED_HOSTS) if hasattr(settings, 'ALLOWED_HOSTS') else 'Not found',
        'middleware': list(settings.MIDDLEWARE) if hasattr(settings, 'MIDDLEWARE') else 'Not found',
        'installed_apps': list(settings.INSTALLED_APPS) if hasattr(settings, 'INSTALLED_APPS') else 'Not found',
    }

    # Check if host is in allowed hosts
    host_check = {
        'request_host': request.get_host(),
        'is_allowed': request.get_host() in settings.ALLOWED_HOSTS if hasattr(settings, 'ALLOWED_HOSTS') else False,
    }

    return JsonResponse({
        'request_info': request_info,
        'settings_info': settings_info,
        'host_check': host_check,
    })


from .business_metrics import calculate_business_metrics
from .group_event import (
    analyze_group_events,
    get_event_counts,
    get_event_details,
    get_top_removers,
    _normalize_events,
    _filter_normalized,
    compute_timeseries,
    compute_distribution,
    compute_most_active_day,
    compute_top_contributors,
    extract_unique_actors,
)
from .sentiment_analyzer import analyze_sentiment
from .summary_generator import (
    generate_total_summary,
    generate_user_messages,
    get_users_in_messages,
    generate_user_messages_for_user,
    generate_weekly_summary,
    generate_brief_summary,
    generate_daily_user_messages,
    generate_user_wise_detailed_report,
    generate_comprehensive_summary
)
from .question_processor import QuestionProcessor
from .study_report_generator import generate_study_report_html

load_dotenv()

# Use Google Gemini API
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}/generateContent"

# Health check endpoint for deployment monitoring
def health_check(request):
    """Health check endpoint for monitoring deployment status"""
    return JsonResponse({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'WhatsApp Group Analytics',
        'version': '1.0.0'
    })


def simple_test(request):
    """Very simple test view to check if basic Django setup is working"""
    from django.http import HttpResponse
    return HttpResponse(b"Simple test view is working!", content_type="text/plain")


def favicon(request):
    """Simple favicon handler to prevent 400 errors"""
    from django.http import HttpResponse
    # Return a simple empty response for favicon requests
    return HttpResponse(b'', content_type='image/x-icon')


def debug_view(request):
    """Debug view to help identify the cause of 400 errors"""
    debug_info = f"""
    <html>
    <body>
        <h1>Debug Information</h1>
        <p><strong>Request Method:</strong> {request.method}</p>
        <p><strong>Request Path:</strong> {request.path}</p>
        <p><strong>Request META:</strong></p>
        <ul>
    """

    for key, value in request.META.items():
        if key.startswith('HTTP_') or key in ['CONTENT_TYPE', 'CONTENT_LENGTH', 'QUERY_STRING']:
            debug_info += f"<li><strong>{key}:</strong> {value}</li>"

    debug_info += """
        </ul>
        <p><strong>Headers:</strong></p>
        <ul>
    """

    for header, value in request.headers.items():
        debug_info += f"<li><strong>{header}:</strong> {value}</li>"

    debug_info += """
        </ul>
        <h2>Server Information</h2>
        <p><strong>Host:</strong> {request.get_host()}</p>
        <p><strong>Is Secure:</strong> {request.is_secure()}</p>
        <p><strong>Build Info:</strong> This is a debug view to help troubleshoot 400 errors</p>
    </body>
    </html>
    """
    return HttpResponse(debug_info.format(request=request).encode('utf-8'))


def debug_detailed_view(request):
    """Detailed debug view to help identify the cause of 400 errors"""
    import json
    from django.http import HttpResponse

    debug_info = {
        'request_method': request.method,
        'request_path': request.path,
        'request_full_path': request.get_full_path(),
        'request_headers': dict(request.headers),
        'request_meta': {},
        'get_params': dict(request.GET),
        'post_params': dict(request.POST),
        'host': request.get_host(),
        'is_secure': request.is_secure(),
        'build_info': 'Debug view to help troubleshoot 400 errors'
    }

    # Add relevant META information
    for key, value in request.META.items():
        if key.startswith('HTTP_') or key in ['CONTENT_TYPE', 'CONTENT_LENGTH', 'QUERY_STRING', 'REMOTE_ADDR']:
            debug_info['request_meta'][key] = value

    # Return JSON response
    return HttpResponse(
        json.dumps(debug_info, indent=2).encode('utf-8'),
        content_type='application/json'
    )


def generate_fallback_answer(question, messages):
    """Generate a comprehensive fallback answer when AI is unavailable"""
    if not messages:
        return "I don't have any messages to analyze for this date range."

    question_lower = question.lower()

    # Analyze basic statistics
    total_messages = len(messages)
    users = set(msg['sender'] for msg in messages)
    user_count = len(users)

    # User activity analysis
    user_msg_count = {}
    for msg in messages:
        user = msg['sender']
        user_msg_count[user] = user_msg_count.get(user, 0) + 1

    most_active_user = max(user_msg_count.items(), key=lambda x: x[1]) if user_msg_count else None

    # Extract meaningful content and filter system messages
    meaningful_messages = []
    meeting_messages = []
    file_messages = []
    topic_messages = []
    question_messages = []
    decision_messages = []
    link_messages = []

    for msg in messages:
        message_text = msg['message'].strip()
        message_lower = message_text.lower()

        # Skip system messages
        if any(term in message_lower for term in ['security code', 'media omitted', 'tap to learn', 'left', 'added', 'removed']):
            continue

        # Collect meaningful messages
        if len(message_text) > 15:
            meaningful_messages.append(msg)

            # Look for meeting-related content
            if any(word in message_lower for word in ['meet', 'meeting', 'मिटिंग', 'मीटिंग', 'दौरा', 'आयोजन', 'उपस्थित', 'schedule', 'appointment']):
                meeting_messages.append(msg)

            # Look for file/document sharing
            if any(ext in message_lower for ext in ['.pdf', '.doc', '.jpg', '.png', '.mp4', '.xlsx', '.jpeg', '.docx', '.txt', '.pptx']):
                file_messages.append(msg)

            # Look for questions
            if any(q_word in message_lower for q_word in ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who']):
                question_messages.append(msg)

            # Look for decisions/announcements
            if any(dec_word in message_lower for dec_word in ['decided', 'decision', 'final', 'confirm', 'approved', 'rejected', 'concluded']):
                decision_messages.append(msg)

            # Look for links
            if 'http' in message_lower or 'www.' in message_lower:
                link_messages.append(msg)

            # Collect other substantial content
            if len(message_text) > 30:
                topic_messages.append(msg)

    # Handle different types of questions with more specific logic
    if any(word in question_lower for word in ['meet', 'meeting', 'schedule', 'appointment', 'मिटिंग', 'दौरा']):
        if meeting_messages:
            answer = "📅 **Meetings Found:**\n\n"
            for i, msg in enumerate(meeting_messages[:5], 1):  # Show up to 5 meetings
                meeting_content = msg['message'][:200] + "..." if len(msg['message']) > 200 else msg['message']
                answer += f"**{i}. Meeting on {msg['timestamp']}**\n"
                answer += f"👤 Organized by: {msg['sender']}\n"
                answer += f"📝 Details: {meeting_content}\n\n"
            return answer
        else:
            return "No meetings found in the conversation history for the selected date range."

    elif any(word in question_lower for word in ['most active', 'who', 'active user', 'top contributor']) and 'least' not in question_lower:
        if most_active_user:
            # Show top 5 users
            sorted_users = sorted(user_msg_count.items(), key=lambda x: x[1], reverse=True)
            answer = "👥 **Most Active Users:**\n\n"
            for i, (user, count) in enumerate(sorted_users[:5], 1):
                percentage = round((count/total_messages)*100, 1)
                answer += f"**{i}. {user}**: {count} messages ({percentage}%)\n"
            return answer
        else:
            return "Unable to determine user activity from the available data."

    elif any(word in question_lower for word in ['least active', 'inactive', 'lowest activity', 'least messages']) or ('least' in question_lower and any(word in question_lower for word in ['active', 'user', 'contributor'])):
        if user_msg_count:
            # Show bottom 5 users (least active)
            sorted_users = sorted(user_msg_count.items(), key=lambda x: x[1])  # Ascending order for least active
            answer = "👥 **Least Active Users:**\n\n"
            for i, (user, count) in enumerate(sorted_users[:5], 1):
                percentage = round((count/total_messages)*100, 1)
                answer += f"**{i}. {user}**: {count} messages ({percentage}%)\n"
            return answer
        else:
            return "Unable to determine user activity from the available data."

    elif any(word in question_lower for word in ['how many', 'total', 'messages', 'count', 'number']):
        answer = f"📊 **Message Statistics:**\n\n"
        answer += f"• **Total Messages**: {total_messages}\n"
        answer += f"• **Total Users**: {user_count}\n"
        answer += f"• **Date Range**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
        answer += f"• **Average per User**: {round(total_messages/user_count, 1)} messages\n"
        return answer

    elif any(word in question_lower for word in ['file', 'document', 'pdf', 'shared', 'attachment']):
        if file_messages:
            answer = "📎 **Files/Documents Shared:**\n\n"
            for i, msg in enumerate(file_messages[:5], 1):
                answer += f"**{i}. {msg['timestamp']}**\n"
                answer += f"👤 Shared by: {msg['sender']}\n"
                answer += f"📄 File: {msg['message'][:100]}...\n\n"
            return answer
        else:
            return "No files or documents were shared in the selected time period."

    elif any(word in question_lower for word in ['link', 'url', 'website']):
        if link_messages:
            answer = "🔗 **Links Shared:**\n\n"
            for i, msg in enumerate(link_messages[:5], 1):
                answer += f"**{i}. {msg['timestamp']}**\n"
                answer += f"👤 Shared by: {msg['sender']}\n"
                # Extract URLs
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', msg['message'])
                for url in urls[:2]:  # Show up to 2 URLs per message
                    answer += f"🌐 Link: {url}\n"
                answer += "\n"
            return answer
        else:
            return "No links were shared in the selected time period."

    elif any(word in question_lower for word in ['when', 'time', 'day', 'date']) or any(time_word in question_lower for time_word in ['morning', 'evening', 'afternoon', 'night', 'am', 'pm']):
        if messages:
            # Enhanced time analysis with specific time filtering
            from collections import defaultdict
            day_counts = defaultdict(int)
            hour_counts = defaultdict(int)
            time_specific_messages = []

            # Extract time-specific keywords from question
            time_keywords = ['morning', 'afternoon', 'evening', 'night']
            specific_time = None
            for keyword in time_keywords:
                if keyword in question_lower:
                    specific_time = keyword
                    break

            for msg in messages:
                try:
                    timestamp = msg['timestamp']
                    dt = parse_timestamp(timestamp)
                    if dt:
                        day_key = dt.strftime('%Y-%m-%d')
                        day_counts[day_key] += 1
                        hour_counts[dt.hour] += 1

                        # Filter for specific time periods if mentioned
                        if specific_time:
                            if (specific_time == 'morning' and 6 <= dt.hour <= 11) or \
                               (specific_time == 'afternoon' and 12 <= dt.hour <= 17) or \
                               (specific_time == 'evening' and 18 <= dt.hour <= 21) or \
                               (specific_time == 'night' and (dt.hour >= 22 or dt.hour <= 5)):
                                time_specific_messages.append(msg)
                except:
                    continue

            answer = f"📅 **Activity Timeline:**\n\n"

            if specific_time and time_specific_messages:
                answer += f"📍 **{specific_time.title()} Activity** ({len(time_specific_messages)} messages):\n\n"
                for i, msg in enumerate(time_specific_messages[:5], 1):
                    answer += f"**{i}. {msg['timestamp']}** - {msg['sender']}: {msg['message'][:100]}...\n"
                answer += "\n"

            if day_counts:
                most_active_day = max(day_counts.items(), key=lambda x: x[1])
                answer += f"• **Most Active Day**: {most_active_day[0]} ({most_active_day[1]} messages)\n"

            if hour_counts:
                most_active_hour = max(hour_counts.items(), key=lambda x: x[1])
                answer += f"• **Most Active Hour**: {most_active_hour[0]}:00 ({most_active_hour[1]} messages)\n"

            answer += f"• **Total Date Range**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
            answer += f"• **Total Days with Activity**: {len(day_counts)}\n"
            return answer

        return f"Messages range from {messages[0]['timestamp']} to {messages[-1]['timestamp']}."

    elif any(word in question_lower for word in ['topic', 'discuss', 'about', 'content', 'summary', 'talk']):
        if topic_messages:
            answer = "💬 **Main Discussion Topics:**\n\n"
            # Group messages by sender to show diverse content
            user_topics = {}
            for msg in topic_messages[:20]:  # Increase sample size
                user = msg['sender']
                if user not in user_topics:
                    user_topics[user] = []
                if len(user_topics[user]) < 5:  # Increased to 5 topics per user for better coverage
                    content = msg['message'][:150] + "..." if len(msg['message']) > 150 else msg['message']
                    user_topics[user].append({
                        'content': content,
                        'timestamp': msg['timestamp']
                    })

            topic_count = 1
            for user, topics in list(user_topics.items())[:7]:  # Show up to 7 users
                for topic in topics:
                    answer += f"**{topic_count}. {topic['timestamp']}**\n"
                    answer += f"👤 {user}: {topic['content']}\n\n"
                    topic_count += 1
                    if topic_count > 25:  # Increased to 25 topics total for better content coverage
                        break
                if topic_count > 25:
                    break

            return answer
        else:
            return "The conversation appears to contain mostly brief exchanges or media sharing."

    elif any(word in question_lower for word in ['question', 'ask', 'query']):
        if question_messages:
            answer = "❓ **Questions Asked:**\n\n"
            for i, msg in enumerate(question_messages[:5], 1):
                question_content = msg['message'][:150] + "..." if len(msg['message']) > 150 else msg['message']
                answer += f"**{i}. {msg['timestamp']}**\n"
                answer += f"👤 Asked by: {msg['sender']}\n"
                answer += f"📝 Question: {question_content}\n\n"
            return answer
        else:
            return "No specific questions were found in the selected time period."

    elif any(word in question_lower for word in ['decision', 'final', 'conclude', 'agree']):
        if decision_messages:
            answer = "✅ **Decisions Made:**\n\n"
            for i, msg in enumerate(decision_messages[:5], 1):
                decision_content = msg['message'][:150] + "..." if len(msg['message']) > 150 else msg['message']
                answer += f"**{i}. {msg['timestamp']}**\n"
                answer += f"👤 By: {msg['sender']}\n"
                answer += f"📝 Decision: {decision_content}\n\n"
            return answer
        else:
            return "No specific decisions were found in the selected time period."

    elif any(word in question_lower for word in ['list', 'show', 'all']):
        # Enhanced listing with user-specific and comprehensive options
        if 'meet' in question_lower or 'मिटिंग' in question_lower:
            # Already handled above
            return generate_fallback_answer("meetings", messages)
        elif 'user' in question_lower:
            # Check if asking for a specific user
            specific_user = None
            for user in user_msg_count.keys():
                if user.lower() in question_lower:
                    specific_user = user
                    break

            if specific_user:
                # Show specific user details
                user_count = user_msg_count.get(specific_user, 0)
                percentage = round((user_count/total_messages)*100, 1)
                answer = f"📊 **Details for {specific_user}:**\n\n"
                answer += f"• **Messages Sent**: {user_count}\n"
                answer += f"• **Percentage of Total**: {percentage}%\n"
                answer += f"• **Rank**: #{sorted(user_msg_count.items(), key=lambda x: x[1], reverse=True).index((specific_user, user_count)) + 1} out of {len(user_msg_count)} users\n\n"

                # Show recent messages from this user
                user_messages = [msg for msg in messages if msg['sender'] == specific_user][-5:]
                if user_messages:
                    answer += f"📝 **Recent Messages:**\n"
                    for i, msg in enumerate(user_messages, 1):
                        content = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
                        answer += f"**{i}. {msg['timestamp']}**: {content}\n"
                return answer
            else:
                # Show all users with numbers
                answer = "👥 **All Users with Message Counts:**\n\n"
                sorted_users = sorted(user_msg_count.items(), key=lambda x: x[1], reverse=True)
                for i, (user, count) in enumerate(sorted_users, 1):
                    percentage = round((count/total_messages)*100, 1)
                    answer += f"{i}. **{user}**: {count} messages ({percentage}%)\n"
                return answer
        else:
            # Show general overview
            answer = "📋 **Chat Overview:**\n\n"
            answer += f"• **{total_messages} messages** from **{user_count} users**\n"
            answer += f"• **Time Period**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
            if meeting_messages:
                answer += f"• **{len(meeting_messages)} meetings** mentioned\n"
            if file_messages:
                answer += f"• **{len(file_messages)} files** shared\n"
            if link_messages:
                answer += f"• **{len(link_messages)} links** shared\n"
            answer += f"• **Most Active**: {most_active_user[0]} ({most_active_user[1]} messages)\n" if most_active_user else ""
            return answer

    else:
        # Enhanced general answer with actual insights
        answer = "📊 **Chat Analysis:**\n\n"
        answer += f"• **Total Activity**: {total_messages} messages from {user_count} users\n"
        if most_active_user:
            answer += f"• **Most Active**: {most_active_user[0]} with {most_active_user[1]} messages\n"
        if meeting_messages:
            answer += f"• **Meetings Mentioned**: {len(meeting_messages)} meeting-related discussions\n"
        if file_messages:
            answer += f"• **Files Shared**: {len(file_messages)} documents/media shared\n"
        if link_messages:
            answer += f"• **Links Shared**: {len(link_messages)} URLs shared\n"
        if question_messages:
            answer += f"• **Questions Asked**: {len(question_messages)} questions\n"
        answer += f"• **Time Range**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n\n"

        # Add sample messages for context
        if meaningful_messages:
            answer += "🗨️ **Sample Messages:**\n"
            for i, msg in enumerate(meaningful_messages[:3], 1):
                sample = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
                answer += f"**{i}. {msg['sender']}**: {sample}\n"
            answer += "\n"

        answer += "💡 **Try asking specific questions like**:\n"
        answer += "• 'List meetings'\n"
        answer += "• 'Who is most active?'\n"
        answer += "• 'What topics were discussed?'\n"
        answer += "• 'Show files shared'\n"
        answer += "• 'Any decisions made?'\n"
        answer += "• 'Links shared in chat?'\n"

        return answer

def generate_with_gemini(prompt):
    """Generate content using Google Gemini API with better error handling"""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return "API_ERROR"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params={"key": api_key}, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "API_ERROR"

    except requests.exceptions.Timeout:
        return "API_ERROR"
    except requests.exceptions.RequestException as e:
        # Check if it's a quota exceeded error
        if "429" in str(e) or "quota" in str(e).lower():
            return "QUOTA_EXCEEDED"
        return "API_ERROR"
    except KeyError as e:
        return "API_ERROR"
    except Exception as e:
        return "API_ERROR"

# def parse_whatsapp(file_path):
#     messages = []
#     current_message = None
#     patterns = [
#         # --- ADD THIS NEW PATTERN AT THE TOP ---
#         r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}) - (.*?): (.*)',

#         r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}\u202F[AP]M) - (.*?): (.*)',
#         r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M) - (.*?): (.*)',
#         r'\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?(?: [AP]M)?)\] (.*?): (.*)',
#         r'(\d{4}-\d{1,2}-\d{1,2}, \d{1,2}:\d{2}) - (.*?): (.*)',
#         r'(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}) - (.*?): (.*)',
#         r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M) - (.*?): (.*)'
#     ]
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             if not line:
#                 continue
#             matched = False
#             for pattern in patterns:
#                 match = re.match(pattern, line, re.IGNORECASE)
#                 if match:
#                     if current_message:
#                         messages.append(current_message)

#                     timestamp, sender, message = match.groups()
#                     current_message = {
#                         'timestamp': timestamp,
#                         'sender': sender,
#                         'message': message
#                     }
#                     matched = True
#                     break
#             if not matched and current_message:
#                 if current_message['message']:
#                     current_message['message'] += '\n' + line
#                 else:
#                     current_message['message'] = line
#     if current_message:
#         messages.append(current_message)
#     return messages


import re

def parse_whatsapp(file_path):
    messages = []
    current_message = None

    # --- REGEX COMPONENTS ---
    # Flexible date: 11/11/25, 2025-11-11, 11/11/2025
    date_pat = r'\d{1,4}[/-]\d{1,2}[/-]\d{2,4}'

    # Flexible time: 10:00, 10:00:00, 10:00 AM, 10:00<nbsp>PM
    # Matches narrow no-break space (\u202F) common in WhatsApp
    time_pat = r'\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202F]?[APap][Mm])?'

    timestamp_pat = f"{date_pat}, {time_pat}"

    patterns = [
        # 1. iOS Style: [Timestamp] Sender: Message
        # ^[\u200E\u200F]* -> Ignores invisible Left-to-Right marks at start of line
        r'^[\u200E\u200F]*\[(' + timestamp_pat + r')\]\s+(.*?):\s+(.*)',

        # 2. Android Style: Timestamp - Sender: Message
        r'^[\u200E\u200F]*(' + timestamp_pat + r')\s-\s(.*?):\s+(.*)'
    ]

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    # If we found a new timestamp, push the previous message to list
                    if current_message:
                        messages.append(current_message)

                    timestamp, sender, message = match.groups()

                    # Clean the extracted timestamp string just in case (remove invisible chars)
                    timestamp = timestamp.replace('\u200e', '').replace('\u200f', '').strip()

                    current_message = {
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message
                    }
                    matched = True
                    break

            # If line didn't match a timestamp, append it to the previous message
            # (This handles multi-line messages)
            if not matched and current_message:
                if current_message['message']:
                    current_message['message'] += '\n' + line
                else:
                    current_message['message'] = line

    # Append the final message after loop ends
    if current_message:
        messages.append(current_message)

    return messages

def get_group_name_from_file(filename):
    # Handle undefined or empty filenames
    if not filename or filename == 'undefined' or not isinstance(filename, str):
        return "Unnamed WhatsApp Group"
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name if name else "Unnamed WhatsApp Group"

def load_all_chats():
    chat_data = {}
    chat_files = ChatFile.objects.all()
    print(f"Found {len(chat_files)} chat files in database")
    for chat_file in chat_files:
        file_path = chat_file.file.path
        group_name = chat_file.group_name
        print(f"Loading file: {chat_file.original_filename}, group: {group_name}, path: {file_path}")
        try:
            messages = parse_whatsapp(file_path)
            print(f"Parsed {len(messages)} messages from {chat_file.original_filename}")
            if group_name not in chat_data:
                chat_data[group_name] = {
                    'filenames': [chat_file.original_filename],
                    'file_ids': [chat_file.id],
                    'messages': messages
                }
            else:
                chat_data[group_name]['filenames'].append(chat_file.original_filename)
                chat_data[group_name]['file_ids'].append(chat_file.id)
                chat_data[group_name]['messages'].extend(messages)
        except Exception as e:
            print(f"Error loading {chat_file.original_filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Loaded groups: {list(chat_data.keys())}")
    print("The return data is:", chat_data)

    try:
        # Sort messages by timestamp for each group
        for group_name, data in chat_data.items():
            messages = data['messages']
            messages.sort(key=lambda msg: parse_timestamp(msg['timestamp']) or datetime.min)

        return chat_data
    except Exception as e:
        import traceback
        print(traceback.format_exc())

def index(request):
    # Redirect legacy root to the new Home page to surface the modern UI
    return redirect('home')

# New pages for modern UI

def home(request):
    """
    Home page with upload + group selection
    Handles all HTTP methods and provides proper error responses
    """
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    # Log the request details
    logger.info(f"Home view accessed: {request.method} {request.path}")
    logger.info(f"Request META: {dict(request.META)}")

    try:
        response = render(request, 'chatapp/home.html')
        logger.info(f"Home view rendered successfully: {response.status_code}")
        return response
    except Exception as e:
        # Log the error for debugging with full traceback
        logger.error(f"Error in home view: {str(e)}", exc_info=True)
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Return a detailed error response to help with debugging
        from django.http import HttpResponse
        error_message = f"Error loading home page: {str(e)}"
        return HttpResponse(error_message.encode('utf-8'), content_type="text/plain")

def test_home_render(request):
    """Test view to debug home template rendering"""
    import logging
    import traceback
    from django.template.loader import render_to_string
    logger = logging.getLogger(__name__)

    try:
        # Test if we can render the template directly
        logger.info("Testing direct template rendering...")
        rendered_content = render_to_string('chatapp/home.html')
        logger.info(f"Direct template rendering successful: {len(rendered_content)} characters")

        # Test the actual render function
        logger.info("Testing Django render function...")
        response = render(request, 'chatapp/home.html')
        logger.info(f"Django render successful: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error in test_home_render: {str(e)}", exc_info=True)
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Return detailed error information
        error_info = f"Error in test_home_render: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return HttpResponse(error_info.encode('utf-8'), content_type="text/plain")

def test_view(request):
    """Simple test view to check if routing is working"""
    return HttpResponse(b"Test view is working!")

def test_api(request):
    """Simple test API endpoint"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Test API endpoint accessed")
    return JsonResponse({"status": "success", "message": "API is working"})

def dashboard(request):
    # Render the old dashboard with date hints
    group = request.GET.get('group', '')
    context = {'group': group}
    if group:
        chat_data = load_all_chats()
        if group in chat_data:
            messages = chat_data[group]['messages']
            if messages:
                from .utils import parse_timestamp
                dates = [parse_timestamp(msg['timestamp']) for msg in messages if parse_timestamp(msg['timestamp'])]
                # Filter out None values before finding min/max
                valid_dates = [date for date in dates if date is not None]
                if valid_dates:
                    first_date = min(valid_dates)
                    last_date = max(valid_dates)
                    context['first_date'] = first_date.strftime('%d / %m / %Y')
                    context['last_date'] = last_date.strftime('%d / %m / %Y')
    return render(request, 'chatapp/dashboard.html', context)

def react_dashboard(request):
    # React + Tailwind + Recharts powered dashboard
    group = request.GET.get('group', '')
    context = {
        'group': group
    }
    if group:
        chat_data = load_all_chats()
        if group in chat_data:
            messages = chat_data[group]['messages']
            if messages:
                from .utils import parse_timestamp
                dates = [parse_timestamp(msg['timestamp']) for msg in messages if parse_timestamp(msg['timestamp'])]
                # Filter out None values before finding min/max
                valid_dates = [date for date in dates if date is not None]
                if valid_dates:
                    start_date = min(valid_dates).strftime('%d / %m / %Y')
                    end_date = max(valid_dates).strftime('%d / %m / %Y')
                    context['chat_start_date'] = start_date
                    context['chat_end_date'] = end_date
    return render(request, 'chatapp/react_dashboard.html', context)

def group_events_page(request):
    return render(request, 'chatapp/group_events_dashboard.html')


@require_http_methods(["GET"])
def get_group_dates(request):
    group = request.GET.get('group', '')
    if not group:
        return JsonResponse({"error": "No group specified"}, status=400)
    chat_data = load_all_chats()
    if group not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)
    messages = chat_data[group]['messages']
    if not messages:
        return JsonResponse({"error": "No messages"}, status=400)
    from .utils import parse_timestamp
    dates = [parse_timestamp(msg['timestamp']) for msg in messages if parse_timestamp(msg['timestamp'])]
    # Filter out None values before finding min/max
    valid_dates = [date for date in dates if date is not None]
    if not valid_dates:
        return JsonResponse({"error": "No valid dates"}, status=400)
    start_date = min(valid_dates).strftime('%d / %m / %Y')
    end_date = max(valid_dates).strftime('%d / %m / %Y')
    return JsonResponse({"start_date": start_date, "end_date": end_date})

@csrf_exempt
@require_http_methods(["POST"])
def group_events_analytics(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    event_types = data.get('event_types')  # list or None
    user = data.get('user')  # string or None

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    # First pass filter coarse by date for performance
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)
    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    # Build events and normalize
    events = analyze_group_events(filtered_messages)
    normalized = _normalize_events(events)

    # Prepare datetime bounds for fine filtering
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
    if end_dt:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    rows = _filter_normalized(normalized, start_dt, end_dt, event_types, user)

    # Aggregations
    timeseries = compute_timeseries(rows)
    distribution = compute_distribution(rows)
    most_active = compute_most_active_day(timeseries)
    top_contributors = compute_top_contributors(rows, limit=5)

    # Card counts from filtered rows
    card_counts = {'added': 0, 'left': 0, 'removed': 0, 'changed_subject': 0, 'changed_icon': 0, 'created': 0}
    for r in rows:
        card_counts[r['event_type']] += 1

    actors = extract_unique_actors(rows)

    return JsonResponse({
        "event_counts": card_counts,
        "insights": {
            "most_active_day": most_active,  # e.g., {date, total, ...}
            "total_events": distribution.get('total', 0),
            "top_contributors": top_contributors,
        },
        "timeseries": timeseries,
        "distribution": distribution,
        "actors": actors,
    })

@csrf_exempt
@require_http_methods(["POST"])
def group_events_logs(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    event_types = data.get('event_types')
    user = data.get('user')

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)
    if not filtered_messages:
        return JsonResponse({"events": []})

    events = analyze_group_events(filtered_messages)
    normalized = _normalize_events(events)

    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
    if end_dt:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    rows = _filter_normalized(normalized, start_dt, end_dt, event_types, user)

    # Shape rows for table
    table_rows = []
    for r in rows:
        table_rows.append({
            'event_type': r['event_type'],
            'actor': r['actor'],
            'target': r['target'],
            'timestamp': r['dt'].strftime('%d-%b-%Y %I:%M %p'),
            'details': r['details'] or '',
        })

    return JsonResponse({"events": table_rows})

@require_http_methods(["GET"])
def get_groups(request):
    chat_data = load_all_chats()
    groups = list(chat_data.keys())
    return JsonResponse({"groups": groups})

@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Upload request received. Method: {request.method}")
        logger.info(f"Files in request: {list(request.FILES.keys())}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Request META: {dict(request.META)}")

        if request.method == 'POST':
            file_obj = request.FILES.get('file')
            if not file_obj:
                logger.error("No file provided in request")
                return JsonResponse({"error": "No file provided"}, status=400)

            logger.info(f"File received: {file_obj.name}, size: {file_obj.size}")

            if not file_obj.name.endswith('.txt'):
                logger.error(f"Invalid file type: {file_obj.name}")
                return JsonResponse({"error": "Only .txt files are supported"}, status=400)

            # Validate filename is not undefined or empty
            if not file_obj.name or file_obj.name == 'undefined':
                logger.error(f"Invalid file name: {file_obj.name}")
                return JsonResponse({"error": "Invalid file name"}, status=400)

            group_name = get_group_name_from_file(file_obj.name)
            logger.info(f"Group name derived: {group_name}")

            chat_file = ChatFile(
                file=file_obj,
                original_filename=file_obj.name,
                group_name=group_name
            )

            logger.info("Saving chat file to database...")
            chat_file.save()
            logger.info(f"File saved successfully with ID: {chat_file.id}")

            return JsonResponse({
                "success": True,
                "group_name": group_name,
                "file_id": chat_file.id
            })

        logger.error("Invalid request method")
        return JsonResponse({"error": "Invalid request method"}, status=405)
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return JsonResponse({"error": f"Upload failed: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def delete_file(request):
    data = json.loads(request.body)
    file_id = data.get('file_id')
    if not file_id:
        return JsonResponse({"error": "No file ID provided"}, status=400)
    try:
        chat_file = ChatFile.objects.get(id=file_id)
        if chat_file.file:
            chat_file.file.delete()
        chat_file.delete()
        return JsonResponse({"success": True})
    except ChatFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def get_uploaded_files(request):
    import logging
    logger = logging.getLogger(__name__)

    try:
        files = []
        chat_files = ChatFile.objects.all().order_by('-uploaded_at')
        logger.info(f"Found {len(chat_files)} chat files in database")

        for chat_file in chat_files:
            files.append({
                'id': chat_file.id,
                'name': chat_file.original_filename,
                'group_name': chat_file.group_name,
                'uploaded_at': chat_file.uploaded_at.strftime('%d-%b-%Y %I:%M %p')
            })

        logger.info(f"Returning {len(files)} files")
        return JsonResponse({"files": files})
    except Exception as e:
        logger.error(f"Error in get_uploaded_files: {str(e)}", exc_info=True)
        return JsonResponse({"error": f"Failed to retrieve files: {str(e)}"}, status=500)

# This function was duplicated and is now removed
# The correct group_events function is implemented below

@csrf_exempt
@require_http_methods(["POST"])
def summarize(request):
    try:
        data = json.loads(request.body)
        group_name = data.get('group_name')
        summary_type = data.get('summary_type', 'total')

        chat_data = load_all_chats()
        if group_name not in chat_data:
            return JsonResponse({"error": "Group not found"}, status=404)

        messages = chat_data[group_name]['messages']
        messages = filter_messages_by_date(messages, data.get('start_date'), data.get('end_date'))

        if not messages:
            return JsonResponse({"error": "No messages found"}, status=400)

        response_data = {"summary_type": summary_type}

        if summary_type == 'total':
            response_data["summary"] = generate_total_summary(messages)
        elif summary_type == 'brief':
            response_data["summary"] = generate_brief_summary(messages)
        elif summary_type == 'weekly_summary':
            # This is the heavy function that needs the delays
            response_data["weekly_summaries"] = generate_weekly_summary(messages)
        elif summary_type == 'comprehensive':
            response_data["report"] = {
                'brief_summary': generate_brief_summary(messages),
                'weekly_summaries': generate_weekly_summary(messages)
            }

        return JsonResponse(response_data)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def generate_study_report(request):
    """
    Generate a formatted study report in HTML format
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    report_format = data.get('format', 'html')

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    try:
        # Generate the study report
        template_path = generate_study_report_html(filtered_messages, start_date_str, end_date_str)

        # For now, we'll return a success message
        # In a full implementation, we would render the template and return the HTML
        return JsonResponse({
            "success": True,
            "message": "Study report generated successfully",
            "template_path": template_path
        })
    except Exception as e:
        return JsonResponse({"error": f"Failed to generate study report: {str(e)}"}, status=500)

@require_http_methods(["GET"])
def get_example_questions(request):
    """Get example questions for the help system."""
    example_questions = {
        "user_activity": [
            "Who are the most active users?",
            "Who are the least active users?",
            "Show me messages from [User Name]",
            "What did [User Name] say?",
            "List messages from [Phone Number]",
            "Who sent the most messages?",
            "Who sent the least messages?",
            "Show activity of [User Name]",
            "How many messages did [User Name] send?",
            "Who are the top 5 most active users?"
        ],
        "time_based": [
            "Show messages from 3:30 PM to 4:30 PM",
            "What was said between 2 PM and 5 PM?",
            "Show messages at 4:00 PM",
            "Messages from morning (6 AM to 12 PM)",
            "Messages from afternoon (12 PM to 6 PM)",
            "Messages from evening (6 PM to 12 AM)",
            "Messages from night (12 AM to 6 AM)",
            "Show messages on [specific date]",
            "What happened between [start time] and [end time]?",
            "Show activity during business hours"
        ],
        "analytics": [
            "How many messages are there?",
            "What's the total message count?",
            "How many users are in the group?",
            "What's the average messages per user?",
            "Show message statistics",
            "What's the activity breakdown?",
            "How many messages per day?",
            "What's the busiest day?",
            "What's the quietest day?",
            "Show daily message counts"
        ],
        "content_analysis": [
            "What topics were discussed?",
            "What were the main subjects?",
            "What did people talk about?",
            "Show me the main themes",
            "What were the key discussions?",
            "What topics came up most?",
            "What was the conversation about?",
            "What subjects were covered?",
            "What were people discussing?",
            "What were the main points?"
        ],
        "meetings_events": [
            "List all meetings mentioned",
            "What meetings were scheduled?",
            "Show meeting discussions",
            "What events were planned?",
            "List upcoming meetings",
            "What meetings happened?",
            "Show meeting details",
            "What was discussed in meetings?",
            "List meeting participants",
            "What meeting topics were covered?"
        ],
        "files_media": [
            "What files were shared?",
            "Show me shared documents",
            "What media was shared?",
            "List all attachments",
            "What files were uploaded?",
            "Show shared links",
            "What documents were sent?",
            "List shared images",
            "What videos were shared?",
            "Show all shared content"
        ],
        "decisions_actions": [
            "What decisions were made?",
            "What was decided?",
            "Show me final decisions",
            "What actions were taken?",
            "What was concluded?",
            "What was agreed upon?",
            "Show me resolutions",
            "What was finalized?",
            "What decisions were reached?",
            "What was the outcome?"
        ],
        "questions_queries": [
            "What questions were asked?",
            "Show me all questions",
            "What did people ask?",
            "List unanswered questions",
            "What questions need answers?",
            "Show question discussions",
            "What queries were raised?",
            "What questions came up?",
            "List all inquiries",
            "What questions were posted?"
        ],
        "sentiment_mood": [
            "What's the overall mood?",
            "How is everyone feeling?",
            "What's the sentiment?",
            "Show positive messages",
            "Show negative messages",
            "What's the emotional tone?",
            "How positive is the chat?",
            "What's the general attitude?",
            "Show happy messages",
            "What's the mood like?"
        ],
        "business_insights": [
            "What business topics were discussed?",
            "Show project updates",
            "What work was mentioned?",
            "List business decisions",
            "What projects were discussed?",
            "Show work-related messages",
            "What business activities happened?",
            "List client discussions",
            "What work was planned?",
            "Show business communications"
        ]
    }

    return JsonResponse({
        "categories": example_questions,
        "total_questions": sum(len(questions) for questions in example_questions.values())
    })

import json
import logging
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Ensure these are imported from your project structure
# from .utils import load_all_chats, generate_with_gemini
# from .processors import QuestionProcessor
logger = logging.getLogger(__name__)


def get_filtered_context(group_name, start_date_str, end_date_str):
    """
    1. Loads chat files for the specific group.
    2. Filters messages between start_date and end_date.
    3. Returns a single string of text (the context).
    """

    # 1. Fetch only relevant files from Django DB
    # We filter by group_name to avoid processing unnecessary files
    chat_files = ChatFile.objects.filter(group_name=group_name)

    if not chat_files.exists():
        return "Error: Group not found."

    all_messages = []

    # 2. Parse the files
    for chat_file in chat_files:
        try:
            # Reuse your existing parse function
            msgs = parse_whatsapp(chat_file.file.path)
            all_messages.extend(msgs)
        except Exception as e:
            print(f"Error reading {chat_file.original_filename}: {e}")

    # 3. Process Dates for filtering
    # Assuming input strings are "YYYY-MM-DD", e.g., "2025-10-28"
    try:
        filter_start = datetime.strptime(start_date_str, "%Y-%m-%d")

        # Set end date to end of the day (23:59:59) to capture all chats on that day
        filter_end = datetime.strptime(end_date_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
    except ValueError:
        return "Error: Dates must be in YYYY-MM-DD format."

    # 4. Filter and Build Context String
    context_lines = []

    # Sort messages first to ensure chronological order in the prompt
    # We use a lambda that handles None results from parse_timestamp safely
    all_messages.sort(key=lambda x: parse_timestamp(x['timestamp']) or datetime.min)

    for msg in all_messages:
        msg_dt = parse_timestamp(msg['timestamp'])

        if msg_dt:
            if filter_start <= msg_dt <= filter_end:
                # Format: [Timestamp] Sender: Message
                line = f"[{msg['timestamp']}] {msg['sender']}: {msg['message']}"
                context_lines.append(line)

    if not context_lines:
        return ""

    # Join with newlines
    return "\n".join(context_lines)


def ask_question_on_chat(group_name, start_date, end_date, question):
    """
    Constructs the final prompt for the LLM.
    """

    # Step 1: Get the chat history for the date range
    context_text = get_filtered_context(group_name, start_date, end_date)

    if not context_text:
        return "No messages found in this date range to analyze."

    # Step 2: Construct the Prompt
    # This structure works well for Gemini/GPT models
    prompt = f"""
You are an intelligent assistant analyzing a WhatsApp group chat.
Below is a segment of the conversation from {start_date} to {end_date}.

--- START OF CHAT CONTEXT ---
{context_text}
--- END OF CHAT CONTEXT ---

Based strictly on the context above, please answer the following question:
Question: {question}
    """

    return prompt

from .summary_generator import generate_with_gemini
@csrf_exempt
@require_http_methods(["POST"])
def ask_question(request):
    try:
        # 1. Validate and Parse Input
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)

        group_name = data.get('group_name')
        user_question = data.get('question')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        print(f"Processing question for group: {group_name}")
        final_prompt = ask_question_on_chat(group_name, start_date, end_date, user_question)
        result = generate_with_gemini(final_prompt)
        return JsonResponse({
            "answer": result,
            "data": result
        })


        import pdb
        pdb.set_trace()

        if not group_name or not user_question:
            return JsonResponse({"error": "Missing group_name or question"}, status=400)

        # 2. Load Data
        try:
            chat_data = load_all_chats()
        except Exception as e:
            logger.error(f"Data load error: {e}")
            return JsonResponse({"error": "Failed to load chat database"}, status=500)

        if group_name not in chat_data:
            return JsonResponse({"error": f"Group '{group_name}' not found"}, status=404)

        # 3. Process Question using Logic
        messages = chat_data[group_name]['messages']
        processor = QuestionProcessor(messages, group_name)

        # Pass date range to the processor
        result = processor.process_question(user_question, start_date=start_date, end_date=end_date)

        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=400)

        # 4. Generate Answer based on Result Type
        result_type = result.get("type", "unknown")
        answer = ""

        # --- CASE A: General AI Query ---
        if result_type == "general":
            answer = generate_ai_answer(user_question, result.get("context", ""))

        # --- CASE B: Structured Data Formatting ---
        else:
            answer = format_structured_response(result_type, result)

        return JsonResponse({
            "answer": answer,
            "data": result
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Internal processing error: {str(e)}"}, status=500)


# --- Helper Functions to keep View Clean ---

def generate_ai_answer(question, context):
    """Handles AI generation for general context questions"""
    prompt = f"""You are an expert WhatsApp chat analyst. Answer the user's question based strictly on the provided context logs.

    Context Logs:
    {context}

    User Question: {question}

    Instructions:
    - Be precise and mention specific user names/numbers if available.
    - If the context is empty, say "I couldn't find specific information about that in the chat."
    - Keep it concise.

    Answer:"""

    try:
        return generate_with_gemini(prompt)
    except Exception as e:
        logger.error(f"AI Generation failed: {e}")
        return f"Based on the chat data: {context[:500]}..." # Fallback


def format_structured_response(result_type, result):
    """Formats structured data (lists, counts) into a readable string"""
    answer = ""

    # 1. Date Based / User Messages / Time Range
    if result_type in ["date_based", "user_messages", "time_based"]:
        title = f"Messages for {result.get('date', 'selected period')}"
        if result_type == "user_messages":
            title = f"Messages from {result.get('user', 'User')}"

        answer = f"**{title}**\nTotal: {result.get('total_messages', 0)}\n\n"

        # Group by sender for cleaner reading
        messages_list = result.get('messages', [])
        current_sender = None

        for msg in messages_list:
            sender = msg.get('sender', 'Unknown')
            timestamp = msg.get('timestamp', '')
            content = msg.get('message', '')

            # Simple header if sender changes
            if sender != current_sender:
                answer += f"\n**{sender}:**\n"
                current_sender = sender

            # Format: [Time]: Message
            time_str = timestamp.split()[-1] if ' ' in timestamp else timestamp
            answer += f"  [{time_str}] {content}\n"

    # 2. Topics
    elif result_type == "topics":
        topics_data = result.get("topics", {})
        main_topics = topics_data.get("main_topics", []) or topics_data.get("topics", [])

        answer = "**Main Topics Discussed:**\n"
        # Handle both list of strings or list of dicts (frequency)
        for i, item in enumerate(main_topics[:10], 1):
            if isinstance(item, dict):
                answer += f"{i}. {item.get('topic')} ({item.get('frequency')} mentions)\n"
            else:
                answer += f"{i}. {item}\n"

        key_msgs = topics_data.get("key_messages", [])
        if key_msgs:
            answer += "\n**Key Context:**\n"
            for msg in key_msgs[:5]:
                answer += f"- **{msg.get('sender')}**: {msg.get('message')[:100]}...\n"

    # 3. User Statistics (Most/Least Active)
    elif result_type in ["most_active_users", "least_active_users"]:
        category = "Most" if result_type == "most_active_users" else "Least"
        answer = f"**{category} Active Users:**\n\n"
        for i, u in enumerate(result.get("users", []), 1):
            answer += f"{i}. **{u['user']}**: {u['message_count']} messages ({u['percentage']}%)\n"

    # 4. General Counts
    elif result_type == "message_count":
        answer = "**Message Statistics:**\n\n"
        answer += f"• Total Messages: {result.get('total_messages', 0)}\n"
        answer += f"• Total Users: {result.get('total_users', 0)}\n"
        if 'user_breakdown' in result:
            answer += "\n**Breakdown:**\n"
            # Sort top 5 for brevity
            sorted_users = sorted(result['user_breakdown'].items(), key=lambda x: x[1], reverse=True)[:10]
            for user, count in sorted_users:
                answer += f"- {user}: {count}\n"

    # 5. Fallback
    else:
        answer = "I processed your request, but the result type is generic.\n"
        if "metrics" in result:
             m = result["metrics"]
             answer += f"Analyzed {m.get('total_messages', 0)} messages from {m.get('total_users', 0)} users."

    return answer

@csrf_exempt
@require_http_methods(["POST"])
def group_events(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)

    print(f"Found {len(filtered_messages)} messages in date range")

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    events = analyze_group_events(filtered_messages)
    event_counts = get_event_counts(events)
    top_removers = get_top_removers(events)

    # Get detailed lists for each event type
    added_list = get_detailed_event_list(events, 'added')
    left_list = get_detailed_event_list(events, 'left')
    removed_list = get_detailed_event_list(events, 'removed')
    changed_subject_list = get_detailed_event_list(events, 'changed_subject')
    changed_icon_list = get_detailed_event_list(events, 'changed_icon')
    created_list = get_detailed_event_list(events, 'created')

    print(f"Event counts: {event_counts}")
    print(f"Total events found: {len(events)}")

    return JsonResponse({
        "event_counts": event_counts,
        "top_removers": top_removers,
        "added_list": added_list,
        "left_list": left_list,
        "removed_list": removed_list,
        "changed_subject_list": changed_subject_list,
        "changed_icon_list": changed_icon_list,
        "created_list": created_list
    })

@csrf_exempt
@require_http_methods(["POST"])
def event_details(request):

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    event_type = data.get('event_type')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)
    if not event_type:
        return JsonResponse({"error": "No event type provided"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)
    events = analyze_group_events(filtered_messages)
    event_details = get_detailed_event_list(events, event_type)

    return JsonResponse({
        "event_type": event_type,
        "events": event_details
    })

# This function was removed to avoid conflict with the imported analyze_group_events from group_event.py
# The imported function returns a dictionary format that is compatible with other group event functions

def is_added_event(text, original_message):
    """Check if message is an 'added' event"""
    patterns = [
        'added',
        'joined',
        'was added',
        'has been added',
        'added to the group',
        'joined the group',
        'was added to the group'
    ]
    return any(pattern in text for pattern in patterns)

def is_left_event(text, original_message):
    """Check if message is a 'left' event"""
    patterns = [
        'left',
        'exited',
        'left the group',
        'exited the group',
        'has left',
        'has exited'
    ]
    return any(pattern in text for pattern in patterns)

def is_removed_event(text, original_message):
    """Check if message is a 'removed' event"""
    patterns = [
        'removed',
        'kicked',
        'was removed',
        'has been removed',
        'removed from the group',
        'kicked from the group',
        'was kicked'
    ]
    return any(pattern in text for pattern in patterns)

def is_subject_changed_event(text, original_message):
    """Check if message is a 'subject changed' event"""
    patterns = [
        'changed the subject',
        'changed subject',
        'subject changed',
        'changed group subject',
        'group subject changed'
    ]
    return any(pattern in text for pattern in patterns)

def is_icon_changed_event(text, original_message):
    """Check if message is an 'icon changed' event"""
    patterns = [
        'changed the group icon',
        'changed group icon',
        'group icon changed',
        'changed the icon',
        'icon changed'
    ]
    return any(pattern in text for pattern in patterns)

def is_group_created_event(text, original_message):
    """Check if message is a 'group created' event"""
    patterns = [
        'created group',
        'group created',
        'created the group',
        'group was created'
    ]
    return any(pattern in text for pattern in patterns)

def get_event_details(events, event_type):
    """Get details for specific event type"""
    # Use the events dictionary directly (imported from group_event.py)
    filtered_events = events.get(event_type, [])

    # Sort by timestamp (newest first)
    filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)

    return filtered_events

def extract_added_details(message):
    """Extract details from 'added' event message"""
    import re

    # Try to extract who added whom
    patterns = [
        r'(\w+(?:\s+\w+)*)\s+added\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+has\s+added\s+(\w+(?:\s+\w+)*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            adder = match.group(1).strip()
            added_person = match.group(2).strip()
            return f"{adder} added {added_person}"

    # Fallback
    if 'added' in message.lower():
        parts = message.split('added')
        if len(parts) > 1:
            return f"Added: {parts[1].strip()}"
    return "Member was added to the group"

def extract_left_details(message):
    """Extract details from 'left' event message"""
    import re

    # Try to extract who left
    patterns = [
        r'(\w+(?:\s+\w+)*)\s+left',
        r'(\w+(?:\s+\w+)*)\s+exited',
        r'(\w+(?:\s+\w+)*)\s+has\s+left',
        r'(\w+(?:\s+\w+)*)\s+has\s+exited'
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            person = match.group(1).strip()
            return f"{person} left the group"

    # Fallback
    if 'left' in message.lower():
        return f"Left: {message.replace('left', '').strip()}"
    elif 'exited' in message.lower():
        return f"Exited: {message.replace('exited', '').strip()}"
    return "Member left the group"

def extract_removed_details(message):
    """Extract details from 'removed' event message"""
    import re

    # Try to extract who removed whom
    patterns = [
        r'(\w+(?:\s+\w+)*)\s+removed\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+kicked\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+has\s+removed\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+has\s+kicked\s+(\w+(?:\s+\w+)*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            remover = match.group(1).strip()
            removed_person = match.group(2).strip()
            return f"{remover} removed {removed_person}"

    # Fallback
    if 'removed' in message.lower():
        parts = message.split('removed')
        if len(parts) > 1:
            return f"Removed: {parts[1].strip()}"
    elif 'kicked' in message.lower():
        parts = message.split('kicked')
        if len(parts) > 1:
            return f"Kicked: {parts[1].strip()}"
    return "Member was removed from the group"

def extract_subject_change_details(message):
    """Extract details from subject change event message"""
    # Common patterns: "User changed the subject to 'New Subject'"
    if 'changed the subject' in message:
        if 'to' in message:
            parts = message.split('to')
            if len(parts) > 1:
                return f"Subject changed to: {parts[1].strip()}"
        return "Group subject was changed"
    return "Group subject was changed"

def extract_person_name(message, event_type):
    """Extract person name from event message"""
    import re

    if event_type == 'added':
        # Patterns: "John Doe added Jane Smith", "John Doe added Jane Smith and 2 others"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+added\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+has\s+added\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+added\s+(\w+(?:\s+\w+)*)\s+to\s+the\s+group'
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(2).strip()

        # Fallback: split by 'added'
        if 'added' in message.lower():
            parts = message.split('added')
            if len(parts) > 1:
                name = parts[1].strip()
                # Remove common suffixes
                name = re.sub(r'\s+and\s+\d+\s+others?', '', name)
                name = re.sub(r'\s+to\s+the\s+group', '', name)
                return name.strip()

    elif event_type == 'left':
        # Patterns: "John Doe left", "John Doe exited"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+left',
            r'(\w+(?:\s+\w+)*)\s+exited',
            r'(\w+(?:\s+\w+)*)\s+has\s+left',
            r'(\w+(?:\s+\w+)*)\s+has\s+exited'
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback
        if 'left' in message.lower():
            return message.replace('left', '').strip()
        elif 'exited' in message.lower():
            return message.replace('exited', '').strip()

    elif event_type == 'removed':
        # Patterns: "John Doe removed Jane Smith", "John Doe kicked Jane Smith"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+removed\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+kicked\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+has\s+removed\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+has\s+kicked\s+(\w+(?:\s+\w+)*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(2).strip()

        # Fallback
        if 'removed' in message.lower():
            parts = message.split('removed')
            if len(parts) > 1:
                return parts[1].strip()
        elif 'kicked' in message.lower():
            parts = message.split('kicked')
            if len(parts) > 1:
                return parts[1].strip()

    return 'Unknown'

def extract_subject_name(message):
    """Extract new subject name from subject change message"""
    if 'changed the subject to' in message:
        parts = message.split('to')
        if len(parts) > 1:
            subject = parts[1].strip()
            # Remove quotes if present
            subject = subject.strip('"').strip("'")
            return subject
    return 'New Subject'

def get_event_counts(events):
    """Get count of each event type"""
    # Use the events dictionary directly (imported from group_event.py)
    return {
        'added': len(events.get('added', [])),
        'left': len(events.get('left', [])),
        'removed': len(events.get('removed', [])),
        'changed_subject': len(events.get('changed_subject', [])),
        'changed_icon': len(events.get('changed_icon', [])),
        'created': len(events.get('created', []))
    }

def get_detailed_event_list(events, event_type):
    """Get detailed list of events for a specific event type"""
    detailed_list = []

    # Use the events dictionary directly
    event_list = events.get(event_type, [])

    for event in event_list:
        if event_type == 'added':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('adder', 'Unknown'),
                'target': event.get('added_person', 'Unknown'),
                'details': event.get('raw_message', '')
            })
        elif event_type == 'left':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('person', 'Unknown'),
                'target': None,
                'details': event.get('raw_message', '')
            })
        elif event_type == 'removed':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('remover', 'Unknown'),
                'target': event.get('removed_person', 'Unknown'),
                'details': event.get('raw_message', '')
            })
        elif event_type == 'changed_subject':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('changer', 'Unknown'),
                'target': event.get('new_subject', 'Unknown'),
                'details': event.get('raw_message', '')
            })
        elif event_type == 'changed_icon':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('changer', 'Unknown'),
                'target': None,
                'details': event.get('raw_message', '')
            })
        elif event_type == 'created':
            detailed_list.append({
                'timestamp': event.get('timestamp', ''),
                'actor': event.get('creator', 'Unknown'),
                'target': None,
                'details': event.get('raw_message', '')
            })

    # Sort by timestamp (newest first)
    detailed_list.sort(key=lambda x: x['timestamp'], reverse=True)
    return detailed_list

def get_top_removers(events):
    """Get top users who removed others"""
    # Use the events dictionary directly (imported from group_event.py)
    removed_events = events.get('removed', [])
    remover_counts = {}

    for event in removed_events:
        remover = event.get('remover', 'Unknown')
        if remover in remover_counts:
            remover_counts[remover] += 1
        else:
            remover_counts[remover] = 1

    # Sort by count and return top 5
    sorted_removers = sorted(remover_counts.items(), key=lambda x: x[1], reverse=True)
    return [{'user': user, 'count': count} for user, count in sorted_removers[:5]]

@csrf_exempt
@require_http_methods(["POST"])
def sentiment(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    print(f"Sentiment analysis request: group={group_name}, start={start_date_str}, end={end_date_str}")

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    try:
        chat_data = load_all_chats()
        print(f"Available groups: {list(chat_data.keys())}")

        if group_name not in chat_data:
            return JsonResponse({"error": "Group not found"}, status=404)

        messages = chat_data[group_name]['messages']
        print(f"Total messages in group: {len(messages)}")

        # Filter messages by date range
        filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)
        print(f"Filtered messages count: {len(filtered_messages)}")

        if not filtered_messages:
            return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

        # Perform sentiment analysis
        print(f"About to call analyze_sentiment with {len(filtered_messages)} messages")
        result = analyze_sentiment(filtered_messages)
        print(f"Sentiment analysis completed. Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if isinstance(result, dict):
            print(f"Sentiment breakdown: {result.get('sentiment_breakdown')}")
            print(f"Overall sentiment: {result.get('overall_sentiment')}")
        else:
            print(f"Unexpected result type: {result}")

        # Ensure we have the expected format for frontend
        if 'sentiment_breakdown' not in result:
            result['sentiment_breakdown'] = result.get('overall_sentiment', {'positive': 0, 'neutral': 0, 'negative': 0})

        # Ensure sentiment_breakdown is a dictionary
        if not isinstance(result.get('sentiment_breakdown', {}), dict):
            result['sentiment_breakdown'] = {'positive': 0, 'neutral': 0, 'negative': 0}
        else:
            # Make sure all keys exist
            sb = result['sentiment_breakdown']
            result['sentiment_breakdown'] = {
                'positive': sb.get('positive', 0),
                'neutral': sb.get('neutral', 0),
                'negative': sb.get('negative', 0)
            }

        # Add total count for frontend display
        total_count = sum(result['sentiment_breakdown'].values())
        result['total_analyzed'] = str(total_count)

        return JsonResponse(result)

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def activity_analysis(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    specific_date_str = data.get('specific_date')  # For hourly analysis
    week_start_str = data.get('week_start')       # For weekly analysis
    week_end_str = data.get('week_end')           # For weekly analysis
    start_date_str = data.get('start_date')       # Generic range
    end_date_str = data.get('end_date')
    user_filter = data.get('user')                # Optional user filter
    include_messages = bool(data.get('include_messages', False))

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    try:
        chat_data = load_all_chats()
        if group_name not in chat_data:
            return JsonResponse({"error": "Group not found"}, status=404)
    except Exception as e:
        print(f"Error loading chat data: {e}")
        return JsonResponse({"error": "Failed to load chat data"}, status=500)

    messages = chat_data[group_name]['messages']

    # Pre-filter by user if provided
    if user_filter:
        messages = [m for m in messages if m.get('sender') == user_filter]

    # Filter messages based on the provided date parameters
    if specific_date_str:
        # For hourly analysis on a specific date
        start_date = datetime.strptime(specific_date_str, '%Y-%m-%d')
        end_date = start_date.replace(hour=23, minute=59, second=59)
        filtered_messages = [msg for msg in messages if
                            parse_timestamp(msg['timestamp']) is not None and
                            start_date <= parse_timestamp(msg['timestamp']) <= end_date]
        analysis_type = "hourly"
    elif week_start_str and week_end_str:
        # For weekly analysis
        start_date = datetime.strptime(week_start_str, '%Y-%m-%d')
        end_date = datetime.strptime(week_end_str, '%Y-%m-%d')
        end_date = end_date.replace(hour=23, minute=59, second=59)
        filtered_messages = [msg for msg in messages if
                            parse_timestamp(msg['timestamp']) is not None and
                            start_date <= parse_timestamp(msg['timestamp']) <= end_date]
        analysis_type = "weekly"
    elif start_date_str and end_date_str:
        # Generic date range analysis
        filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)
        analysis_type = "range"
    else:
        # Default to all messages
        filtered_messages = messages
        analysis_type = "all"


    # Apply user filter again after date filtering (safety)
    if user_filter:
        filtered_messages = [m for m in filtered_messages if m.get('sender') == user_filter]

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    try:
        # For very large datasets, sample messages to get representative data
        max_messages = 10000  # Reduced for better performance
        if len(filtered_messages) > max_messages:
            print(f"Large dataset detected ({len(filtered_messages)} messages), sampling {max_messages} messages")
            # Sample messages evenly across the time period to maintain patterns
            step = len(filtered_messages) // max_messages
            filtered_messages = filtered_messages[::step][:max_messages]

        print(f"Processing {len(filtered_messages)} messages for activity analysis")
        raw_metrics = calculate_business_metrics(filtered_messages)
        print(f"Raw metrics calculated: {len(raw_metrics)} metrics")
    except Exception as e:
        print(f"Error calculating business metrics: {e}")
        return JsonResponse({"error": "Failed to calculate metrics"}, status=500)

    # Transform to frontend-expected shape
    # Hourly: array 0..23 aligned with labels
    raw_hourly_activity = raw_metrics.get('activity_by_hour', {}) or {}
    hourly_activity = [int(raw_hourly_activity.get(str(h), 0) or 0) for h in range(24)]

    print(f"Raw hourly activity: {raw_hourly_activity}")
    print(f"Processed hourly activity: {hourly_activity}")

    raw_daily_activity = raw_metrics.get('activity_by_day', {}) or {}
    daily_activity = [0 for _ in range(7)]

    print(f"Raw daily activity: {raw_daily_activity}")

    for key, cnt in raw_daily_activity.items():
        try:
            numeric_key = int(key)
            if 0 <= numeric_key <= 6:
                daily_activity[numeric_key] = int(cnt)
                continue
        except (TypeError, ValueError):
            pass

        day_name_map = {
            'Sunday': 0,
            'Monday': 1,
            'Tuesday': 2,
            'Wednesday': 3,
            'Thursday': 4,
            'Friday': 5,
            'Saturday': 6
        }
        idx = day_name_map.get(str(key))
        if idx is not None:
            daily_activity[idx] = int(cnt)

    print(f"Processed daily activity: {daily_activity}")

    message_counts = raw_metrics.get('messages_per_user', {})
    print(f"Message counts per user: {message_counts}")

    # Ensure we have at least the top 5 users for a good pie chart
    sorted_users = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top users: {sorted_users[:5]}")

    # Build list of all users for the selected period ignoring the user filter
    available_users = []
    try:
        base_msgs = chat_data[group_name]['messages']
        if specific_date_str:
            _start = datetime.strptime(specific_date_str, '%Y-%m-%d')
            _end = _start.replace(hour=23, minute=59, second=59)
            period_msgs = [m for m in base_msgs if parse_timestamp(m['timestamp']) is not None and _start <= parse_timestamp(m['timestamp']) <= _end]
        elif week_start_str and week_end_str:
            _start = datetime.strptime(week_start_str, '%Y-%m-%d')
            _end = datetime.strptime(week_end_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            period_msgs = [m for m in base_msgs if parse_timestamp(m['timestamp']) is not None and _start <= parse_timestamp(m['timestamp']) <= _end]
        elif start_date_str and end_date_str:
            period_msgs = filter_messages_by_date(base_msgs, start_date_str, end_date_str)
        else:
            period_msgs = base_msgs
        available_users = sorted({m.get('sender') for m in period_msgs if m.get('sender')})
    except Exception:
        available_users = []

    # --- Week splitting logic for frontend week cards ---
    weeks = []
    if start_date_str and end_date_str:
        try:
            # Split the selected period into weeks (Monday-Sunday)
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            # Align start to previous Monday
            start_of_week = start_dt - timedelta(days=start_dt.weekday())
            current = start_of_week
            while current <= end_dt:
                week_start = current
                week_end = min(current + timedelta(days=6), end_dt)
                week_msgs = [m for m in filtered_messages if parse_timestamp(m['timestamp']) is not None and week_start <= parse_timestamp(m['timestamp']) <= week_end]
                week_users = sorted({m.get('sender') for m in week_msgs if m.get('sender')})
                week_message_counts = {}
                for m in week_msgs:
                    sender = m.get('sender')
                    if sender:
                        week_message_counts[sender] = week_message_counts.get(sender, 0) + 1
                # Find most active user and peak hour for the week
                most_active_user = max(week_message_counts.items(), key=lambda x: x[1])[0] if week_message_counts else None
                # Hourly activity for the week
                week_hourly = {h: 0 for h in range(24)}
                for m in week_msgs:
                    dt = parse_timestamp(m['timestamp'])
                    if dt:
                        week_hourly[dt.hour] += 1
                peak_hour = max(week_hourly.items(), key=lambda x: x[1])[0] if week_hourly else None
                # Daily activity for the week (0=Sunday, 1=Monday, ..., 6=Saturday)
                week_daily = {i: 0 for i in range(7)}
                for m in week_msgs:
                    dt = parse_timestamp(m['timestamp'])
                    if dt:
                        # Convert weekday() to frontend format: 0=Sunday, 1=Monday, etc.
                        day_index = (dt.weekday() + 1) % 7  # Monday=0 -> Sunday=0, Tuesday=1 -> Monday=1, etc.
                        week_daily[day_index] += 1
                week_data = {
                    'start': week_start.strftime('%Y-%m-%d'),
                    'end': week_end.strftime('%Y-%m-%d'),
                    'message_count': len(week_msgs),
                    'users': week_users,
                    'message_counts': week_message_counts,
                    'most_active_user': most_active_user,
                    'peak_hour': peak_hour,
                    'daily_activity': {int(k): v for k, v in week_daily.items()},  # Convert string keys to int
                    'hourly_activity': {int(k): v for k, v in week_hourly.items()},  # Convert string keys to int
                    'messages': week_msgs,
                }
                print(f"Week {len(weeks)+1}: {week_data['start']} - {week_data['end']}, Messages: {len(week_msgs)}, Daily: {week_daily}, Hourly: {week_hourly}")
                weeks.append(week_data)
                current += timedelta(days=7)
        except Exception as e:
            print(f"Error in week splitting logic: {e}")
            # Continue without weeks data rather than failing completely

    activity_data = {
        'total_messages': raw_metrics.get('total_messages', 0),
        'total_users': raw_metrics.get('total_users', 0),
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'message_counts': message_counts,
        'analysis_type': analysis_type,
        'all_users': available_users,
        'weeks': weeks if weeks else None,
    }

    if include_messages:
        activity_data['messages'] = filtered_messages

    return JsonResponse(activity_data)

@csrf_exempt
@require_http_methods(["POST"])
def export_data(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    group_name = data.get('group_name')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    # For PDF export compatibility, define these as None if not present
    week_start_str = data.get('week_start', None)
    week_end_str = data.get('week_end', None)
    specific_date_str = data.get('specific_date', None)
    export_features = data.get('features', [])
    export_format = data.get('format', 'json')

    if not group_name:
        return JsonResponse({"error": "Invalid group name"}, status=400)

    chat_data = load_all_chats()
    if group_name not in chat_data:
        return JsonResponse({"error": "Group not found"}, status=404)

    messages = chat_data[group_name]['messages']
    filtered_messages = filter_messages_by_date(messages, start_date_str, end_date_str)

    if not filtered_messages:
        return JsonResponse({"error": "No messages found in the selected date range"}, status=400)

    export_data = {}

    if 'summary' in export_features or 'all' in export_features:
        export_data['summary'] = generate_total_summary(filtered_messages)

    if 'sentiment' in export_features or 'all' in export_features:
        export_data['sentiment'] = analyze_sentiment(filtered_messages)

    if 'activity' in export_features or 'all' in export_features:
        export_data['activity'] = calculate_business_metrics(filtered_messages)

    if 'events' in export_features or 'all' in export_features:
        events = analyze_group_events(filtered_messages)
        export_data['events'] = {
            'event_counts': get_event_counts(events),
            'top_removers': [{'user': user, 'count': count} for user, count in get_top_removers(events)]
        }

    if 'messages' in export_features or 'all' in export_features:
        export_data['messages'] = filtered_messages

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{group_name}_chat_analysis_{timestamp}"

    # Remove JSON export and keep only CSV export
    if export_format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
        writer = csv.writer(response)
        writer.writerow(['Timestamp', 'Sender', 'Message'])
        for msg in filtered_messages:
            writer.writerow([msg['timestamp'], msg['sender'], msg['message']])
        return response

    elif export_format == 'excel':
        try:
            import io
            import xlsxwriter
        except Exception as e:
            return JsonResponse({"error": "Excel export requires xlsxwriter"}, status=500)
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        ws = workbook.add_worksheet('Messages')
        headers = ['Timestamp', 'Sender', 'Message']
        for c, h in enumerate(headers):
            ws.write(0, c, h)
        for r, msg in enumerate(filtered_messages, start=1):
            ws.write(r, 0, msg.get('timestamp', ''))
            ws.write(r, 1, msg.get('sender', ''))
            ws.write(r, 2, msg.get('message', ''))
        workbook.close()
        output.seek(0)
        response = HttpResponse(output.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="{filename}.xlsx"'
        return response

    elif export_format == 'pdf':
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import cm
            import io
        except Exception:
            return JsonResponse({"error": "PDF export requires reportlab"}, status=500)
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        p.setFont("Helvetica-Bold", 14)
        p.drawString(2*cm, y, f"Chat Analysis Report: {group_name}")
        y -= 1*cm
        p.setFont("Helvetica", 10)
        p.drawString(2*cm, y, f"Date range: {start_date_str or week_start_str or specific_date_str} - {end_date_str or week_end_str or specific_date_str}")
        y -= 1*cm
        p.setFont("Helvetica-Bold", 12)
        p.drawString(2*cm, y, "Messages:")
        y -= 0.7*cm
        p.setFont("Helvetica", 9)
        for msg in filtered_messages[:1000]:
            line = f"{msg.get('timestamp','')} - {msg.get('sender','')}: {msg.get('message','')[:150]}"
            if y < 2*cm:
                p.showPage(); y = height - 2*cm; p.setFont("Helvetica", 9)
            p.drawString(2*cm, y, line)
            y -= 0.5*cm
        p.showPage()
        p.save()
        pdf = buffer.getvalue()
        buffer.close()
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}.pdf"'
        return response

    else:
        # Default to CSV export if unsupported format is requested
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
        writer = csv.writer(response)
        writer.writerow(['Timestamp', 'Sender', 'Message'])
        for msg in filtered_messages:
            writer.writerow([msg['timestamp'], msg['sender'], msg['message']])
        return response

@csrf_exempt
@require_http_methods(["GET", "POST"])
def debug_groups(request):
    """Debug endpoint to list available groups and basic info"""
    try:
        chat_data = load_all_chats()
        groups_info = {}

        for group_name, group_data in chat_data.items():
            messages = group_data.get('messages', [])
            groups_info[group_name] = {
                'total_messages': len(messages),
                'first_message_date': None,
                'last_message_date': None
            }

            # Find date range
            dates = [parse_timestamp(msg['timestamp']) for msg in messages if parse_timestamp(msg['timestamp'])]
            if dates:
                groups_info[group_name]['first_message_date'] = min(dates).strftime('%Y-%m-%d')
                groups_info[group_name]['last_message_date'] = max(dates).strftime('%Y-%m-%d')

        return JsonResponse({
            'available_groups': list(chat_data.keys()),
            'groups_info': groups_info,
            'total_groups': len(chat_data)
        })

    except Exception as e:
        print(f"Error in debug_groups: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)
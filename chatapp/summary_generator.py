import re
from datetime import datetime, timedelta
import google.generativeai as genai
from django.conf import settings
import logging
from typing import Optional
import os
import requests

from .utils import parse_timestamp

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable
model: Optional[genai.GenerativeModel] = None

# Google Gemini API configuration
MODEL_NAME = "gemini-2.0-flash-exp"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


def initialize_gemini_model():
    """Initialize the Gemini AI model with proper configuration"""
    global model

    try:
        # Get API key from environment or Django settings
        api_key = os.getenv('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)

        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment or settings")
            return False

        # Configure Gemini AI (simplified configuration without ClientOptions)
        genai.configure(api_key=api_key)

        # Initialize the model with fallback
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            logger.warning(f"Could not initialize gemini-2.5-flash, falling back to gemini-2.0-flash: {e}")
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e2:
                logger.warning(f"Could not initialize gemini-2.0-flash, falling back to gemini-flash-latest: {e2}")
                try:
                    model = genai.GenerativeModel('gemini-flash-latest')
                except Exception as e3:
                    logger.warning(f"Could not initialize gemini-flash-latest, falling back to gemini-pro-latest: {e3}")
                    model = genai.GenerativeModel('gemini-pro-latest')

        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return False

# Initialize the model when module is loaded, but don't fail if it doesn't work
try:
    initialize_gemini_model()
except Exception as e:
    logger.warning(f"Failed to initialize Gemini model on module load: {e}")

def generate_user_messages(messages):
    """
    Groups all messages by user.
    Returns: Dictionary { 'User A': [msg1, msg2], 'User B': [msg3] }
    """
    if not messages:
        return {}

    user_groups = {}
    for msg in messages:
        sender = msg.get('sender', 'Unknown')
        # Filter out system messages if needed
        if 'media omitted' in msg.get('message', '').lower():
            continue

        if sender not in user_groups:
            user_groups[sender] = []

        # Clean up the message object
        clean_msg = {
            'timestamp': msg.get('timestamp', ''),
            'message': msg.get('message', ''),
            'sender': sender
        }
        user_groups[sender].append(clean_msg)

    return user_groups

def get_users_in_messages(messages):
    """
    Extracts a sorted list of unique participants from the messages.
    Returns: List ['Alice', 'Bob', 'Charlie']
    """
    if not messages:
        return []

    # Use a set to get unique senders, then convert to sorted list
    users = set()
    for msg in messages:
        sender = msg.get('sender')
        if sender and sender != 'Unknown':
            users.add(sender)

    return sorted(list(users))

def generate_user_messages_for_user(messages, user):
    """Generate messages for a specific user"""
    user_messages = []
    for msg in messages:
        if msg['sender'] == user:
            user_messages.append(msg)
    return user_messages

def generate_weekly_summary(messages):
    """
    Generates weekly summaries with ARTIFICIAL DELAYS to prevent 429 errors.
    """
    if not messages: return []

    weekly_messages = {}
    for msg in messages:
        try:
            ts = parse_timestamp(msg['timestamp'])
            if ts:
                monday = ts - timedelta(days=ts.weekday())
                key = monday.strftime('%Y-%m-%d')
                if key not in weekly_messages: weekly_messages[key] = []
                weekly_messages[key].append(msg)
        except: continue

    results = []
    # Sort weeks
    sorted_weeks = sorted(weekly_messages.items())

    # Limit to last 8 weeks to prevent hitting daily limits if history is huge
    if len(sorted_weeks) > 8:
        logger.info(f"Limiting summary to last 8 weeks (Total weeks: {len(sorted_weeks)})")
        sorted_weeks = sorted_weeks[-8:]

    for i, (week_start, msgs) in enumerate(sorted_weeks):
        summary = generate_total_summary(msgs)
        results.append({
            'week_start': week_start,
            'message_count': len(msgs),
            'summary': summary
        })

        # *** CRITICAL: SLEEP BETWEEN REQUESTS ***
        # If this is not the last item, wait 4 seconds to respect rate limits
        if i < len(sorted_weeks) - 1:
            time.sleep(4)

    return results


def generate_total_summary(messages):
    if not messages: return "No messages."

    # Filter and limit context
    filtered = [m for m in messages if 'media omitted' not in m['message'].lower()]
    sample = filtered[:150] # Limit to conserve tokens
    chat_text = "\n".join([f"{m['sender']}: {m['message']}" for m in sample])

    prompt = f"""
    Summarize this WhatsApp chat briefly.
    1. Main Topic
    2. Key Decisions
    3. Action Items

    Chat:
    {chat_text}
    """
    return generate_with_gemini(prompt)

def generate_brief_summary(messages):
    # Same logic as before
    if not messages: return "No messages."
    chat_text = "\n".join([f"{m['sender']}: {m['message']}" for m in messages[:150]])

    prompt = f"""
    Create an HTML summary (<h3>Header</h3>, <ul><li>Item</li></ul>).
    Sections: Overview, Topics, Actions.
    Chat: {chat_text}
    """
    return generate_with_gemini(prompt)


def generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range):
    """Generate a fallback brief summary when AI is unavailable"""
    summary_parts = []

    # Overview
    summary_parts.append(f"📊 **CONVERSATION OVERVIEW**")
    summary_parts.append(f"Total Messages: {total_messages} from {user_count} participants over {date_range} days")
    summary_parts.append("")

    # Key Participants
    if most_active_user:
        summary_parts.append(f"👥 **KEY PARTICIPANTS**")
        summary_parts.append(f"Most Active: {most_active_user[0]} with {most_active_user[1]} messages")
        summary_parts.append("")

    # Activity Patterns
    summary_parts.append(f"⏰ **ACTIVITY PATTERNS**")
    summary_parts.append(f"Peak Activity: {peak_hour if peak_hour is not None else 'N/A'}:00 hours on {peak_day if peak_day else 'N/A'}")
    summary_parts.append("")

    # Main Discussion Topics
    summary_parts.append(f"💬 **MAIN DISCUSSION TOPICS**")
    if messages:
        # Show first few messages as examples
        for i, msg in enumerate(messages[:5]):
            summary_parts.append(f"- {msg['sender']}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}")
    summary_parts.append("")

    # Important Resources
    summary_parts.append(f"📁 **IMPORTANT RESOURCES**")
    summary_parts.append(f"Files Shared: {len(file_shares)} | Links Shared: {len(links)}")
    summary_parts.append("")

    # Actionable Insights
    summary_parts.append(f"✅ **ACTIONABLE INSIGHTS**")
    summary_parts.append(f"Decisions Made: {len(decisions)} | Action Items: {len(action_items)} | Meetings Planned: {len(meetings)}")

    return "\n".join(summary_parts)

def generate_daily_user_messages(messages):
    """Generate daily summaries grouped by user"""
    if not messages:
        return []

    # Group messages by date and user
    daily_user_messages = {}

    for msg in messages:
        try:
            timestamp = parse_timestamp(msg['timestamp'])
            if timestamp:
                date_key = timestamp.strftime('%Y-%m-%d')
                user = msg['sender']

                if date_key not in daily_user_messages:
                    daily_user_messages[date_key] = {}

                if user not in daily_user_messages[date_key]:
                    daily_user_messages[date_key][user] = []

                daily_user_messages[date_key][user].append(msg)
        except Exception as e:
            logger.warning(f"Error parsing timestamp: {e}")
            continue

    # Generate summary for each day
    daily_summaries = []
    for date, user_messages in sorted(daily_user_messages.items()):
        summary_parts = [f"**{date}**"]
        for user, messages in user_messages.items():
            summary_parts.append(f"- {user}: {len(messages)} messages")

        daily_summaries.append({
            'date': date,
            'message_count': sum(len(msgs) for msgs in user_messages.values()),
            'summary': "\n".join(summary_parts)
        })

    return daily_summaries

def generate_user_wise_detailed_report(messages, target_user):
    """
    Filters messages for a specific user.
    Returns: List of message objects for that user.
    """
    if not messages or not target_user:
        return []

    user_msgs = []
    for msg in messages:
        if msg.get('sender') == target_user:
            user_msgs.append({
                'timestamp': msg.get('timestamp', ''),
                'message': msg.get('message', ''),
                'sender': target_user
            })

    return user_msgs

def generate_comprehensive_summary(messages, start_date_str=None, end_date_str=None):
    """Generate a comprehensive summary combining multiple analysis types"""
    if not messages:
        return {
            'brief_summary': "No messages found in the selected date range.",
            'weekly_summaries': []
        }

    # Generate brief summary
    brief_summary = generate_brief_summary(messages)

    # Generate weekly summaries
    weekly_summaries = generate_weekly_summary(messages, start_date_str, end_date_str)

    return {
        'brief_summary': brief_summary,
        'weekly_summaries': weekly_summaries
    }

def calculate_date_range(messages):
    """Calculate the number of days between first and last message"""
    if not messages:
        return 0
    try:
        d1 = parse_timestamp(messages[0]['timestamp'])
        d2 = parse_timestamp(messages[-1]['timestamp'])
        if d1 and d2:
            return abs((d2 - d1).days) + 1
    except:
        pass
    return 1

def generate_fallback_request(prompt, api_key):
    """Fallback to 1.5 Flash if 2.0 is forbidden"""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass
    return "Error: Fallback model failed."
import time
def generate_with_gemini(prompt, max_retries=3):
    """
    Generate content with automatic retry for 429 (Quota Exceeded) errors.
    """
    api_key = os.getenv('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)

    if not api_key:
        logger.error("GEMINI_API_KEY not found.")
        return "Error: API Key missing."

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048}
    }

    # RETRY LOOP
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                params={"key": api_key},
                json=data,
                timeout=60
            )

            # CASE 1: QUOTA EXCEEDED (429)
            if response.status_code == 429:
                if attempt < max_retries:
                    # Exponential Backoff: Wait 2s, 4s, 8s...
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Quota exceeded (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Daily/Minute Quota Exceeded. Please try again later."

            # CASE 2: PERMISSION ERROR (403)
            if response.status_code == 403:
                logger.error(f"403 Forbidden on {MODEL_NAME}. Trying fallback...")
                return generate_fallback_request(prompt, api_key)

            # CASE 3: SUCCESS (200)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Error: Empty response from AI."

            # Other errors
            logger.error(f"API Error {response.status_code}: {response.text}")
            return f"Error: API returned {response.status_code}"

        except Exception as e:
            logger.error(f"Request failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return "Error: Connection failed."

    return "Error: Maximum retries exceeded."

def generate_fallback_summary(messages):
    """Simple non-AI summary"""
    count = len(messages)
    senders = set(m['sender'] for m in messages)
    return f"**System Summary**: Processed {count} messages from {len(senders)} participants. AI Service is currently unavailable."

def generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range):
    """Generate a fallback brief summary when AI is unavailable"""
    summary_parts = []
    summary_parts.append(f"<h2 style='color:green;'>📊 CONVERSATION OVERVIEW</h2>")
    summary_parts.append(f"Total Messages: {total_messages} from {user_count} participants over {date_range} days")

    if most_active_user:
        summary_parts.append(f"<h2 style='color:green;'>👥 KEY PARTICIPANTS</h2>")
        summary_parts.append(f"Most Active: {most_active_user[0]} ({most_active_user[1]} messages)")

    summary_parts.append(f"<h2 style='color:green;'>💬 RECENT MESSAGES</h2>")
    if messages:
        for msg in messages[:5]:
             summary_parts.append(f"- {msg['sender']}: {msg['message'][:100]}")

    return "\n".join(summary_parts)


def generate_structured_summary(messages):
    """Generates a JSON summary"""
    chat_text = str(messages[:200]) # Limit for safety

    prompt = (
        "You are an expert WhatsApp chat analyst. "
        "Return strictly valid JSON. No Markdown formatting."
        "Properties: activity_summary (string), key_topics (array), "
        "notable_events (array), social_dynamics (string), recommended_actions (array).\n\n"
        f"Messages: {chat_text}"
    )

    response = generate_with_gemini(prompt)

    # Clean up markdown code blocks if the AI adds them
    if response and "```json" in response:
        response = response.replace("```json", "").replace("```", "")

    return response

# Generate answer to specific questions
def generate_question_answer(messages, question):
    """Generate an answer to a specific question based on chat messages"""
    try:
        # Prepare the chat context
        chat_context = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in messages[:200]])  # Limit to 200 messages

        # Create a prompt for question answering with enhanced instructions
        prompt = f"""You are an expert WhatsApp chat analyzer. Based on the following WhatsApp chat conversation, please answer the question accurately and comprehensively.

Conversation context:
{chat_context}

Question: {question}

Instructions:
1. Analyze the conversation context carefully to find relevant information
2. Provide a clear, concise, and accurate answer based on the conversation
3. If the information is not available in the conversation, state that clearly
4. For questions about user activity, message counts, or statistics, provide specific numbers when available
5. For time-based questions, reference specific timestamps when relevant
6. Format your response clearly with appropriate headings and bullet points when needed

Please provide your answer:"""

        # Use the existing generate_with_gemini function
        response = generate_with_gemini(prompt)

        # Check for quota or API errors
        if response == "QUOTA_EXCEEDED":
            # Fallback answer using pattern matching
            return generate_fallback_answer(question, messages)
        elif response == "API_ERROR":
            return "Unable to generate answer due to technical issues. Please try again later."
        else:
            return response

    except Exception as e:
        logger.error(f"Error generating question answer: {e}")
        return {"status": "error", "message": str(e)}


def generate_fallback_answer(question, messages):
    """Generate a fallback answer when AI is unavailable"""
    if not messages:
        return "I don't have any messages to analyze for this question."

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
    least_active_user = min(user_msg_count.items(), key=lambda x: x[1]) if user_msg_count else None

    # Extract meaningful content and filter system messages
    meaningful_messages = []
    meeting_messages = []
    file_messages = []
    topic_messages = []

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
            if any(word in message_lower for word in ['meet', 'meeting', 'call', 'zoom', 'teams', 'hangout', 'discuss', 'schedule']):
                meeting_messages.append(msg)

            # Look for file/document sharing
            if any(ext in message_lower for ext in ['.pdf', '.doc', '.jpg', '.png', '.mp4', '.xlsx', '.jpeg', '.docx']):
                file_messages.append(msg)

            # Collect other substantial content
            if len(message_text) > 30:
                topic_messages.append(msg)

    # Handle different types of questions with improved pattern matching

    # User-specific message count questions
    if any(word in question_lower for word in ['how many', 'count', 'number of']) and 'message' in question_lower:
        # Extract user name from question (simple approach)
        # This is a basic implementation - in practice, you might want to use NLP for better entity extraction
        user_name = None
        # Look for common user name patterns
        for user in user_msg_count.keys():
            # Check if user name appears in the question
            if user.lower() in question_lower:
                user_name = user
                break

        if user_name:
            user_message_count = count_user_messages(messages, user_name)
            return f"📊 **Message Statistics for {user_name}:**\n\n• **Messages Sent**: {user_message_count}\n• **Percentage of Total**: {round((user_message_count/total_messages)*100, 1)}%"
        else:
            # Fall back to general statistics
            answer = f"📊 **Message Statistics:**\n\n"
            answer += f"• **Total Messages**: {total_messages}\n"
            answer += f"• **Total Users**: {user_count}\n"
            answer += f"• **Date Range**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
            answer += f"• **Average per User**: {round(total_messages/user_count, 1)} messages\n"

            # Add most and least active users
            if most_active_user:
                answer += f"• **Most Active User**: {most_active_user[0]} ({most_active_user[1]} messages)\n"
            if least_active_user:
                answer += f"• **Least Active User**: {least_active_user[0]} ({least_active_user[1]} messages)\n"

            return answer

    # Meeting-related questions
    if any(word in question_lower for word in ['meet', 'meeting', 'call', 'schedule', 'appointment']):
        if meeting_messages:
            answer = "📅 **Meetings Found:**\n\n"
            for i, msg in enumerate(meeting_messages[:5], 1):  # Show up to 5 meetings
                meeting_content = msg['message'][:200] + "..." if len(msg['message']) > 200 else msg['message']
                answer += f"**{i}. Meeting on {msg['timestamp']}**\n"
                answer += f"👤 Organized by: {msg['sender']}\n"
                answer += f"📝 Details: {meeting_content}\n\n"
            return answer
        else:
            return "No meetings found in the conversation history."

    # Most active user questions
    elif (any(word in question_lower for word in ['most active', 'top user', 'highest activity']) or
          (any(word in question_lower for word in ['who', 'active user']) and 'most' in question_lower)) and 'least' not in question_lower:
        if most_active_user:
            # Show top 3 users
            sorted_users = sorted(user_msg_count.items(), key=lambda x: x[1], reverse=True)
            answer = "👥 **Most Active Users:**\n\n"
            for i, (user, count) in enumerate(sorted_users[:3], 1):
                percentage = round((count/total_messages)*100, 1)
                answer += f"**{i}. {user}**: {count} messages ({percentage}%)\n"
            return answer
        else:
            return "Unable to determine user activity from the available data."

    # Least active user questions
    elif (any(word in question_lower for word in ['least active', 'lowest activity', 'inactive']) or
          (any(word in question_lower for word in ['who', 'active user']) and 'least' in question_lower)):
        if least_active_user:
            # Show bottom 3 users (sorted by message count ascending)
            sorted_users = sorted(user_msg_count.items(), key=lambda x: x[1])
            answer = "👥 **Least Active Users:**\n\n"
            for i, (user, count) in enumerate(sorted_users[:3], 1):
                percentage = round((count/total_messages)*100, 1)
                answer += f"**{i}. {user}**: {count} messages ({percentage}%)\n"
            return answer
        else:
            return "Unable to determine user activity from the available data."

    # General statistics questions
    elif any(word in question_lower for word in ['how many', 'total', 'messages', 'count', 'number of', 'statistics', 'stats']):
        answer = f"📊 **Message Statistics:**\n\n"
        answer += f"• **Total Messages**: {total_messages}\n"
        answer += f"• **Total Users**: {user_count}\n"
        answer += f"• **Date Range**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
        answer += f"• **Average per User**: {round(total_messages/user_count, 1)} messages\n"

        # Add most and least active users
        if most_active_user:
            answer += f"• **Most Active User**: {most_active_user[0]} ({most_active_user[1]} messages)\n"
        if least_active_user:
            answer += f"• **Least Active User**: {least_active_user[0]} ({least_active_user[1]} messages)\n"

        return answer

    # File/document sharing questions
    elif any(word in question_lower for word in ['file', 'document', 'pdf', 'shared', 'share', 'attachment']):
        if file_messages:
            answer = "📎 **Files/Documents Shared:**\n\n"
            for i, msg in enumerate(file_messages[:5], 1):
                answer += f"**{i}. {msg['timestamp']}**\n"
                answer += f"👤 Shared by: {msg['sender']}\n"
                answer += f"📄 File: {msg['message'][:100]}...\n\n"
            return answer
        else:
            return "No files or documents were shared."

    # Time range questions
    elif any(word in question_lower for word in ['show me', 'messages on', 'from', 'to', 'between']):
        # This is a basic implementation - in practice, you would want to parse dates properly
        # For now, we'll just indicate that time filtering should be done in the UI
        return "Please use the date filters in the UI to view messages for specific time ranges."

    # Specific date questions (e.g., "on 11/04/2024")
    elif re.search(r'\b(on|for)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', question_lower):
        # Extract date from question
        date_match = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', question_lower)
        if date_match and messages:
            requested_date = date_match.group()
            # Format the date to match the message timestamps
            formatted_messages = []
            for msg in messages:
                # Check if the message timestamp contains the requested date
                if requested_date.replace('/', '-') in msg['timestamp'] or requested_date.replace('-', '/') in msg['timestamp']:
                    formatted_messages.append(msg)

            if formatted_messages:
                answer = f"📝 **Messages on {requested_date}:**\n\n"
                # Group messages by sender
                sender_messages = {}
                for msg in formatted_messages:
                    sender = msg['sender']
                    if sender not in sender_messages:
                        sender_messages[sender] = []
                    sender_messages[sender].append(msg)

                # List senders and their messages
                for sender, sender_msgs in sender_messages.items():
                    answer += f"**{sender}:**\n"
                    for msg in sender_msgs:
                        answer += f"  • {msg['message']}\n"
                    answer += "\n"
                return answer
            else:
                return f"No messages found for {requested_date}."
        else:
            return "Unable to parse the date from your question."

    # Time range questions (e.g., "from 3 pm to 8 pm")
    elif 'pm' in question_lower and ('from' in question_lower or 'between' in question_lower):
        # This would require more sophisticated time parsing
        return "I can see you're asking for a specific time range. Please use the time filters in the UI for more accurate results."

    # Topic/content questions
    elif any(word in question_lower for word in ['topic', 'discuss', 'about', 'content', 'summary', 'talk about', 'conversation']):
        if topic_messages:
            answer = "💬 **Main Discussion Topics:**\n\n"
            # Group messages by sender to show diverse content
            user_topics = {}
            for msg in topic_messages[:15]:
                user = msg['sender']
                if user not in user_topics:
                    user_topics[user] = []
                if len(user_topics[user]) < 4:  # Increased to 4 topics per user for better coverage
                    content = msg['message'][:120] + "..." if len(msg['message']) > 120 else msg['message']
                    user_topics[user].append({
                        'content': content,
                        'timestamp': msg['timestamp']
                    })

            topic_count = 1
            for user, topics in list(user_topics.items())[:min(8, len(user_topics))]:  # Dynamically show users based on content
                for topic in topics:
                    answer += f"**{topic_count}. {topic['timestamp']}**\n"
                    answer += f"👤 {user}: {topic['content']}\n\n"
                    topic_count += 1
                    if topic_count > 20:  # Increased to 20 topics total for better content coverage
                        break
                if topic_count > 20:
                    break

            return answer
        else:
            return "The conversation appears to contain mostly brief exchanges."

    else:
        # General fallback with comprehensive overview
        answer = "📋 **Chat Overview:**\n\n"
        answer += f"• **{total_messages} messages** from **{user_count} users**\n"
        answer += f"• **Time Period**: {messages[0]['timestamp']} to {messages[-1]['timestamp']}\n"
        if meeting_messages:
            answer += f"• **{len(meeting_messages)} meetings** mentioned\n"
        if file_messages:
            answer += f"• **{len(file_messages)} files** shared\n"

        # Add user activity information
        if most_active_user:
            percentage = round((most_active_user[1]/total_messages)*100, 1)
            answer += f"• **Most Active User**: {most_active_user[0]} ({most_active_user[1]} messages, {percentage}% of total)\n"
        if least_active_user:
            percentage = round((least_active_user[1]/total_messages)*100, 1)
            answer += f"• **Least Active User**: {least_active_user[0]} ({least_active_user[1]} messages, {percentage}% of total)\n"

        return answer

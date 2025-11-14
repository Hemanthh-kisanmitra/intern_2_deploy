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
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}/generateContent"

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

def generate_fallback_summary(messages):
    """Generate structured summary with actual message content when AI is unavailable"""
    if not messages:
        return "**ACTIVITY OVERVIEW**: No messages during this week\n**MAIN DISCUSSION TOPICS**: No conversations recorded"

    # Basic statistics
    total_messages = len(messages)
    users = set(msg['sender'] for msg in messages)
    user_count = len(users)

    # Most active user
    user_msg_count = {}
    for msg in messages:
        user = msg['sender']
        user_msg_count[user] = user_msg_count.get(user, 0) + 1

    most_active_user = max(user_msg_count.items(), key=lambda x: x[1]) if user_msg_count else None

    # Extract actual message content (not system messages)
    actual_messages = []
    file_names = []
    conversation_snippets = []

    for msg in messages:
        message_text = msg['message']
        message_lower = message_text.lower()

        # Skip system messages
        if any(term in message_lower for term in ['media omitted', 'security code changed', 'tap to learn more', 'this message was deleted', 'messages and calls are end-to-end encrypted']):
            continue

        # Look for file names and documents
        if any(ext in message_lower for ext in ['.pdf', '.doc', '.jpg', '.png', '.mp4', '.xlsx']):
            file_names.append(message_text.strip())

        # Collect meaningful conversation content
        if len(message_text.strip()) >= 10:  # Meaningful messages
            conversation_snippets.append(f"{msg['sender']}: {message_text.strip()[:100]}..." if len(message_text) > 100 else f"{msg['sender']}: {message_text.strip()}")
            actual_messages.append(message_text)

    # Build structured summary with actual content
    summary_parts = []

    # Activity Overview
    summary_parts.append(f"**ACTIVITY OVERVIEW**: {total_messages} messages from {user_count} participants during this week")

    # Key Participants
    if most_active_user:
        percentage = round((most_active_user[1] / total_messages) * 100)
        summary_parts.append(f"**KEY PARTICIPANTS**: {most_active_user[0]} was most active with {most_active_user[1]} messages ({percentage}% of activity)")

    # Main Discussion Topics with actual content
    if file_names or conversation_snippets:
        summary_parts.append("**MAIN DISCUSSION TOPICS**:")

        topic_count = 1
        # Add file/document sharing topics
        for file_name in file_names[:3]:  # Show up to 3 files
            summary_parts.append(f"- Topic {topic_count}: Document shared - \"{file_name}\"")
            topic_count += 1

        # Add conversation content from ALL participants, not just most active
        participant_messages = {}
        for snippet in conversation_snippets[:15]:  # Get more messages
            participant = snippet.split(':')[0]
            if participant not in participant_messages:
                participant_messages[participant] = []
            participant_messages[participant].append(snippet)

        # Distribute topics across different participants dynamically based on content
        max_topics = min(len(participant_messages), 10)  # Dynamically adjust based on participants, up to 10
        topic_index = 1
        for participant, messages in list(participant_messages.items()):
            if topic_index <= max_topics:
                sample_msg = messages[0] if messages else f"{participant}: [No detailed message]"
                summary_parts.append(f"- Topic {topic_index}: {sample_msg}")
                topic_index += 1
    else:
        # If no meaningful content found, show the actual raw messages to debug
        if len(messages) > 0:
            sample_messages = []
            for i, msg in enumerate(messages[:5]):  # Show first 5 messages for debugging
                sample_messages.append(f"{msg['sender']}: {msg['message'][:100]}")

            summary_parts.append("**MAIN DISCUSSION TOPICS**:")
            for i, sample in enumerate(sample_messages, 1):
                summary_parts.append(f"- Topic {i}: {sample}")
        else:
            summary_parts.append("**MAIN DISCUSSION TOPICS**: No messages found")

    # Social Dynamics with actual interaction content
    if user_count > 1 and conversation_snippets:
        summary_parts.append(f"**SOCIAL DYNAMICS**: Active interaction among {user_count} participants with meaningful exchanges")
        if len(conversation_snippets) >= 2:
            # Show actual conversation examples
            summary_parts.append(f"- Example interaction: {conversation_snippets[0]}")
            if len(conversation_snippets) > 1:
                summary_parts.append(f"- Follow-up: {conversation_snippets[1]}")

    return "\n".join(summary_parts)

def generate_total_summary(messages):
    """Generate a comprehensive summary of all messages"""
    if not messages:
        return "No messages to summarize."

    # Filter out system messages
    filtered_messages = []
    for msg in messages:
        message_lower = msg['message'].lower()
        if not any(term in message_lower for term in ['media omitted', 'security code changed', 'tap to learn more', 'this message was deleted']):
            filtered_messages.append(msg)

    if not filtered_messages:
        return "No meaningful messages to summarize."

    # Use AI if available
    if model:
        try:
            prompt = f"""
            Please provide a comprehensive summary of the following WhatsApp chat messages.
            Include:
            1. Overall activity level
            2. Key participants and their activity
            3. Main topics discussed
            4. Important events or decisions
            5. Overall sentiment

            Messages:
            """

            # Add messages to prompt (limit to avoid token limits)
            for msg in filtered_messages[:100]:  # Limit to 100 messages
                prompt += f"\n{msg['timestamp']} - {msg['sender']}: {msg['message']}"

            response = model.generate_content(prompt)
            if response.text:
                return response.text

        except Exception as e:
            logger.warning(f"AI summary failed: {e}")

    # Fallback to structured summary
    return generate_fallback_summary(filtered_messages)

def generate_user_messages(messages):
    """Generate a summary of messages grouped by user"""
    if not messages:
        return {}

    user_messages = {}
    for msg in messages:
        sender = msg['sender']
        if sender not in user_messages:
            user_messages[sender] = []
        user_messages[sender].append(msg)

    return user_messages

def get_users_in_messages(messages):
    """Get a list of all users in the messages"""
    users = set()
    for msg in messages:
        users.add(msg['sender'])
    return sorted(list(users))

def generate_user_messages_for_user(messages, user):
    """Generate messages for a specific user"""
    user_messages = []
    for msg in messages:
        if msg['sender'] == user:
            user_messages.append(msg)
    return user_messages

def generate_weekly_summary(messages, start_date_str=None, end_date_str=None):
    """Generate weekly summaries"""
    if not messages:
        return []

    # Group messages by week
    weekly_messages = {}

    for msg in messages:
        try:
            timestamp = parse_timestamp(msg['timestamp'])
            if timestamp:
                # Get the Monday of the week as the key
                monday = timestamp - timedelta(days=timestamp.weekday())
                week_key = monday.strftime('%Y-%m-%d')

                if week_key not in weekly_messages:
                    weekly_messages[week_key] = []
                weekly_messages[week_key].append(msg)
        except Exception as e:
            logger.warning(f"Error parsing timestamp: {e}")
            continue

    # Generate summary for each week
    weekly_summaries = []
    for week_start, week_messages in sorted(weekly_messages.items()):
        week_end = (datetime.strptime(week_start, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')
        summary = generate_total_summary(week_messages)
        weekly_summaries.append({
            'week_start': week_start,
            'week_end': week_end,
            'message_count': len(week_messages),
            'summary': summary
        })

    return weekly_summaries

def generate_brief_summary(messages):
    """Generate a comprehensive brief summary with actionable insights for decision making"""
    if not messages:
        return "No messages found in the selected date range."

    print(f"Generating brief summary for {len(messages)} messages")

    # Limit message count to prevent memory issues (more aggressive limit)
    max_messages = 200
    if len(messages) > max_messages:
        messages = messages[:max_messages]
        print(f"Limited messages to {len(messages)} for memory management")

    # Basic statistics for enhanced insights
    total_messages = len(messages)
    users = set(msg['sender'] for msg in messages)
    user_count = len(users)

    # Most active user
    user_msg_count = {}
    for msg in messages:
        user = msg['sender']
        user_msg_count[user] = user_msg_count.get(user, 0) + 1

    most_active_user = max(user_msg_count.items(), key=lambda x: x[1]) if user_msg_count else None

    # Calculate activity patterns
    hourly_activity = {}
    daily_activity = {}
    for msg in messages:
        dt = parse_timestamp(msg['timestamp'])
        if dt:
            hour = dt.hour
            day = dt.strftime('%A')
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            daily_activity[day] = daily_activity.get(day, 0) + 1

    peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None
    peak_day = max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None

    # Enhanced analysis for short periods (7 days or less)
    date_range = calculate_date_range(messages)
    is_short_period = date_range <= 7

    # Extract detailed information
    file_shares = []
    links = []
    meetings = []
    decisions = []
    action_items = []
    questions = []
    announcements = []
    technical_discussions = []

    for msg in messages:
        message_text = msg['message'].lower()
        original_text = msg['message']
        sender = msg['sender']

        # Check for file shares
        if any(ext in message_text for ext in ['.pdf', '.doc', '.jpg', '.png', '.mp4', '.xlsx', '.docx', '.pptx']):
            file_shares.append(f"{sender}: {original_text}")

        # Check for links
        if 'http' in message_text or 'www.' in message_text:
            links.append(f"{sender}: {original_text}")

        # Check for meeting related keywords
        if any(keyword in message_text for keyword in ['meeting', 'call', 'zoom', 'teams', 'hangout', 'discuss', 'schedule', 'मिटिंग', 'दौरा']):
            meetings.append(f"{sender}: {original_text}")

        # Check for decision keywords
        if any(keyword in message_text for keyword in ['decided', 'decision', 'final', 'agreed']):
            decisions.append(f"{sender}: {original_text}")

        # Check for action items
        if any(keyword in message_text for keyword in ['need to', 'should', 'must', 'todo', 'action', 'complete', 'finish', 'करावे', 'करायचे']):
            action_items.append(f"{sender}: {original_text}")

        # Check for questions
        if '?' in original_text or any(word in message_text for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'काय', 'कसे', 'केव्हा']):
            questions.append(f"{sender}: {original_text}")

        # Check for announcements
        if any(word in message_text for word in ['announce', 'notice', 'alert', 'अलर्ट', 'सूचना', 'जाहिरात']):
            announcements.append(f"{sender}: {original_text}")

        # Check for technical discussions
        if any(word in message_text for word in ['technical', 'method', 'process', 'procedure', 'technique', 'तंत्रज्ञान', 'पद्धत']):
            technical_discussions.append(f"{sender}: {original_text}")

    try:
        # Create enhanced prompt based on period length
        chat_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in messages[:200]])

        if is_short_period:
            # Enhanced analysis for short periods
            comprehensive_prompt = f"""Analyze this WhatsApp group conversation from the last {date_range} days and create a DETAILED brief summary with specific insights and actionable information.

**CRITICAL INSTRUCTIONS FOR SHORT PERIOD ANALYSIS**:
1. Extract EXACT quotes and specific content from messages
2. Identify specific problems, solutions, or advice mentioned
3. Show actual conversation flow and responses between participants
4. Include specific names, dates, times, and locations mentioned
5. Highlight any urgent or important information shared
6. Show what each participant contributed specifically

**REQUIRED STRUCTURE**:
<h2 style='color:green;'>📊 CONVERSATION OVERVIEW</h2>
==Total Messages: {total_messages} from {user_count} participants over {date_range} days==

<h2 style='color:green;'>👥 KEY PARTICIPANTS & CONTRIBUTIONS</h2>
==Most Active: {most_active_user[0] if most_active_user else 'N/A'} with {most_active_user[1] if most_active_user else 0} messages==
==Show what each participant specifically contributed with actual quotes==

<h2 style='color:green;'>⏰ ACTIVITY PATTERNS</h2>
==Peak Activity: {peak_hour if peak_hour is not None else 'N/A'}:00 hours on {peak_day if peak_day else 'N/A'}==
==Show specific times when important conversations happened==

<h2 style='color:green;'>💬 MAIN DISCUSSION TOPICS (with actual quotes)</h2>
==Provide key topics with EXACT quotes from messages in original language==
==Show the actual conversation flow and responses==Dynamic number of topics based on content diversity==

<h2 style='color:green;'>📁 IMPORTANT RESOURCES SHARED</h2>
==Files Shared: {len(file_shares)} | Links Shared: {len(links)}==
==List specific files and links mentioned with who shared them==

<h2 style='color:green;'>❓ QUESTIONS ASKED</h2>
==Show actual questions asked with who asked them and any answers provided==

<h2 style='color:green;'>📢 ANNOUNCEMENTS & ALERTS</h2>
==List specific announcements with exact content and who made them==

<h2 style='color:green;'>🔧 TECHNICAL DISCUSSIONS</h2>
==Show any technical advice, methods, or procedures discussed==

<h2 style='color:green;'>✅ ACTIONABLE INSIGHTS</h2>
==Decisions Made: {len(decisions)} | Action Items: {len(action_items)} | Meetings Planned: {len(meetings)}==
==Show specific decisions and action items with who mentioned them==

<h2 style='color:green;'>🎯 IMMEDIATE NEXT STEPS</h2>
==Based on the conversation, what should be done next?==

Conversation content:
{chat_text}"""
        else:
            # Standard analysis for longer periods
            comprehensive_prompt = f"""Analyze this WhatsApp group conversation and create a comprehensive brief summary that provides actionable insights for decision making.

**REQUIRED STRUCTURE**:
<h2 style='color:green;'>CONVERSATION OVERVIEW</h2>
==Total Messages: {total_messages} from {user_count} participants==

<h2 style='color:green;'>KEY PARTICIPANTS</h2>
==Most Active: {most_active_user[0] if most_active_user else 'N/A'} with {most_active_user[1] if most_active_user else 0} messages==

<h2 style='color:green;'>ACTIVITY PATTERNS</h2>
==Peak Activity: {peak_hour if peak_hour is not None else 'N/A'}:00 hours on {peak_day if peak_day else 'N/A'}==

<h2 style='color:green;'>MAIN DISCUSSION TOPICS</h2>
==Provide key topics discussed with brief descriptions==
==Dynamic number of topics based on content variety==

<h2 style='color:green;'>IMPORTANT RESOURCES</h2>
==Files Shared: {len(file_shares)} | Links Shared: {len(links)}==

<h2 style='color:green;'>ACTIONABLE INSIGHTS</h2>
==Decisions Made: {len(decisions)} | Action Items: {len(action_items)} | Meetings Planned: {len(meetings)}==

<h2 style='color:green;'>RECOMMENDATIONS</h2>
==Provide 2-3 actionable recommendations based on the conversation==

Conversation content:
{chat_text}"""

        response = generate_with_gemini(comprehensive_prompt)
        print(f"AI response received: {response[:100]}...")

        # Check if API quota exceeded or error occurred
        if response == "QUOTA_EXCEEDED":
            # Enhanced fallback summary with actual content for brief summary
            print("Gemini API quota exceeded, using fallback summary")
            return generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range)

        elif response == "API_ERROR":
            # Use fallback summary when API is unavailable
            return generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range)

        return response

    except Exception as e:
        logger.error(f"Error in generate_brief_summary: {e}")
        # Fallback to structured summary when AI fails
        return generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range)

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

def generate_user_wise_detailed_report(messages, user):
    """Generate a detailed report for a specific user"""
    if not messages:
        return []

    user_messages = [msg for msg in messages if msg['sender'] == user]

    if not user_messages:
        return f"No messages found for user {user}."

    # Group messages by date
    daily_messages = {}
    for msg in user_messages:
        try:
            timestamp = parse_timestamp(msg['timestamp'])
            if timestamp:
                date_key = timestamp.strftime('%Y-%m-%d')
                if date_key not in daily_messages:
                    daily_messages[date_key] = []
                daily_messages[date_key].append(msg)
        except Exception as e:
            logger.warning(f"Error parsing timestamp: {e}")
            continue

    # Generate report
    report_parts = [f"**Detailed Report for {user}**"]
    report_parts.append(f"Total messages: {len(user_messages)}")
    report_parts.append("")

    for date, messages in sorted(daily_messages.items()):
        report_parts.append(f"**{date}** ({len(messages)} messages):")
        for msg in messages:
            report_parts.append(f"- {msg['message']}")
        report_parts.append("")

    return "\n".join(report_parts)

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

    dates = []
    for msg in messages:
        dt = parse_timestamp(msg['timestamp'])
        if dt:
            dates.append(dt.date())

    if len(dates) < 2:
        return 1

    return (max(dates) - min(dates)).days + 1

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

        return generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages, questions, announcements, technical_discussions, date_range)

def generate_fallback_brief_summary(total_messages, user_count, most_active_user, peak_hour, peak_day, file_shares, links, meetings, decisions, action_items, messages=None, questions=None, announcements=None, technical_discussions=None, date_range=None):
    """Generate an enhanced fallback brief summary when AI is unavailable"""
    fallback_parts = []

    # Determine if this is a short period for enhanced analysis
    is_short_period = date_range and date_range <= 7

    # Overview with actual conversation analysis
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>📊 CONVERSATION OVERVIEW</h2>")
        fallback_parts.append(f"==Total Messages: {total_messages} from {user_count} participants over {date_range} days==")
    else:
        fallback_parts.append("<h2 style='color:green;'>CONVERSATION OVERVIEW</h2>")
        fallback_parts.append(f"==Total Messages: {total_messages} from {user_count} participants==")

    # Most active user with their contributions
    if most_active_user:
        if is_short_period:
            fallback_parts.append("<h2 style='color:green;'>👥 KEY PARTICIPANTS & CONTRIBUTIONS</h2>")
            fallback_parts.append(f"==Most Active: {most_active_user[0]} with {most_active_user[1]} messages ({round((most_active_user[1]/total_messages)*100)}% of activity)==")
            # Add actual message content for short periods
            if messages:
                user_messages = [msg for msg in messages if msg['sender'] == most_active_user[0]]
                if user_messages:
                    fallback_parts.append("==Recent contributions from most active participant:==")
                    for i, msg in enumerate(user_messages[:3], 1):
                        content = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
                        fallback_parts.append(f"=={i}. \"{content}\"==")
        else:
            fallback_parts.append("<h2 style='color:green;'>KEY PARTICIPANTS</h2>")
            fallback_parts.append(f"==Most Active: {most_active_user[0]} with {most_active_user[1]} messages ({round((most_active_user[1]/total_messages)*100)}% of activity)==")

    # Activity patterns
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>⏰ ACTIVITY PATTERNS</h2>")
    else:
        fallback_parts.append("<h2 style='color:green;'>ACTIVITY PATTERNS</h2>")

    if peak_hour is not None:
        hour_12 = peak_hour % 12 or 12
        am_pm = "AM" if peak_hour < 12 else "PM"
        fallback_parts.append(f"==Peak Activity: {hour_12}:00 {am_pm} on {peak_day if peak_day else 'N/A'}==")
        if is_short_period and messages:
            # Show specific times when messages were sent
            times = []
            for msg in messages:
                dt = parse_timestamp(msg['timestamp'])
                if dt:
                    time_str = dt.strftime('%I:%M %p')
                    if time_str not in times:
                        times.append(time_str)
            if times:
                fallback_parts.append(f"==Messages sent at: {', '.join(times[:5])}==")
    else:
        fallback_parts.append("==No significant activity patterns identified==")

    # Main topics - extract from actual messages with more detail for short periods
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>💬 MAIN DISCUSSION TOPICS (with actual quotes)</h2>")
        if messages:
            # Extract actual topics from messages
            topics = []
            for msg in messages[:5]:  # Show first 5 messages as topics
                content = msg['message'][:80] + "..." if len(msg['message']) > 80 else msg['message']
                topics.append(f"=={msg['sender']}: \"{content}\"==")
            if topics:
                fallback_parts.extend(topics)
            else:
                fallback_parts.append("==No substantial conversation topics identified==")
        else:
            fallback_parts.append("==No conversation content available==")
    else:
        fallback_parts.append("<h2 style='color:green;'>MAIN DISCUSSION TOPICS</h2>")
        if file_shares or links or meetings:
            topic_points = []
            if file_shares:
                topic_points.append(f"Document sharing: {len(file_shares)} files shared")
            if links:
                topic_points.append(f"Resource sharing: {len(links)} links shared")
            if meetings:
                topic_points.append(f"Coordination: {len(meetings)} meeting-related discussions")
            fallback_parts.append(f"=={'; '.join(topic_points)}==")
        else:
            fallback_parts.append("==General group conversation==")

    # Questions asked (for short periods)
    if is_short_period and questions:
        fallback_parts.append("<h2 style='color:green;'>❓ QUESTIONS ASKED</h2>")
        for i, question in enumerate(questions[:3], 1):
            fallback_parts.append(f"=={i}. {question}==")

    # Announcements (for short periods)
    if is_short_period and announcements:
        fallback_parts.append("<h2 style='color:green;'>📢 ANNOUNCEMENTS & ALERTS</h2>")
        for i, announcement in enumerate(announcements[:3], 1):
            fallback_parts.append(f"=={i}. {announcement}==")

    # Technical discussions (for short periods)
    if is_short_period and technical_discussions:
        fallback_parts.append("<h2 style='color:green;'>🔧 TECHNICAL DISCUSSIONS</h2>")
        for i, discussion in enumerate(technical_discussions[:3], 1):
            fallback_parts.append(f"=={i}. {discussion}==")

    # Important resources
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>📁 IMPORTANT RESOURCES SHARED</h2>")
    else:
        fallback_parts.append("<h2 style='color:green;'>IMPORTANT RESOURCES</h2>")
    fallback_parts.append(f"==Files Shared: {len(file_shares)} | Links Shared: {len(links)}==")

    if is_short_period and (file_shares or links):
        if file_shares:
            fallback_parts.append("==Files mentioned:==")
            for file_share in file_shares[:3]:
                fallback_parts.append(f"==• {file_share}==")
        if links:
            fallback_parts.append("==Links shared:==")
            for link in links[:3]:
                fallback_parts.append(f"==• {link}==")

    # Actionable insights
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>✅ ACTIONABLE INSIGHTS</h2>")
    else:
        fallback_parts.append("<h2 style='color:green;'>ACTIONABLE INSIGHTS</h2>")
    fallback_parts.append(f"==Decisions Made: {len(decisions)} | Action Items: {len(action_items)} | Meetings Planned: {len(meetings)}==")

    if is_short_period and (decisions or action_items):
        if decisions:
            fallback_parts.append("==Specific decisions mentioned:==")
            for decision in decisions[:3]:
                fallback_parts.append(f"==• {decision}==")
        if action_items:
            fallback_parts.append("==Action items identified:==")
            for action in action_items[:3]:
                fallback_parts.append(f"==• {action}==")

    # Recommendations
    if is_short_period:
        fallback_parts.append("<h2 style='color:green;'>🎯 IMMEDIATE NEXT STEPS</h2>")
    else:
        fallback_parts.append("<h2 style='color:green;'>RECOMMENDATIONS</h2>")

    recommendations = []
    if is_short_period:
        if total_messages < 5:
            recommendations.append("Consider encouraging more group participation")
        if questions and messages and not any('answer' in str(messages).lower() for msg in messages):
            recommendations.append("Follow up on unanswered questions")
        if announcements:
            recommendations.append("Review and act on recent announcements")
        if not recommendations:
            recommendations.append("Continue monitoring group activity")
    else:
        if not meetings and user_count > 1:
            recommendations.append("Schedule a team meeting to discuss key topics")
        if len(file_shares) > 5:
            recommendations.append("Organize shared files in a central repository")
        if most_active_user and (most_active_user[1]/total_messages) > 0.4:
            recommendations.append("Encourage more balanced participation from all members")
        if not recommendations:
            recommendations.append("Continue current communication practices")

    fallback_parts.append(f"=={'; '.join(recommendations)}==")

    return '\n'.join(fallback_parts)


# Generate structured weekly summary
def generate_structured_summary(messages):
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # type: ignore
        prompt = (
            "You are an expert WhatsApp chat analyst. Read the conversation messages provided "
            "in JSON-like list form and produce a rich weekly summary as a JSON object with "
            "these properties: activity_summary (string), key_topics (array of strings), "
            "notable_events (array of strings), social_dynamics (string), and "
            "recommended_actions (array of strings)."
            "Make sure each property is filled based only on evidence in the messages. "
            "If there is insufficient information for a property, set it to an empty string "
            "or an empty array as appropriate. Use concise language, but include concrete "
            "details like usernames, counts, dates, and times when available.\n\n"
            f"Messages: {messages}"
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json"
            }
        )
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        logger.error(f"Error generating structured summary: {e}")
        return {"status": "error", "message": str(e)}


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

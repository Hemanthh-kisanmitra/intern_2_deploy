#!/usr/bin/env python
"""
Test script to verify question classification and processing
"""

import sys
import os
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Mock the imports that would normally come from the project
def parse_timestamp(timestamp_str):
    """Mock timestamp parsing function"""
    # Try different formats
    formats = [
        '%d/%m/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M %p',
        '%Y-%m-%d, %H:%M'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return None

def filter_messages_by_date(messages, start_date_str=None, end_date_str=None):
    """Mock date filtering function"""
    if not start_date_str and not end_date_str:
        return messages
    
    filtered = []
    for msg in messages:
        msg_date = parse_timestamp(msg['timestamp'])
        if not msg_date:
            continue
            
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            if msg_date < start_date:
                continue
                
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            end_date = end_date.replace(hour=23, minute=59, second=59)
            if msg_date > end_date:
                continue
                
        filtered.append(msg)
    
    return filtered

# Simplified version of the QuestionProcessor class for testing
class QuestionProcessor:
    """
    Simplified version of the QuestionProcessor for testing classification and processing
    """
    
    def __init__(self, messages, group_name):
        self.messages = messages if messages else []
        self.group_name = group_name
        self.users = list(set(msg.get('sender', 'Unknown') for msg in self.messages if msg.get('sender')))
        
    def classify_question(self, question: str) -> dict:
        """
        Classify the question type and extract relevant parameters.
        """
        question_lower = question.lower()
        print(f"Classifying question: {question}")
        
        # Date-based queries (check first to avoid misclassification)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 7/02/2024, 7-02-2024
            r'(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'\b\d{1,2}(st|nd|rd|th)\b',  # 7th, 15th
            r'(today|yesterday|tomorrow)',  # Relative dates
        ]
        
        is_date_query = any(re.search(pattern, question_lower) for pattern in date_patterns)
        print(f"Is date query: {is_date_query}")
        if is_date_query and any(keyword in question_lower for keyword in ['on', 'date', 'day', 'today', 'yesterday', 'tomorrow']):
            print("Classifying as date_based query")
            date_info = self._extract_date_info(question)
            print(f"Extracted date info: {date_info}")
            return {
                'type': 'date_based',
                'date_info': date_info,
                'original_question': question
            }
        
        # Topic/subject queries
        if any(keyword in question_lower for keyword in ['topic', 'subject', 'discuss', 'talk about', 'about', 'main topic', 'main themes']):
            return {
                'type': 'topics',
                'time_range': self._extract_time_range(question),
                'original_question': question
            }
        
        # Enhanced user-specific message queries
        # Check for more specific user-related questions
        user_related_patterns = [
            'messages from', 'message list', 'what did', 'what messages', 
            'show messages', 'list messages', 'messages by', 'messages of',
            'message on this day', 'messages on this day', 'this particular user', 'user message list'
        ]
        if any(pattern in question_lower for pattern in user_related_patterns):
            user_match = self._extract_user_from_question(question)
            if user_match:
                return {
                    'type': 'user_messages',
                    'user': user_match,
                    'time_range': self._extract_time_range(question),
                    'original_question': question
                }
        
        # Analytics queries
        if any(keyword in question_lower for keyword in ['active', 'most', 'least', 'top', 'bottom', 'count', 'number', 'how many', 'less', 'messages']):
            return {
                'type': 'analytics',
                'metric': self._extract_metric_type(question),
                'time_range': self._extract_time_range(question),
                'original_question': question
            }
        
        # Time-based queries
        if any(keyword in question_lower for keyword in ['time', 'hour', 'when', 'between', 'from', 'to', 'at', 'morning', 'afternoon', 'evening', 'night']):
            time_range = self._extract_time_range(question)
            if time_range:
                return {
                    'type': 'time_based',
                    'time_range': time_range,
                    'original_question': question
                }
        
        # Sentiment queries
        if any(keyword in question_lower for keyword in ['sentiment', 'mood', 'emotion', 'positive', 'negative', 'happy', 'sad', 'feel']):
            return {
                'type': 'sentiment',
                'time_range': self._extract_time_range(question),
                'original_question': question
            }
        
        # Default to general query
        return {
            'type': 'general',
            'time_range': self._extract_time_range(question),
            'original_question': question
        }
    
    def _extract_user_from_question(self, question: str):
        """Extract user name or phone number from question."""
        question_lower = question.lower()
        
        # Look for phone numbers (various formats)
        phone_patterns = [
            r'(\+?91\s?\d{5}\s?\d{5})',  # +91 12345 67890
            r'(\d{10})',  # 1234567890
            r'(\+91\d{10})',  # +911234567890
            r'(\d{5}\s?\d{5})',  # 12345 67890
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, question)
            if phone_match:
                phone = phone_match.group(1)
                # Find matching user in messages
                for user in self.users:
                    user_clean = user.replace(' ', '').replace('+', '').replace('-', '')
                    phone_clean = phone.replace(' ', '').replace('+', '').replace('-', '')
                    if phone_clean in user_clean or user_clean in phone_clean:
                        return user
        
        # Simple user name matching for testing
        for user in self.users:
            if user.lower() in question_lower:
                return user
                
        return None
    
    def _extract_time_range(self, question: str):
        """Extract time range from question."""
        return None
    
    def _extract_metric_type(self, question: str) -> str:
        """Extract the type of metric being asked about."""
        question_lower = question.lower()
        
        # More specific patterns for least active users
        if any(keyword in question_lower for keyword in ['least active', 'inactive', 'less active', 'lowest activity', 'less user', 'less messages']):
            return 'least_active_users'
        # More specific patterns for most active users
        elif any(keyword in question_lower for keyword in ['most active', 'active', 'highest activity', 'top contributors', 'active user', 'most messages']):
            return 'most_active_users'
        elif 'top' in question_lower:
            return 'top_users'
        elif any(keyword in question_lower for keyword in ['count', 'how many', 'total', 'number of', 'messages']):
            return 'message_count'
        else:
            return 'general_analytics'
    
    def _extract_date_info(self, question: str):
        """Extract date information from question."""
        question_lower = question.lower()
        print(f"Extracting date from question: {question}")
        
        # Handle relative dates
        today = datetime.now().date()
        if 'today' in question_lower:
            return {
                'date': today.strftime('%Y-%m-%d'),
                'type': 'specific_date'
            }
        elif 'yesterday' in question_lower:
            yesterday = today - timedelta(days=1)
            return {
                'date': yesterday.strftime('%Y-%m-%d'),
                'type': 'specific_date'
            }
        elif 'tomorrow' in question_lower:
            tomorrow = today + timedelta(days=1)
            return {
                'date': tomorrow.strftime('%Y-%m-%d'),
                'type': 'specific_date'
            }
        
        # Look for date patterns (support multiple formats)
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY or MM/DD/YYYY or DD-MM-YYYY
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})',   # DD/MM/YY or MM/DD/YY or DD-MM-YY
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(st|nd|rd|th)?,?\s*(\d{4})',  # January 7th, 2024
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(st|nd|rd|th)?\s*(\d{4})?',  # January 7 2024
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # Handle month name format
                    if groups[0].lower() in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']:
                        try:
                            month_name = groups[0].lower()
                            month_map = {
                                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                'september': 9, 'october': 10, 'november': 11, 'december': 12
                            }
                            month = month_map[month_name]
                            day = int(groups[1])
                            year_str = groups[3] if len(groups) > 3 and groups[3] else str(datetime.now().year)
                            year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str))
                            
                            result = {
                                'date': f"{year}-{month:02d}-{day:02d}",
                                'type': 'specific_date'
                            }
                            print(f"Extracted date: {result}")
                            return result
                        except (ValueError, IndexError):
                            continue
                    else:
                        # Handle numeric date format
                        part1, part2 = groups[0], groups[1]
                        year = groups[2] if len(groups) > 2 else str(datetime.now().year)
                        
                        print(f"Found date parts: {part1}, {part2}, {year}")
                        
                        # Handle 2-digit years
                        if len(year) == 2:
                            year = f"20{year}" if int(year) < 50 else f"19{year}"
                        
                        # For the format 7/02/2024, we need to determine if it's DD/MM/YYYY or MM/DD/YYYY
                        # Let's use the same heuristic as in utils.py
                        try:
                            a = int(part1)
                            b = int(part2)
                            if a > 12:
                                # First number > 12, so it must be DD/MM
                                day = part1
                                month = part2
                            elif b > 12:
                                # Second number > 12, so it must be MM/DD
                                month = part1
                                day = part2
                            else:
                                # Both <= 12, default to DD/MM (international format)
                                day = part1
                                month = part2
                        except ValueError:
                            # If conversion fails, default to DD/MM
                            day = part1
                            month = part2
                        
                        print(f"Interpreted as: day={day}, month={month}, year={year}")
                        
                        # Validate the date parts
                        try:
                            day_int = int(day)
                            month_int = int(month)
                            year_int = int(year)
                            
                            # Basic validation
                            if 1 <= day_int <= 31 and 1 <= month_int <= 12:
                                result = {
                                    'date': f"{year_int}-{month_int:02d}-{day_int:02d}",
                                    'type': 'specific_date'
                                }
                                print(f"Extracted date: {result}")
                                return result
                        except ValueError:
                            # If conversion fails, continue to next pattern
                            continue
        
        print("No date found in question")
        return None

    def _extract_main_topics(self, messages):
        """Extract main topics from messages."""
        if not messages:
            return {}
        
        # Collect all message content
        all_text = " ".join([msg.get('message', '') for msg in messages])
        
        # Simple keyword-based topic extraction
        common_topics = [
            'meeting', 'project', 'work', 'update', 'plan', 'schedule', 'event', 
            'discussion', 'question', 'issue', 'problem', 'solution', 'idea',
            'feedback', 'review', 'decision', 'agreement', 'disagreement'
        ]
        
        # Find topic words in messages
        found_topics = []
        for topic in common_topics:
            count = all_text.lower().count(topic)
            if count > 0:
                found_topics.append({
                    'topic': topic,
                    'frequency': count
                })
        
        # Sort by frequency
        found_topics.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Extract key messages as examples
        key_messages = []
        for msg in messages[:10]:  # First 10 messages as examples
            # Simple heuristic: longer messages or messages with punctuation might be more important
            if len(msg.get('message', '')) > 20 or any(p in msg.get('message', '') for p in '.!?'):
                key_messages.append({
                    'sender': msg.get('sender', 'Unknown'),
                    'timestamp': msg.get('timestamp', 'Unknown'),
                    'message': msg.get('message', '')
                })
        
        return {
            'main_topics': found_topics[:10],  # Top 10 topics
            'key_messages': key_messages,
            'total_topics_found': len(found_topics)
        }

    def _get_most_active_users(self, messages):
        """Get most active users with their message counts."""
        user_counts = Counter(msg['sender'] for msg in messages)
        total_messages = len(messages)
        
        # Sort by message count (descending)
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            "type": "most_active_users",
            "users": []
        }
        
        for user, count in sorted_users[:10]:  # Top 10 most active
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            result["users"].append({
                "user": user,
                "message_count": count,
                "percentage": round(percentage, 1)
            })
        
        return result
    
    def _get_least_active_users(self, messages):
        """Get least active users with their message counts."""
        user_counts = Counter(msg['sender'] for msg in messages)
        total_messages = len(messages)
        
        # Sort by message count (ascending)
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1])
        
        result = {
            "type": "least_active_users",
            "users": []
        }
        
        for user, count in sorted_users[:10]:  # Top 10 least active
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            result["users"].append({
                "user": user,
                "message_count": count,
                "percentage": round(percentage, 1)
            })
        
        return result

    def _handle_date_based_query(self, messages, classification):
        """Handle date-based queries."""
        date_info = classification.get('date_info')
        
        if not date_info:
            return {"error": "Could not extract date information from your question"}
        
        target_date = date_info.get('date')
        
        # Filter messages by the specific date
        date_messages = []
        for msg in messages:
            timestamp = parse_timestamp(msg['timestamp'])
            if timestamp:
                msg_date = timestamp.strftime('%Y-%m-%d')
                if msg_date == target_date:
                    date_messages.append(msg)
        
        if not date_messages:
            return {"error": f"No messages found on {target_date}"}
        
        # Format response
        response = {
            "type": "date_based",
            "date": target_date,
            "total_messages": len(date_messages),
            "messages": []
        }
        
        # Group messages by user for better organization
        user_messages = {}
        for msg in date_messages:
            sender = msg['sender']
            if sender not in user_messages:
                user_messages[sender] = []
            user_messages[sender].append(msg)
        
        # Add messages grouped by user
        for sender, msgs in user_messages.items():
            for msg in msgs:
                response["messages"].append({
                    "sender": sender,
                    "timestamp": msg['timestamp'],
                    "message": msg['message']
                })
        
        return response

    def _handle_user_messages_query(self, messages, classification):
        """Handle queries about specific user messages."""
        user = classification['user']
        
        if not user:
            return {"error": "Could not identify the user in your question."}
        
        # Filter messages by user
        user_messages = [msg for msg in messages if msg['sender'] == user]
        
        if not user_messages:
            return {"error": f"No messages found from user: {user}"}
        
        # Format response
        response = {
            "type": "user_messages",
            "user": user,
            "total_messages": len(user_messages),
            "messages": []
        }
        
        # Add actual messages (limit to recent ones if too many)
        max_messages = 20
        recent_messages = user_messages[-max_messages:] if len(user_messages) > max_messages else user_messages
        
        for msg in recent_messages:
            response["messages"].append({
                "timestamp": msg['timestamp'],
                "message": msg['message']
            })
        
        if len(user_messages) > max_messages:
            response["note"] = f"Showing {max_messages} most recent messages out of {len(user_messages)} total messages"
        
        return response

    def _handle_analytics_query(self, messages, classification):
        """Handle analytics queries."""
        metric_type = classification.get('metric', 'general_analytics')
        
        if metric_type == 'least_active_users':
            return self._get_least_active_users(messages)
        elif metric_type == 'most_active_users':
            return self._get_most_active_users(messages)
        else:
            return {"type": "analytics", "metric": metric_type, "message": "Analytics query processed"}

    def _handle_topics_query(self, messages, classification):
        """Handle topic analysis queries."""
        # Extract main topics from messages
        topics = self._extract_main_topics(messages)
        
        return {
            "type": "topics",
            "topics": topics,
            "total_messages_analyzed": len(messages)
        }

    def process_question(self, question: str, start_date = None, end_date = None):
        """
        Process the question and return appropriate response.
        """
        try:
            # Validate inputs
            if not question or not question.strip():
                return {"error": "Question cannot be empty"}
            
            if not self.messages:
                return {"error": "No messages available for analysis"}
            
            # For date-based queries, we should NOT filter by the general date range
            # because the user is asking about a specific date that might be outside the selected range
            classification = self.classify_question(question)
            print(f"Classification result: {classification}")
            
            # Route to appropriate handler
            if classification['type'] == 'date_based':
                return self._handle_date_based_query(self.messages, classification)
            elif classification['type'] == 'user_messages':
                return self._handle_user_messages_query(self.messages, classification)
            elif classification['type'] == 'analytics':
                return self._handle_analytics_query(self.messages, classification)
            elif classification['type'] == 'topics':
                return self._handle_topics_query(self.messages, classification)
            else:
                return {"type": "general", "message": "General query processed"}
                
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}

# Mock messages data for testing
mock_messages = [
    {
        'timestamp': '12/02/24, 10:30 AM',
        'sender': 'John Doe',
        'message': 'Hello everyone! Let\'s discuss the project meeting tomorrow.'
    },
    {
        'timestamp': '12/02/24, 10:32 AM',
        'sender': 'Jane Smith',
        'message': 'Hi John! I have some updates on the project work.'
    },
    {
        'timestamp': '07/02/24, 2:15 PM',
        'sender': 'John Doe',
        'message': 'Meeting today at 3 PM to discuss project schedule and work plan.'
    },
    {
        'timestamp': '07/02/24, 2:15 PM',
        'sender': 'Jane Smith',
        'message': 'I will be late for the meeting, but I have feedback on the project.'
    },
    {
        'timestamp': '07/02/24, 2:16 PM',
        'sender': 'Bob Johnson',
        'message': 'See you at the meeting. I have some ideas for the project work.'
    }
]

def test_question_processing():
    """Test the actual question processing functionality"""
    
    # Initialize the question processor with mock messages
    processor = QuestionProcessor(mock_messages, "Test Group")
    
    print("\n\nTesting Question Processing:")
    print("=" * 50)
    
    # Test most active users query
    print("\n1. Testing 'most active users' query:")
    result = processor.process_question("list the most active users")
    print(f"Result: {result}")
    
    # Test least active users query
    print("\n2. Testing 'least active users' query:")
    result = processor.process_question("list the least active users")
    print(f"Result: {result}")
    
    # Test date-based query
    print("\n3. Testing date-based query:")
    result = processor.process_question("messages on 07/02/2024")
    print(f"Result: {result}")
    
    # Test user-specific query
    print("\n4. Testing user-specific query:")
    result = processor.process_question("show messages from John Doe")
    print(f"Result: {result}")
    
    # Test topic query
    print("\n5. Testing topic query:")
    result = processor.process_question("what are the main topics discussed?")
    print(f"Result: {result}")
    
    # Test relative date query
    print("\n6. Testing relative date query:")
    result = processor.process_question("what did we talk about today?")
    print(f"Result: {result}")

def test_question_classification():
    """Test the question classification for various query types"""
    
    # Initialize the question processor
    processor = QuestionProcessor(mock_messages, "Test Group")
    
    # Test questions
    test_questions = [
        "list the most active users",
        "list the least active users", 
        "messages on 07/02/2024",
        "this particular user John Doe message list",
        "show messages from Jane Smith",
        "who sent the most messages?",
        "who are the least active users?",
        "what happened on February 7th 2024?",
        "list messages from Bob Johnson",
        "what are the main topics discussed?",
        "what did we talk about today?",
        "show messages from yesterday"
    ]
    
    print("Testing Question Classification:")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        classification = processor.classify_question(question)
        print(f"Classification: {classification}")
        
        # Test processing (without actual date filtering for mock data)
        try:
            result = processor.process_question(question)
            print(f"Result type: {result.get('type', 'unknown')}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("Processing: SUCCESS")
        except Exception as e:
            print(f"Processing error: {e}")

def test_topic_extraction():
    """Test the topic extraction functionality"""
    processor = QuestionProcessor(mock_messages, "Test Group")
    
    print("\n\nTesting Topic Extraction:")
    print("=" * 50)
    
    topics = processor._extract_main_topics(mock_messages)
    print(f"Extracted topics: {topics}")

if __name__ == "__main__":
    test_question_classification()
    test_question_processing()
    test_topic_extraction()
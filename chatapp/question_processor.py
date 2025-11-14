import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from .utils import parse_timestamp, filter_messages_by_date
from .business_metrics import calculate_business_metrics

# Import sentiment analyzer only when needed to avoid Django settings issues
try:
    from .sentiment_analyzer import analyze_sentiment
except Exception:
    analyze_sentiment = None


class QuestionProcessor:
    """
    Intelligent question processor for WhatsApp chat analysis.
    Classifies questions and provides specialized handlers for different query types.
    """
    
    def __init__(self, messages: List[Dict], group_name: str):
        self.messages = messages if messages else []
        self.group_name = group_name
        self.users = list(set(msg.get('sender', 'Unknown') for msg in self.messages if msg.get('sender')))
        
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

    def classify_question(self, question: str) -> Dict[str, Any]:
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
        
        # Enhanced user-specific message queries
        # Check for more specific user-related questions
        user_related_patterns = [
            'messages from', 'message list', 'what did', 'what messages', 
            'show messages', 'list messages', 'messages by', 'messages of',
            'message on this day', 'messages on this day', 'this particular user', 'user message list',
            'what did.*say', 'what.*said'
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
        
        # Topic/subject queries
        if any(keyword in question_lower for keyword in ['topic', 'subject', 'discuss', 'talk about', 'about', 'main topic', 'main themes']):
            return {
                'type': 'topics',
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

    def _extract_date_info(self, question: str) -> Optional[Dict[str, Any]]:
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
                            year_str = groups[3] if len(groups) > 3 and groups[3] else str(today.year)
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
                        year = groups[2] if len(groups) > 2 else str(today.year)
                        
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

    def process_question(self, question: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
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
            
            # Check if this is a date-based query
            if classification['type'] == 'date_based':
                print("Processing as date_based query")
                # For date-based queries, use ALL messages and let the query handler filter by the specific date
                filtered_messages = self.messages
            else:
                print("Processing as regular query with date filtering")
                # For all other queries, apply the general date range filtering
                filtered_messages = filter_messages_by_date(self.messages, start_date, end_date)
            
            if not filtered_messages and classification['type'] != 'date_based':
                return {"error": "No messages found in the selected date range"}
            
            # Route to appropriate handler
            if classification['type'] == 'date_based':
                return self._handle_date_based_query(filtered_messages, classification)
            elif classification['type'] == 'user_messages':
                return self._handle_user_messages_query(filtered_messages, classification)
            elif classification['type'] == 'analytics':
                return self._handle_analytics_query(filtered_messages, classification)
            elif classification['type'] == 'time_based':
                return self._handle_time_based_query(filtered_messages, classification)
            elif classification['type'] == 'sentiment':
                return self._handle_sentiment_query(filtered_messages, classification)
            elif classification['type'] == 'topics':
                return self._handle_topics_query(filtered_messages, classification)
            else:
                return self._handle_general_query(filtered_messages, classification)
                
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}

    def _handle_date_based_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle date-based queries."""
        date_info = classification.get('date_info')
        
        if not date_info:
            return {"error": "Could not extract date information from your question"}
        
        target_date = date_info.get('date')
        
        # Debug information
        print(f"Searching for messages on date: {target_date}")
        
        # Filter messages by the specific date
        date_messages = []
        all_dates = set()  # To collect all available dates for debugging
        
        for msg in messages:
            timestamp = parse_timestamp(msg['timestamp'])
            if timestamp:
                msg_date = timestamp.strftime('%Y-%m-%d')
                all_dates.add(msg_date)  # Collect all dates for debugging
                if msg_date == target_date:
                    date_messages.append(msg)
        
        print(f"Found {len(date_messages)} messages for date {target_date}")
        
        if not date_messages:
            # Show some sample dates for debugging
            sample_dates = sorted(list(all_dates))[:10]  # First 10 dates
            date_range_info = f"Available dates range from {min(all_dates) if all_dates else 'None'} to {max(all_dates) if all_dates else 'None'}"
            print(f"Sample available dates: {sample_dates}")
            print(f"Date range info: {date_range_info}")
            
            return {"error": f"No messages found on {target_date}. {date_range_info}"}
        
        # Format response with better organization
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
    
    def _handle_user_messages_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle queries about specific user messages."""
        user = classification['user']
        time_range = classification.get('time_range')
        
        if not user:
            return {"error": "Could not identify the user in your question. Please specify the user name or phone number."}
        
        # Filter messages by user
        user_messages = [msg for msg in messages if msg['sender'] == user]
        
        if not user_messages:
            return {"error": f"No messages found from user: {user}"}
        
        # Apply time filtering if specified
        if time_range:
            user_messages = self._filter_messages_by_time(user_messages, time_range)
        
        if not user_messages:
            return {"error": f"No messages found from {user} in the specified time range"}
        
        # Format response
        response = {
            "type": "user_messages",
            "user": user,
            "total_messages": len(user_messages),
            "messages": []
        }
        
        # Add actual messages (limit to recent ones if too many)
        max_messages = 100  # Increased limit for more detailed information
        recent_messages = user_messages[-max_messages:] if len(user_messages) > max_messages else user_messages
        
        for msg in recent_messages:
            response["messages"].append({
                "timestamp": msg['timestamp'],
                "message": msg['message']
            })
        
        if len(user_messages) > max_messages:
            response["note"] = f"Showing {max_messages} most recent messages out of {len(user_messages)} total messages"
        
        return response
    
    def _handle_analytics_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle analytics queries."""
        metric_type = classification.get('metric', 'general_analytics')
        time_range = classification.get('time_range')
        
        # Apply time filtering if specified
        if time_range:
            messages = self._filter_messages_by_time(messages, time_range)
        
        if metric_type == 'least_active_users':
            return self._get_least_active_users(messages)
        elif metric_type == 'most_active_users':
            return self._get_most_active_users(messages)
        elif metric_type == 'top_users':
            return self._get_top_users(messages)
        elif metric_type == 'message_count':
            return self._get_message_count(messages)
        else:
            return self._get_general_analytics(messages)
    
    def _handle_time_based_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle time-based queries."""
        time_range = classification.get('time_range')
        
        if not time_range:
            return {"error": "Could not extract time information from your question"}
        
        # Filter messages by time
        filtered_messages = self._filter_messages_by_time(messages, time_range)
        
        if not filtered_messages:
            return {"error": "No messages found in the specified time range"}
        
        return {
            "type": "time_based",
            "time_range": time_range,
            "total_messages": len(filtered_messages),
            "messages": filtered_messages[-20:] if len(filtered_messages) > 20 else filtered_messages  # Show last 20 messages
        }
    
    def _handle_sentiment_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle sentiment analysis queries."""
        time_range = classification.get('time_range')
        
        # Apply time filtering if specified
        if time_range:
            messages = self._filter_messages_by_time(messages, time_range)
        
        # Perform sentiment analysis if available
        if analyze_sentiment:
            sentiment_data = analyze_sentiment(messages)
        else:
            sentiment_data = {"error": "Sentiment analysis not available"}
        
        return {
            "type": "sentiment",
            "data": sentiment_data
        }
    
    def _handle_topics_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle topic analysis queries."""
        time_range = classification.get('time_range')
        
        # Apply time filtering if specified
        if time_range:
            messages = self._filter_messages_by_time(messages, time_range)
        
        # Extract main topics from messages
        topics = self._extract_main_topics(messages)
        
        return {
            "type": "topics",
            "topics": topics,
            "total_messages_analyzed": len(messages)
        }
    
    def _handle_general_query(self, messages: List[Dict], classification: Dict) -> Dict[str, Any]:
        """Handle general queries using AI."""
        time_range = classification.get('time_range')
        
        # Apply time filtering if specified
        if time_range:
            messages = self._filter_messages_by_time(messages, time_range)
        
        # Create context for AI
        context = self._create_ai_context(messages)
        
        return {
            "type": "general",
            "context": context,
            "question": classification['original_question']
        }
    
    def _filter_messages_by_time(self, messages: List[Dict], time_range: Dict) -> List[Dict]:
        """Filter messages by specific time range."""
        filtered = []
        
        for msg in messages:
            timestamp = parse_timestamp(msg['timestamp'])
            if not timestamp:
                continue
            
            msg_time = timestamp.time()
            
            if time_range['type'] == 'time_range':
                start_time = datetime.strptime(time_range['start_time'], '%H:%M').time()
                end_time = datetime.strptime(time_range['end_time'], '%H:%M').time()
                
                if start_time <= msg_time <= end_time:
                    filtered.append(msg)
            elif time_range['type'] == 'specific_time':
                target_time = datetime.strptime(time_range['time'], '%H:%M').time()
                # Allow 30-minute window around the target time
                time_diff = abs((datetime.combine(datetime.today(), msg_time) - 
                               datetime.combine(datetime.today(), target_time)).total_seconds())
                if time_diff <= 1800:  # 30 minutes
                    filtered.append(msg)
        
        return filtered
    
    def _get_least_active_users(self, messages: List[Dict]) -> Dict[str, Any]:
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
    
    def _get_most_active_users(self, messages: List[Dict]) -> Dict[str, Any]:
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
    
    def _get_top_users(self, messages: List[Dict]) -> Dict[str, Any]:
        """Get top users (same as most active)."""
        return self._get_most_active_users(messages)
    
    def _get_message_count(self, messages: List[Dict]) -> Dict[str, Any]:
        """Get message count statistics."""
        user_counts = Counter(msg['sender'] for msg in messages)
        
        return {
            "type": "message_count",
            "total_messages": len(messages),
            "total_users": len(user_counts),
            "average_messages_per_user": round(len(messages) / len(user_counts), 1) if user_counts else 0,
            "user_breakdown": dict(user_counts)
        }
    
    def _get_general_analytics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Get general analytics."""
        metrics = calculate_business_metrics(messages)
        
        return {
            "type": "general_analytics",
            "metrics": metrics
        }
    
    def _create_ai_context(self, messages: List[Dict]) -> str:
        """Create context for AI processing."""
        # Limit context size
        max_messages = 100
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context = f"Chat Group: {self.group_name}\n"
        context += f"Total Messages: {len(messages)}\n"
        context += f"Users: {', '.join(self.users)}\n\n"
        context += "Recent Messages:\n"
        
        for msg in recent_messages:
            context += f"{msg['sender']}: {msg['message']}\n"
        
        return context
    
    def _extract_user_from_question(self, question: str) -> Optional[str]:
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
        
        # Enhanced user name matching with better partial matching
        best_match = None
        best_score = 0
        
        for user in self.users:
            # Extract name part (before phone number if present)
            name_part = user.split('+')[0].strip()
            if not name_part or len(name_part) < 2:
                continue
            
            name_lower = name_part.lower()
            score = 0
            
            # Split the question into words for better matching
            question_words = re.findall(r'\b\w+\b', question_lower)
            name_words = re.findall(r'\b\w+\b', name_lower)
            
            # Check for exact word matches (highest priority)
            for q_word in question_words:
                if len(q_word) >= 3:  # Only consider words with 3+ characters
                    for n_word in name_words:
                        if q_word == n_word:
                            score += 10  # Exact word match
                        elif q_word in n_word or n_word in q_word:
                            score += 5   # Partial word match
            
            # Check for partial name matches
            if name_lower in question_lower:
                score += 8  # Full name in question
            elif any(word in name_lower for word in question_words if len(word) >= 3):
                score += 3  # Some words match
            
            # Check for reverse matching (question words in name)
            if any(n_word in question_lower for n_word in name_words if len(n_word) >= 3):
                score += 2
            
            # Bonus for longer matches
            if len(name_part) > 5:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = user
        
        # Only return if we have a reasonable match score
        # Increased threshold to avoid false matches
        if best_score >= 5:
            return best_match
        
        return None

    def _extract_time_range(self, question: str) -> Optional[Dict[str, Any]]:
        """Extract time range from question."""
        question_lower = question.lower()
        
        # Look for specific time patterns
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(am|pm)?\s*to\s*(\d{1,2}):(\d{2})\s*(am|pm)?',
            r'from\s*(\d{1,2}):(\d{2})\s*(am|pm)?\s*to\s*(\d{1,2}):(\d{2})\s*(am|pm)?',
            r'between\s*(\d{1,2}):(\d{2})\s*(am|pm)?\s*and\s*(\d{1,2}):(\d{2})\s*(am|pm)?',
            r'at\s*(\d{1,2}):(\d{2})\s*(am|pm)?',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, question_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    start_hour = int(groups[0])
                    start_min = int(groups[1])
                    start_ampm = groups[2] if len(groups) > 2 else None
                    
                    if start_ampm == 'pm' and start_hour != 12:
                        start_hour += 12
                    elif start_ampm == 'am' and start_hour == 12:
                        start_hour = 0
                    
                    if len(groups) >= 5:  # Has end time
                        end_hour = int(groups[3])
                        end_min = int(groups[4])
                        end_ampm = groups[5] if len(groups) > 5 else None
                        
                        if end_ampm == 'pm' and end_hour != 12:
                            end_hour += 12
                        elif end_ampm == 'am' and end_hour == 12:
                            end_hour = 0
                        
                        return {
                            'start_time': f"{start_hour:02d}:{start_min:02d}",
                            'end_time': f"{end_hour:02d}:{end_min:02d}",
                            'type': 'time_range'
                        }
                    else:  # Single time
                        return {
                            'time': f"{start_hour:02d}:{start_min:02d}",
                            'type': 'specific_time'
                        }
        
        return None

    def _extract_main_topics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract main topics from messages."""
        if not messages:
            return {}
        
        # Collect all message content
        all_text = " ".join([msg.get('message', '') for msg in messages])
        
        # Simple keyword-based topic extraction
        # In a real implementation, this would use NLP techniques
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

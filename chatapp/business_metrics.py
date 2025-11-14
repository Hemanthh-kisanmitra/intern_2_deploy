from datetime import datetime
from collections import Counter, defaultdict
import re
from .utils import parse_timestamp

def calculate_business_metrics(messages):
    if not messages:
        return {"error": "No messages found"}
    
    # Initialize all hours and days to ensure comprehensive data
    metrics = {
        'total_messages': len(messages),
        'total_users': len(set(msg['sender'] for msg in messages if msg.get('sender'))),
        'messages_per_user': {},
        'activity_by_hour': {str(h): 0 for h in range(24)},  # Initialize all hours
        'activity_by_day': {},
        'activity_by_hour_with_users': {},
        'top_keywords': {},
        'business_keywords_count': {},
        'peak_hour': None,
        'peak_day': None,
        'most_active_user': None
    }
    
    # Initialize all days
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    for day in day_names:
        metrics['activity_by_day'][day] = 0
    
    user_counts = Counter(msg['sender'] for msg in messages if msg.get('sender'))
    metrics['messages_per_user'] = dict(user_counts)
    
    # Find most active user
    if user_counts:
        metrics['most_active_user'] = user_counts.most_common(1)[0][0]
    
    # Initialize activity_by_hour_with_users for all hours
    for hour in range(24):
        metrics['activity_by_hour_with_users'][hour] = {}
    
    # Process messages for hourly activity
    hourly_counts = Counter()
    daily_counts = Counter()
    
    for msg in messages:
        if not msg.get('sender'):  # Skip system messages
            continue
            
        timestamp = parse_timestamp(msg['timestamp'])
        if timestamp:
            hour = timestamp.hour
            hour_str = str(hour)
            
            # Count hourly activity
            metrics['activity_by_hour'][hour_str] += 1
            hourly_counts[hour] += 1
            
            # Track user activity by hour
            sender = msg['sender']
            if sender not in metrics['activity_by_hour_with_users'][hour]:
                metrics['activity_by_hour_with_users'][hour][sender] = 0
            metrics['activity_by_hour_with_users'][hour][sender] += 1
            
            # Count daily activity
            day = timestamp.strftime('%A')
            metrics['activity_by_day'][day] += 1
            daily_counts[day] += 1
    
    # Find peak hour and day
    if hourly_counts:
        metrics['peak_hour'] = hourly_counts.most_common(1)[0][0]
    if daily_counts:
        metrics['peak_day'] = daily_counts.most_common(1)[0][0]
    
    all_text = ' '.join([msg['message'].lower() for msg in messages])
    words = re.findall(r'\b\w+\b', all_text)
    word_counts = Counter(words)
    
    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 'the']
    
    filtered_words = {word: count for word, count in word_counts.items() if word not in common_words and len(word) > 2}
    metrics['top_keywords'] = dict(Counter(filtered_words).most_common(20))
    
    business_keywords = ['price', 'cost', 'order', 'delivery', 'payment', 'product', 'service', 'meeting', 'client', 'customer', 'project', 'deadline', 'invoice', 'contract', 'deal', 'offer', 'discount', 'profit', 'loss', 'revenue', 'sales', 'marketing', 'promotion']
    
    for keyword in business_keywords:
        count = all_text.count(keyword)
        if count > 0:
            metrics['business_keywords_count'][keyword] = count
    
    return metrics
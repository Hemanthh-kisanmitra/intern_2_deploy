"""
Sample data generator for testing activity analysis with rich data patterns
"""
import random
from datetime import datetime, timedelta
import json

def generate_comprehensive_sample_data():
    """Generate sample chat data with realistic activity patterns like the second image"""
    
    # Sample users with realistic names
    users = [
        "Far Shivaji Shankar Desi",
        "Sri Kalpanjay Nathe", 
        "Far Balkrushna Sukhdev",
        "+91 90112 38856",
        "Far Tanaji Madhav Gawa",
        "Rajesh Kumar",
        "Priya Sharma",
        "Amit Patel"
    ]
    
    messages = []
    
    # Generate messages over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    current_date = start_date
    message_id = 1
    
    while current_date <= end_date:
        # Determine if this day should have activity (80% chance)
        if random.random() < 0.8:
            # Generate 5-50 messages for this day
            daily_messages = random.randint(5, 50)
            
            for _ in range(daily_messages):
                # Choose random hour with realistic patterns
                # More activity during 8-22 hours
                if random.random() < 0.7:
                    hour = random.randint(8, 22)
                else:
                    hour = random.randint(0, 23)
                
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                
                timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                
                # Choose random user
                sender = random.choice(users)
                
                # Generate realistic message content
                message_templates = [
                    "Hello everyone!",
                    "Good morning",
                    "How are you?",
                    "Thanks for sharing",
                    "That's interesting",
                    "I agree with you",
                    "Let me check and get back",
                    "Sure, no problem",
                    "Great work!",
                    "See you tomorrow",
                    "Have a nice day",
                    "What time is the meeting?",
                    "Can you share the details?",
                    "I'll be there in 10 minutes",
                    "Perfect timing",
                    "Let's discuss this later",
                    "Good point",
                    "I think we should proceed",
                    "Any updates on this?",
                    "Looking forward to it"
                ]
                
                message_content = random.choice(message_templates)
                
                # Format timestamp like WhatsApp export
                formatted_timestamp = timestamp.strftime("%d/%m/%y, %I:%M %p")
                
                message = {
                    'id': message_id,
                    'timestamp': formatted_timestamp,
                    'sender': sender,
                    'message': message_content
                }
                
                messages.append(message)
                message_id += 1
        
        current_date += timedelta(days=1)
    
    # Create the chat data structure
    chat_data = {
        'SAMPLE_COMPREHENSIVE_CHAT': {
            'messages': messages,
            'metadata': {
                'total_messages': len(messages),
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'users': users
            }
        }
    }
    
    return chat_data

def save_sample_data_to_file():
    """Save the generated sample data to a JSON file"""
    sample_data = generate_comprehensive_sample_data()
    
    with open('sample_comprehensive_chat.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Generated {len(sample_data['SAMPLE_COMPREHENSIVE_CHAT']['messages'])} sample messages")
    print("Sample data saved to sample_comprehensive_chat.json")
    
    return sample_data

if __name__ == "__main__":
    save_sample_data_to_file()

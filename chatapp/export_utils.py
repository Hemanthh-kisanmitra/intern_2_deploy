import json
import csv
from datetime import datetime

def export_to_json(data, filename):
    """Export data to JSON format"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return filename

def export_to_csv(data, filename):
    """Export data to CSV format"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        
        writer.writerow(['Timestamp', 'Sender', 'Message'])
        
        
        for msg in data.get('messages', []):
            writer.writerow([msg.get('timestamp', ''), msg.get('sender', ''), msg.get('message', '')])
    
    return filename

def generate_export_filename(group_name, format_type):
    """Generate a filename for export based on group name and format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{group_name}_chat_analysis_{timestamp}.{format_type}"
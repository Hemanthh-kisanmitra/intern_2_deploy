from datetime import datetime


def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to datetime object.

    Supports multiple WhatsApp export formats, including:
    - 12-hour and 24-hour clocks
    - With and without seconds
    - MM/DD/YY and DD/MM/YY (and YYYY variants)
    - Normalizes narrow and non-breaking spaces
    """
    if not timestamp_str:
        return None

    # Normalize special spaces used in WhatsApp exports
    timestamp_str = (
        timestamp_str
        .replace('\u202F', ' ')  # narrow no-break space
        .replace('\u00A0', ' ')  # non-breaking space
        .strip()
    )

    # Determine likely date order for slash-separated formats (MM/DD vs DD/MM)
    # Heuristic: if first number > 12 -> DD/MM; if second number > 12 -> MM/DD; if both <= 12 -> prefer MM/DD (common in exports)
    order_preference = 'MDY'
    try:
        date_part = timestamp_str.split(',')[0].strip()
        if '/' in date_part:
            parts = date_part.split('/')
            if len(parts) >= 2:
                a = int(parts[0])
                b = int(parts[1])
                if a > 12:
                    order_preference = 'DMY'
                elif b > 12:
                    order_preference = 'MDY'
                else:
                    order_preference = 'DMY'  # default to international if ambiguous
    except Exception:
        pass

    # Build formats with preferred order first
    mdy_formats = [
        '%m/%d/%y, %I:%M %p', '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %I:%M:%S %p', '%m/%d/%Y, %I:%M:%S %p',
        '%m/%d/%y, %H:%M', '%m/%d/%Y, %H:%M',
        '%m/%d/%y, %H:%M:%S', '%m/%d/%Y, %H:%M:%S',
    ]
    dmy_formats = [
        '%d/%m/%y, %I:%M %p', '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %I:%M:%S %p', '%d/%m/%Y, %I:%M:%S %p',
        '%d/%m/%y, %H:%M', '%d/%m/%Y, %H:%M',
        '%d/%m/%y, %H:%M:%S', '%d/%m/%Y, %H:%M:%S',
    ]
    iso_formats = ['%Y-%m-%d, %H:%M', '%Y-%m-%d, %H:%M:%S']

    formats = (dmy_formats + mdy_formats if order_preference == 'DMY' else mdy_formats + dmy_formats) + iso_formats

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return None


def filter_messages_by_date(messages, start_date_str, end_date_str):
    """Filter messages by date range"""
    if not start_date_str and not end_date_str:
        return messages
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
    return filtered_messages
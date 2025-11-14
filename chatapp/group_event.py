import re
from datetime import datetime, timedelta


def parse_timestamp(timestamp_str):
    timestamp_str = timestamp_str.replace('\u202F', ' ').replace('\u00A0', ' ')
    formats = [
        '%m/%d/%y, %I:%M %p',
        '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %H:%M',
        '%m/%d/%Y, %H:%M',
        '%d/%m/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %H:%M',
        '%d/%m/%Y, %H:%M',
        '%Y-%m-%d, %H:%M',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            pass
    return None


def analyze_group_events(messages):
    events = {
        'added': [],
        'left': [],
        'removed': [],
        'changed_subject': [],
        'changed_icon': [],
        'created': []
    }

    for msg in messages:
        text = msg['message']
        sender = msg['sender']
        timestamp = msg['timestamp']

        # Added events - improved pattern matching
        if re.search(r'added|joined', text, re.IGNORECASE):
            # More comprehensive pattern for added events
            match = re.search(r'(.+?)\s+(?:added|joined)\s+(.+)', text, re.IGNORECASE)
            if match:
                adder = match.group(1).strip()
                added_person = match.group(2).strip()
                # Clean up special characters
                adder = adder.replace('~', '').strip()
                added_person = added_person.replace('~', '').strip()
                events['added'].append({
                    'timestamp': timestamp,
                    'adder': adder,
                    'added_person': added_person,
                    'raw_message': text
                })

        # Left events
        elif re.search(r'left|exited', text, re.IGNORECASE):
            match = re.search(r'(.+?)\s+(?:left|exited)', text, re.IGNORECASE)
            if match:
                person_left = match.group(1).strip().replace('~', '').strip()
                events['left'].append({
                    'timestamp': timestamp,
                    'person': person_left,
                    'raw_message': text
                })

        # Removed events
        elif re.search(r'removed|kicked', text, re.IGNORECASE):
            match = re.search(r'(.+?)\s+(?:removed|kicked)\s+(.+)', text, re.IGNORECASE)
            if match:
                remover = match.group(1).strip().replace('~', '').strip()
                removed_person = match.group(2).strip().replace('~', '').strip()
                events['removed'].append({
                    'timestamp': timestamp,
                    'remover': remover,
                    'removed_person': removed_person,
                    'raw_message': text
                })

        # Changed subject events
        elif 'changed the subject to' in text:
            match = re.search(r'(.+?)\s+changed the subject to\s*[\'"](.+?)[\'"]', text, re.IGNORECASE)
            if match:
                changer = match.group(1).strip().replace('~', '').strip()
                new_subject = match.group(2).strip()
                events['changed_subject'].append({
                    'timestamp': timestamp,
                    'changer': changer,
                    'new_subject': new_subject,
                    'raw_message': text
                })

        # Changed icon events
        elif 'changed' in text and 'icon' in text:
            match = re.search(r'(.+?)\s+changed.*icon', text, re.IGNORECASE)
            if match:
                changer = match.group(1).strip().replace('~', '').strip()
                events['changed_icon'].append({
                    'timestamp': timestamp,
                    'changer': changer,
                    'raw_message': text
                })

        # Created group events
        elif 'created' in text and 'group' in text:
            match = re.search(r'(.+?)\s+created.*group', text, re.IGNORECASE)
            if match:
                creator = match.group(1).strip().replace('~', '').strip()
                events['created'].append({
                    'timestamp': timestamp,
                    'creator': creator,
                    'raw_message': text
                })

    return events


def get_event_counts(events):
    return {
        'added': len(events['added']),
        'left': len(events['left']),
        'removed': len(events['removed']),
        'changed_subject': len(events['changed_subject']),
        'changed_icon': len(events['changed_icon']),
        'created': len(events['created'])
    }


def get_event_details(events, event_type, start_date=None, end_date=None):
    event_list = events.get(event_type, [])
    if start_date or end_date:
        filtered_events = []
        for event in event_list:
            event_date = parse_timestamp(event['timestamp'])
            if event_date is None:
                continue
            if start_date and event_date < start_date:
                continue
            if end_date and event_date > end_date:
                continue
            filtered_events.append(event)
        return filtered_events
    return event_list


def get_top_removers(events, limit=5):
    remover_counts = {}
    for event in events['removed']:
        remover = event['remover']
        remover_counts[remover] = remover_counts.get(remover, 0) + 1

    sorted_removers = sorted(remover_counts.items(), key=lambda x: x[1], reverse=True)
    # Convert tuples to dictionaries for consistency
    return [{'name': name, 'count': count} for name, count in sorted_removers[:limit]]


# ------------------- New helpers for analytics dashboard -------------------

def _normalize_events(events):
    """Flatten to a standard schema for easier filtering/aggregation."""
    normalized = []
    def add_item(event_type, ts, actor, target, details):
        normalized.append({
            'event_type': event_type,
            'timestamp': ts,
            'actor': actor,
            'target': target,
            'details': details,
        })

    for e in events['added']:
        add_item('added', e['timestamp'], e['adder'], e.get('added_person'), e.get('raw_message'))
    for e in events['left']:
        add_item('left', e['timestamp'], e['person'], None, e.get('raw_message'))
    for e in events['removed']:
        add_item('removed', e['timestamp'], e['remover'], e.get('removed_person'), e.get('raw_message'))
    for e in events['changed_subject']:
        add_item('changed_subject', e['timestamp'], e['changer'], None, f"New subject: {e.get('new_subject')}")
    for e in events['changed_icon']:
        add_item('changed_icon', e['timestamp'], e['changer'], None, 'Icon changed')
    for e in events['created']:
        add_item('created', e['timestamp'], e['creator'], None, 'Group created')
    return normalized


def _filter_normalized(normalized, start_date=None, end_date=None, event_types=None, user=None):
    out = []
    types_set = set([t for t in (event_types or [])]) if event_types else None
    user_lower = user.lower() if user else None
    for row in normalized:
        dt = parse_timestamp(row['timestamp'])
        if dt is None:
            continue
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue
        if types_set and row['event_type'] not in types_set:
            continue
        if user_lower and not (
            (row['actor'] and user_lower in row['actor'].lower()) or
            (row['target'] and user_lower in row['target'].lower())
        ):
            continue
        out.append({**row, 'dt': dt})
    return out


def compute_timeseries(normalized_rows):
    by_day = {}
    for r in normalized_rows:
        day = r['dt'].date().isoformat()
        d = by_day.setdefault(day, {'total': 0, 'added': 0, 'left': 0, 'removed': 0, 'changed_subject': 0, 'changed_icon': 0, 'created': 0})
        d['total'] += 1
        d[r['event_type']] += 1
    series = [
        {'date': day, **counts}
        for day, counts in by_day.items()
    ]
    series.sort(key=lambda x: x['date'])
    return series


def compute_distribution(normalized_rows):
    counts = {'added': 0, 'left': 0, 'removed': 0, 'changed_subject': 0, 'changed_icon': 0, 'created': 0}
    for r in normalized_rows:
        counts[r['event_type']] += 1
    total = sum(counts.values()) or 1
    percentages = {k: (v * 100.0) / total for k, v in counts.items()}
    return {'counts': counts, 'percentages': percentages, 'total': total}


def compute_most_active_day(timeseries):
    if not timeseries:
        return None
    return max(timeseries, key=lambda x: x['total'])


def compute_top_contributors(normalized_rows, limit=5):
    # Actor could be the person who left; treat them as contributor as well
    counts = {}
    for r in normalized_rows:
        actor = r['actor'] or 'Unknown'
        counts[actor] = counts.get(actor, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{'name': k, 'count': v} for k, v in ranked[:limit]]


def extract_unique_actors(normalized_rows):
    actors = set()
    for r in normalized_rows:
        if r['actor']:
            actors.add(r['actor'])
        if r['target']:
            actors.add(r['target'])
    return sorted(actors)
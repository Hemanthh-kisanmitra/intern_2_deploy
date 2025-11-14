import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Constants for topic analysis
TOPIC_MIN_WORD_LENGTH = 3
TOPIC_MAX_TOPICS = 5

def extract_topics(messages: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Extract topics from messages using TF-IDF and LDA
    
    Args:
        messages: List of message dictionaries with 'message' and 'sender' keys
        top_n: Number of top topics to return
        
    Returns:
        List of topic dictionaries with topic words, method, score, and examples
    """
    if not messages:
        return []
    
    # Validate messages and filter out system messages
    valid_messages = []
    for msg in messages:
        text = msg.get('message', '').strip()
        # Skip system messages and media
        if text and not any(skip in text.lower() for skip in [
            'media omitted', 'security code', 'tap to learn', 
            'this message was deleted', 'messages and calls are end-to-end encrypted'
        ]):
            valid_messages.append(msg)
    
    # If no valid messages after validation, return empty list
    if not valid_messages:
        return []
    
    # Limit messages for memory efficiency on Render
    if len(valid_messages) > 500:
        valid_messages = valid_messages[:500]
    
    stopwords = set([
        'the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 
        'this', 'that', 'it', 'as', 'are', 'was', 'from', 'or', 'but', 'not', 'have', 'has', 'had', 
        'you', 'i', 'we', 'they', 'he', 'she', 'his', 'her', 'them', 'our', 'your', 'my', 'me', 
        'so', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'about', 'just', 
        'if', 'then', 'than', 'too', 'very', 'all', 'any', 'some', 'no', 'yes', 'one', 'two', 
        'up', 'down', 'out', 'over', 'under', 'again', 'more', 'most', 'such', 'only', 'own', 
        'same', 'other', 'new', 'now', 'after', 'before', 'because', 'how', 'when', 'where', 
        'who', 'what', 'which', 'why', 'whom', 'whose', 'been', 'being', 'into', 'during', 
        'while', 'through', 'each', 'few', 'many', 'much', 'every', 'both', 'either', 'neither', 
        'between', 'among', 'against', 'per', 'via', 'like', 'unlike', 'within', 'without', 
        'across', 'toward', 'upon', 'off', 'onto', 'beside', 'besides', 'along', 'around', 
        'behind', 'beyond', 'despite', 'except', 'inside', 'outside', 'past', 'since', 'until', 
        'upon', 'via', 'within', 'without'
    ])
    
    processed_messages = []
    for msg in valid_messages:
        text = msg.get('message', '').lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z ]', ' ', text)
        processed_messages.append(text)
    
    # If after processing we have no valid text, return empty list
    if not any(processed_messages):
        return []
    
    all_text = ' '.join(processed_messages)
    words = [word for word in all_text.split() if word not in stopwords and len(word) > TOPIC_MIN_WORD_LENGTH]
    
    # Reduce max features for memory efficiency
    vectorizer = TfidfVectorizer(max_features=500, stop_words=list(stopwords))
    tfidf_matrix = vectorizer.fit_transform(processed_messages)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    lda_topics = []
    # Only run LDA if we have enough messages and features
    if len(processed_messages) > 10 and tfidf_matrix.shape[1] > 10:
        # Reduce number of components for memory efficiency
        n_components = min(min(top_n, TOPIC_MAX_TOPICS), len(processed_messages), tfidf_matrix.shape[1])
        if n_components > 1:
            lda = LatentDirichletAllocation(n_components=n_components, random_state=42, max_iter=5)
            lda.fit(tfidf_matrix)
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                lda_topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weight': float(topic.sum())
                })
    
    topics = []
    
    # Add TF-IDF keywords
    for i, (keyword, score) in enumerate(tfidf_keywords[:top_n]):
        examples = []
        for msg in valid_messages:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + re.escape(keyword) + r'\b', msg.get('message', '').lower()):
                examples.append({
                    'sender': msg['sender'],
                    'timestamp': msg['timestamp'],
                    'message': msg['message'][:100] + '...' if len(msg['message']) > 100 else msg['message']
                })
                if len(examples) >= 2: 
                    break
        
        # Only add topic if we found at least one example
        if examples:
            topics.append({
                'topic': keyword,
                'method': 'tfidf',
                'score': float(score),
                'examples': examples
            })
    
    # Add LDA topics
    for topic in lda_topics:
        examples = []
        topic_words = set(topic['words'])
        for msg in valid_messages:
            # Preprocess message text the same way as during training
            msg_text = msg.get('message', '').lower()
            msg_text = re.sub(r'[^a-z ]', ' ', msg_text)
            msg_words = set(msg_text.split())
            
            # Check if any topic words appear in the message
            if topic_words.intersection(msg_words):
                examples.append({
                    'sender': msg['sender'],
                    'timestamp': msg['timestamp'],
                    'message': msg['message'][:100] + '...' if len(msg['message']) > 100 else msg['message']
                })
                if len(examples) >= 2:  
                    break
        
        # Only add topic if we found at least one example
        if examples:
            topics.append({
                'topic': ', '.join(topic['words']),
                'method': 'lda',
                'score': topic['weight'],
                'examples': examples
            })
    
    # Sort topics by score and return top_n
    topics = sorted(topics, key=lambda x: x['score'], reverse=True)
    return topics[:top_n]
import google.generativeai as genai
from django.conf import settings
import json
import re
from .config import SENTIMENT_THRESHOLD
from .utils import parse_timestamp
import logging
from typing import Dict, List, Any
import time
from collections import defaultdict
import gc  # Import garbage collector

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Configure Gemini AI
if settings.GEMINI_API_KEY:
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize the model with correct name
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Successfully initialized gemini-2.0-flash model")
    except Exception as e:
        logger.error(f"❌ Could not initialize gemini-2.0-flash: {e}")
        try:
            model = genai.GenerativeModel('gemini-flash-latest')
            print("✅ Fallback: Successfully initialized gemini-flash-latest model")
        except Exception as e2:
            logger.error(f"❌ Could not initialize any Gemini model: {e2}")
            model = None

def batch_analyze_sentiment_with_gemini(messages_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of messages using Gemini AI
    """
    if not messages_batch:
        return []
    
    # Prepare messages for analysis
    messages_text = []
    for i, msg in enumerate(messages_batch):
        messages_text.append(f"Message {i+1}: {msg['message']}")
    
    # Create the prompt for batch analysis with improved instructions
    messages_joined = "\n".join(messages_text)
    prompt = f"""You are an expert sentiment analyzer for WhatsApp chat messages. Analyze the sentiment of these {len(messages_batch)} messages carefully.

IMPORTANT GUIDELINES:
- Be more decisive in classification - avoid defaulting to neutral
- Positive: happiness, excitement, gratitude, love, enthusiasm, agreement, support
- Negative: anger, sadness, frustration, disappointment, fear, criticism, complaints
- Neutral: only for truly factual/informational messages with no emotional tone

For each message, provide:
1. Overall sentiment: "positive", "neutral", or "negative" (be decisive!)
2. Confidence score: 0.0 to 1.0
3. Emotion category: "joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"
4. Key emotional indicators: list of words/phrases that influenced the sentiment
5. Reason for classification: brief explanation

Messages to analyze:
{messages_joined}

Respond in this exact JSON format:
{{
  "results": [
    {{
      "message_index": 1,
      "sentiment": "positive/neutral/negative",
      "confidence": 0.85,
      "emotion": "joy/anger/sadness/fear/surprise/disgust/neutral",
      "emotional_indicators": ["happy", "great", "love"],
      "reason": "Message expresses happiness and positive emotions",
      "polarity_score": 0.7
    }}
  ]
}}

Ensure valid JSON with all {len(messages_batch)} messages analyzed."""
    
    try:
        # Check if model is available
        if model is None:
            logger.error("Gemini model not initialized")
            raise Exception("Gemini model not available")
            
        # Generate response using Gemini
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        if response_text.startswith('```json'):  
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        # Parse the JSON response
        try:
            analysis_results = json.loads(response_text)
            return analysis_results.get('results', [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Response: {response_text[:500]}")
            # Return fallback results
            return [{
                "message_index": i+1,
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "neutral",
                "emotional_indicators": [],
                "reason": "Analysis failed, defaulting to neutral",
                "polarity_score": 0.0
            } for i in range(len(messages_batch))]
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        
        # Enhanced fallback with basic sentiment analysis
        print(f"\u26a0\ufe0f Gemini API unavailable (quota exceeded), using enhanced fallback analysis...")
        
        fallback_results = []
        for i, msg in enumerate(messages_batch):
            sentiment, confidence, emotion, polarity = analyze_with_fallback(msg['message'])
            
            fallback_results.append({
                "message_index": i+1,
                "sentiment": sentiment,
                "confidence": confidence,
                "emotion": emotion,
                "emotional_indicators": get_emotional_indicators(msg['message'], sentiment),
                "reason": f"Fallback analysis: {sentiment} sentiment detected",
                "polarity_score": polarity
            })
        
        return fallback_results
    finally:
        # Clear GPU memory if using CUDA
        if hasattr(genai, 'cuda') and genai.cuda.is_available():
            genai.cuda.empty_cache()
        # Force garbage collection
        gc.collect()

def analyze_sentiment(messages):
    if not messages:
        return {"error": "No messages found"}
    
    print(f"Starting Enhanced Gemini sentiment analysis for {len(messages)} messages")
    
    sentiment_data = {
        'overall_sentiment': {'positive': 0, 'neutral': 0, 'negative': 0},
        'sentiment_breakdown': {'positive': 0, 'neutral': 0, 'negative': 0},
        'daily_sentiment': {},
        'user_sentiment': {},
        'user_sentiments': {},
        'sentiment_trend': [],
        'emotional_keywords': {'positive': [], 'negative': []},
        'sentiment_scores': [],
        'negative_messages': [],
        'all_messages_with_sentiment': [],
        'emotion_analysis': defaultdict(int),  # Emotion categories
        'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},  # Confidence levels
        'gemini_insights': {  # AI-generated insights
            'top_emotions': [],
            'mood_patterns': [],
            'communication_style': "",
            'key_findings': [],
            'recommendations': []
        },
        'analysis_metadata': {  # New: Analysis details
            'total_processed': 0,
            'processing_time': None,
            'api_calls_made': 0,
            'fallback_count': 0,
            'memory_optimized': True
        }
    }
    
    import time
    start_time = time.time()
    
    # Process messages in batches of 8 for optimal API usage
    batch_size = 8
    total_batches = (len(messages) + batch_size - 1) // batch_size
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        batch_num = i//batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)")
        
        # Analyze batch with Gemini
        batch_results = batch_analyze_sentiment_with_gemini(batch)
        sentiment_data['analysis_metadata']['api_calls_made'] += 1
        
        # Process results immediately to save memory
        for j, result in enumerate(batch_results):
            if j >= len(batch):  # Safety check
                continue
                
            msg = batch[j]
            timestamp = parse_timestamp(msg['timestamp'])
            date_str = timestamp.strftime('%Y-%m-%d') if timestamp else 'unknown'
            
            # Initialize daily sentiment if needed
            if date_str not in sentiment_data['daily_sentiment']:
                sentiment_data['daily_sentiment'][date_str] = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            # Extract sentiment information from Gemini results
            sentiment = result.get('sentiment', 'neutral').lower()
            confidence = result.get('confidence', 0.5)
            emotion = result.get('emotion', 'neutral')
            polarity_score = result.get('polarity_score', 0.0)
            emotional_indicators = result.get('emotional_indicators', [])
            reason = result.get('reason', '')
            
            # Handle fallback cases
            if sentiment not in ['positive', 'neutral', 'negative']:
                sentiment = 'neutral'
                sentiment_data['analysis_metadata']['fallback_count'] += 1
            
            # Update sentiment counts
            sentiment_data['overall_sentiment'][sentiment] += 1
            sentiment_data['sentiment_breakdown'][sentiment] += 1
            sentiment_data['daily_sentiment'][date_str][sentiment] += 1
            
            # Update emotion analysis
            sentiment_data['emotion_analysis'][emotion] += 1
            
            # Update confidence distribution
            if confidence >= 0.8:
                sentiment_data['confidence_distribution']['high'] += 1
            elif confidence >= 0.6:
                sentiment_data['confidence_distribution']['medium'] += 1
            else:
                sentiment_data['confidence_distribution']['low'] += 1
            
            # Update user sentiment
            user = msg['sender']
            if user not in sentiment_data['user_sentiment']:
                sentiment_data['user_sentiment'][user] = {'positive': 0, 'neutral': 0, 'negative': 0}
            if user not in sentiment_data['user_sentiments']:
                sentiment_data['user_sentiments'][user] = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            sentiment_data['user_sentiment'][user][sentiment] += 1
            sentiment_data['user_sentiments'][user][sentiment] += 1
            
            # Store sentiment scores
            sentiment_data['sentiment_scores'].append({
                'timestamp': msg['timestamp'],
                'sender': user,
                'polarity': polarity_score,
                'confidence': confidence,
                'emotion': emotion
            })
            
            # Store complete message info with sentiment
            message_with_sentiment = {
                'timestamp': msg['timestamp'],
                'sender': user,
                'message': msg['message'],
                'sentiment': sentiment,
                'polarity': polarity_score,
                'confidence': confidence,
                'emotion': emotion,
                'emotional_indicators': emotional_indicators,
                'reason': reason,
                'date': date_str
            }
            sentiment_data['all_messages_with_sentiment'].append(message_with_sentiment)
            
            # Store negative messages for drill-down
            if sentiment == 'negative':
                negative_message_detail = {
                    'timestamp': msg['timestamp'],
                    'sender': user,
                    'message': msg['message'],
                    'polarity': polarity_score,
                    'confidence': confidence,
                    'emotion': emotion,
                    'reason': reason,
                    'emotional_indicators': emotional_indicators,
                    'date': date_str
                }
                sentiment_data['negative_messages'].append(negative_message_detail)
            
            # Add to sentiment trend
            if timestamp:
                sentiment_data['sentiment_trend'].append({
                    'date': date_str,
                    'sentiment': sentiment,
                    'polarity': polarity_score,
                    'confidence': confidence,
                    'emotion': emotion
                })
            
            # Extract emotional keywords
            for indicator in emotional_indicators:
                keyword_entry = {
                    'keyword': indicator,
                    'message': msg['message'][:100] + '...' if len(msg['message']) > 100 else msg['message'],
                    'sender': user,
                    'timestamp': msg['timestamp'],
                    'polarity': polarity_score,
                    'confidence': confidence
                }
                
                if sentiment == 'positive':
                    sentiment_data['emotional_keywords']['positive'].append(keyword_entry)
                elif sentiment == 'negative':
                    sentiment_data['emotional_keywords']['negative'].append(keyword_entry)
        
        # Add a small delay to respect API rate limits
        time.sleep(0.5)  # Increased delay for more conservative rate limiting
    
    # Calculate processing time
    processing_time = time.time() - start_time
    sentiment_data['analysis_metadata']['processing_time'] = round(processing_time, 2)
    sentiment_data['analysis_metadata']['total_processed'] = len(messages)
    
    # Generate AI insights
    sentiment_data['gemini_insights'] = generate_gemini_insights(sentiment_data)
    
    # Add useful calculated metrics
    total_messages = sum(sentiment_data['overall_sentiment'].values())
    sentiment_data['total_analyzed'] = total_messages
    
    if total_messages > 0:
        sentiment_data['sentiment_percentages'] = {
            'positive': round((sentiment_data['overall_sentiment']['positive'] / total_messages) * 100, 1),
            'neutral': round((sentiment_data['overall_sentiment']['neutral'] / total_messages) * 100, 1),
            'negative': round((sentiment_data['overall_sentiment']['negative'] / total_messages) * 100, 1)
        }
    
    print(f"Enhanced Gemini sentiment analysis completed in {processing_time:.2f}s")
    print(f"Results: {sentiment_data['overall_sentiment']} ({sentiment_data['analysis_metadata']['api_calls_made']} API calls)")
    print(f"Confidence distribution: {sentiment_data['confidence_distribution']}")
    
    return sentiment_data

def generate_gemini_insights(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate high-level insights about the conversation using Gemini
    """
    try:
        # Prepare summary data for insight generation
        total_messages = sum(sentiment_data['overall_sentiment'].values())
        emotion_summary = dict(sentiment_data['emotion_analysis'])
        top_emotions = sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Create sample of messages for analysis
        sample_messages = sentiment_data['all_messages_with_sentiment'][:20]  # Analyze first 20 messages
        
        messages_for_analysis = "\n".join([
            f"[{msg['sentiment'].upper()}] {msg['sender']}: {msg['message'][:100]}"
            for msg in sample_messages
        ])
        
        insight_prompt = f"""
Analyze this WhatsApp conversation and provide insights:

Conversation Statistics:
- Total messages: {total_messages}
- Sentiment breakdown: {sentiment_data['overall_sentiment']}
- Top emotions: {[f"{emotion}: {count}" for emotion, count in top_emotions]}
- Confidence distribution: {sentiment_data['confidence_distribution']}

Sample messages:
{messages_for_analysis}

Provide insights in the following JSON format:
{{
  "communication_style": "Brief description of overall communication style",
  "mood_patterns": ["pattern1", "pattern2", "pattern3"],
  "key_findings": ["finding1", "finding2", "finding3"],
  "recommendations": ["recommendation1", "recommendation2"]
}}
"""
        
        # Check if model is available
        if model is None:
            logger.warning("Gemini model not available for insights generation")
            return {
                'communication_style': 'Analysis unavailable - model not initialized',
                'mood_patterns': ['Unable to analyze patterns - model not available'],
                'key_findings': ['Insight generation failed - model not initialized'],
                'recommendations': ['Please check Gemini API configuration'],
                'top_emotions': [f"{emotion} ({count})" for emotion, count in sentiment_data['emotion_analysis'].items()][:3]
            }
            
        response = model.generate_content(insight_prompt)
        response_text = response.text.strip()
        
        # Clean up JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        insights = json.loads(response_text)
        insights['top_emotions'] = [f"{emotion} ({count} messages)" for emotion, count in top_emotions]
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {
            'communication_style': 'Analysis unavailable',
            'mood_patterns': ['Unable to analyze patterns'],
            'key_findings': ['Insight generation failed'],
            'recommendations': ['Please try again later'],
            'top_emotions': [f"{emotion} ({count})" for emotion, count in sentiment_data['emotion_analysis'].items()][:3]
        }

def get_negative_reason(message, gemini_analysis=None, polarity_score=0.0):
    """
    Enhanced negative reason detection using Gemini analysis
    """
    if gemini_analysis:
        return gemini_analysis.get('reason', 'Negative sentiment detected by AI analysis')
    
    # Fallback to basic analysis if Gemini analysis not available
    message_lower = message.lower()
    reasons = []
    
    # Basic negative indicators
    anger_words = ['angry', 'furious', 'rage', 'mad', 'pissed', 'annoyed', 'frustrated', 'irritated']
    sadness_words = ['sad', 'depressed', 'miserable', 'devastated', 'heartbroken', 'crying', 'upset']
    fear_words = ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'concerned']
    disgust_words = ['disgusted', 'sick', 'revolted', 'gross', 'awful', 'terrible', 'horrible']
    
    if any(word in message_lower for word in anger_words):
        reasons.append("Contains expressions of anger or frustration")
    if any(word in message_lower for word in sadness_words):
        reasons.append("Contains expressions of sadness or distress")
    if any(word in message_lower for word in fear_words):
        reasons.append("Contains expressions of fear or anxiety")
    if any(word in message_lower for word in disgust_words):
        reasons.append("Contains expressions of disgust or strong dislike")
    
    if not reasons:
        if polarity_score < -0.5:
            reasons.append("Strongly negative overall sentiment")
        else:
            reasons.append("Negative sentiment detected")
    
    return "; ".join(reasons)

def analyze_with_fallback(message):
    """
    Enhanced fallback sentiment analysis when Gemini API is not available
    """
    message_lower = message.lower().strip()
    
    # Positive indicators
    positive_words = [
        'love', 'like', 'good', 'great', 'awesome', 'amazing', 'wonderful', 'excellent',
        'happy', 'joy', 'glad', 'excited', 'fantastic', 'perfect', 'best', 'beautiful',
        'thank', 'thanks', 'grateful', 'appreciate', 'congrats', 'congratulations',
        'yes', 'sure', 'absolutely', 'definitely', 'correct', 'right', 'agreed',
        '😊', '😍', '❤️', '👍', '😂', '🎉', '✅', '👏', '🔥', '💯'
    ]
    
    # Negative indicators  
    negative_words = [
        'hate', 'bad', 'terrible', 'awful', 'horrible', 'worst', 'ugly', 'stupid',
        'angry', 'mad', 'furious', 'frustrated', 'annoyed', 'upset', 'disappointed',
        'sad', 'depressed', 'crying', 'hurt', 'pain', 'worried', 'scared', 'afraid',
        'no', 'never', 'wrong', 'incorrect', 'disagree', 'problem', 'issue', 'error',
        '😢', '😭', '😡', '😠', '💔', '😞', '😔', '👎', '😤', '🤬'
    ]
    
    # Count positive and negative indicators
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    # Determine sentiment
    if positive_count > negative_count and positive_count > 0:
        sentiment = 'positive'
        confidence = min(0.7 + (positive_count * 0.1), 0.95)
        emotion = 'joy'
        polarity = 0.5 + (positive_count * 0.2)
    elif negative_count > positive_count and negative_count > 0:
        sentiment = 'negative'
        confidence = min(0.7 + (negative_count * 0.1), 0.95)
        emotion = 'anger' if any(word in message_lower for word in ['angry', 'mad', 'furious']) else 'sadness'
        polarity = -0.5 - (negative_count * 0.2)
    else:
        # Check for questions or informational content
        if '?' in message or any(word in message_lower for word in ['what', 'when', 'where', 'who', 'how', 'why']):
            sentiment = 'neutral'
            confidence = 0.8
            emotion = 'neutral'
            polarity = 0.0
        else:
            sentiment = 'neutral'
            confidence = 0.6
            emotion = 'neutral'
            polarity = 0.0
    
    return sentiment, confidence, emotion, polarity

def get_emotional_indicators(message, sentiment):
    """
    Extract emotional indicators from message based on sentiment
    """
    message_lower = message.lower()
    indicators = []
    
    if sentiment == 'positive':
        positive_indicators = ['love', 'like', 'good', 'great', 'awesome', 'amazing', 'happy', 'thank', 'yes']
        indicators = [word for word in positive_indicators if word in message_lower]
    elif sentiment == 'negative':
        negative_indicators = ['hate', 'bad', 'terrible', 'angry', 'sad', 'no', 'wrong', 'problem']
        indicators = [word for word in negative_indicators if word in message_lower]
    
    # Add emoji indicators
    emoji_indicators = {
        'positive': ['😊', '😍', '❤️', '👍', '😂', '🎉'],
        'negative': ['😢', '😭', '😡', '😠', '💔', '😞']
    }
    
    if sentiment in emoji_indicators:
        indicators.extend([emoji for emoji in emoji_indicators[sentiment] if emoji in message])
    
    return indicators[:5]  # Return top 5 indicators

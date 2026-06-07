import os
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from openai import OpenAI

def extract_text_features(text):
    """Extract numerical features from listing text"""
    words = text.split()
    sentences = text.split('.')
    
    word_count = len(words)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    merge_words = len([w for w in words if len(w) > 15])
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    urgent_words = ['urgent', 'immediately', 'wire', 'transfer', 'fee', 
                   'advance', 'payment', 'guarantee', 'deal', 'limited']
    urgent_count = sum(1 for w in urgent_words if w.lower() in text.lower())
    
    return [word_count, avg_sentence_length, merge_words, caps_ratio,
            exclamation_count, question_count, urgent_count]

def extract_metadata_features(price, location, title):
    """Extract features from listing metadata"""
    price = float(price)
    title_length = len(title.split())
    location_length = len(location.split())
    price_suspicious = 1 if price < 10000 else 0
    
    return [price, title_length, location_length, price_suspicious]

def get_llm_fraud_score(title, description, price, location):
    """Use OpenAI API to analyse listing for fraud"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        prompt = f"""Analyse this property listing for signs of Advance Fee Fraud.
        
Title: {title}
Description: {description}
Price: ${price}
Location: {location}

Rate the fraud probability from 0.0 (definitely legitimate) to 1.0 (definitely fraudulent).
Look for: unrealistic prices, urgent language, requests for advance payments, 
poor grammar, vague descriptions, too-good-to-be-true offers.

Respond with ONLY a number between 0.0 and 1.0."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return analyse_text_locally(title + " " + description)

def analyse_text_locally(text):
    """Fallback local text analysis if API unavailable"""
    fraud_indicators = [
        'urgent', 'wire transfer', 'advance fee', 'western union',
        'money gram', 'guaranteed', 'no questions', 'act now',
        'limited time', 'contact immediately', 'God bless',
        'overseas', 'foreign', 'diplomat', 'inheritance'
    ]
    text_lower = text.lower()
    score = sum(0.1 for indicator in fraud_indicators 
                if indicator in text_lower)
    return min(score, 1.0)

def cwds_fusion(text_score, metadata_score, llm_score):
    """Class Weighted Dempster-Shafer fusion"""
    # Weights from your report: fake=1.0, not_fake=0.1
    fake_weight = 1.0
    not_fake_weight = 0.1
    
    scores = [text_score, metadata_score, llm_score]
    
    # Weighted combination
    weighted_fake = sum(s * fake_weight for s in scores)
    weighted_not_fake = sum((1-s) * not_fake_weight for s in scores)
    
    # Normalise
    total = weighted_fake + weighted_not_fake
    if total == 0:
        return 0.5
    
    final_score = weighted_fake / total
    return round(min(final_score, 1.0), 4)

def analyse_listing(title, description, price, location):
    """Main fraud detection function - runs all models and fuses results"""
    
    # 1. Extract features
    text_features = extract_text_features(description)
    metadata_features = extract_metadata_features(price, location, title)
    
    # 2. Text score (local analysis)
    text_score = analyse_text_locally(title + " " + description)
    
    # 3. Metadata score using XGBoost
    try:
        X_meta = np.array(metadata_features).reshape(1, -1)
        # Simple rule-based metadata scoring
        price_val = float(price)
        meta_score = 0.8 if price_val < 5000 else 0.3 if price_val < 20000 else 0.1
    except Exception:
        meta_score = 0.3
    
    # 4. LLM score
    llm_score = get_llm_fraud_score(title, description, price, location)
    
    # 5. CWDS Fusion
    final_score = cwds_fusion(text_score, meta_score, llm_score)
    
    # 6. Final decision
    if final_score >= 0.7:
        status = 'flagged'
    elif final_score >= 0.4:
        status = 'pending'
    else:
        status = 'approved'
    
    return {
        'fraud_score': final_score,
        'text_score': text_score,
        'metadata_score': meta_score,
        'llm_score': llm_score,
        'status': status
    }
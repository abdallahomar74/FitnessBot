import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import string
import os

# تحميل punkt tokenizer مرة واحدة
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# إعدادات OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# دالة معالجة نصوص متقدمة للإنجليزية فقط
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# تحميل قاعدة المعرفة
with open('data/fitness_knowledge_base.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

corpus_raw = [item['question'] for item in kb if item['type'] == 'qa']
answers = [item['answer'] for item in kb if item['type'] == 'qa']

# تطبيق المعالجة المسبقة
corpus = [preprocess(q) for q in corpus_raw]

# تدريب نموذج TF-IDF مع n-grams
vectorizer = TfidfVectorizer(ngram_range=(1,2))
corpus_tfidf = vectorizer.fit_transform(corpus)

# دالة Jaccard similarity
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return float(len(intersection)) / len(union)

# إنشاء تطبيق FastAPI
app = FastAPI(title="Semantic Fitness Chatbot (LLM on low similarity only)")

class Query(BaseModel):
    question: str
    use_llm: bool = True

# حساب أفضل إجابة من قاعدة المعرفة (ensemble)
def get_best_answers(question, top_k=1):
    preprocessed_q = preprocess(question)
    question_tfidf = vectorizer.transform([preprocessed_q])
    cos_scores = cosine_similarity(question_tfidf, corpus_tfidf)[0]
    jaccard_scores = np.array([jaccard_similarity(preprocessed_q, c) for c in corpus])
    # ensemble: weighted average (يمكنك تعديل الوزن حسب التجربة)
    ensemble_scores = 0.7 * cos_scores + 0.3 * jaccard_scores
    top_results = np.argpartition(-ensemble_scores, range(top_k))[:top_k]
    results = []
    for idx in top_results:
        results.append({
            "question": corpus_raw[idx],
            "answer": answers[idx],
            "similarity": float(ensemble_scores[idx])
        })
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results

# تحسين الإجابة باستخدام LLM من OpenRouter
def polish_with_openrouter(question, base_answer):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Fitness Chatbot"
    }
    if base_answer.strip():
        prompt = (
            f"You are a fitness expert chatbot. "
            f"User asked: '{question}'\n"
            f"Here is a factual answer from the knowledge base: '{base_answer}'\n"
            f"Rewrite the answer in a more detailed, practical, and actionable way. "
            f"Expand on the core principle, add 2-3 specific strategies, and finish with a practical tip. "
            f"Do NOT just repeat the original answer. Make it more helpful and professional."
        )
    else:
        prompt = (
            f"You are a fitness expert chatbot. "
            f"User asked: '{question}'\n"
            f"Provide a detailed, practical, and actionable answer. "
            f"Include 2-3 specific strategies and finish with a practical tip."
        )
    data = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        print(result)
        return result['choices'][0]['message']['content'].strip()
    return base_answer

@app.post("/chat")
def chat(q: Query):
    user_q = q.question.strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    results = get_best_answers(user_q, 1)
    similarity = results[0]["similarity"]
    matched_question = results[0]["question"]

    if similarity > 0.6:
        return {
            "answer": results[0]["answer"],
            "matched_question": matched_question,
            "similarity": similarity,
            "used_llm": False
        }
    else:
        improved = polish_with_openrouter(user_q, "")
        return {
            "answer": improved,
            "matched_question": matched_question,
            "similarity": similarity,
            "used_llm": True
        }

@app.get("/")
def root():
    return {"message": "Semantic Fitness Chatbot is running!"}

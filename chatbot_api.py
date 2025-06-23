import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

with open('data/fitness_knowledge_base.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)
corpus = [item['question'] for item in kb if item['type'] == 'qa']
answers = [item['answer'] for item in kb if item['type'] == 'qa']
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

app = FastAPI(title="Semantic Fitness Chatbot (LLM on low similarity only)")

class Query(BaseModel):
    question: str
    use_llm: bool = True

def get_best_answers(question, top_k=1):
    question_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    results = []
    for idx in top_results:
        results.append({
            "question": corpus[idx],
            "answer": answers[idx],
            "similarity": float(cos_scores[idx])
        })
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results

def summarize_text(text, sentence_count=2):
    if len(text.split()) < 40:
        return text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    result = " ".join(str(sentence) for sentence in summary)
    return result if result else text

def polish_with_openrouter(question, base_answer):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Fitness Chatbot"
    }
    prompt = (
        f"You are a fitness expert chatbot. "
        f"User asked: '{question}'\n"
        f"Here is a factual answer from the knowledge base: '{base_answer}'\n"
        f"Rewrite the answer in a more detailed, practical, and actionable way. "
        f"Expand on the core principle, add 2-3 specific strategies, and finish with a practical tip. "
        f"Do NOT just repeat the original answer. Make it more helpful and professional."
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
        return result['choices'][0]['message']['content'].strip()
    return base_answer

def polish_with_llm(question, base_answer):
    return polish_with_openrouter(question, base_answer)

@app.post("/chat")
def chat(q: Query):
    user_q = q.question.strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    results = get_best_answers(user_q, 1)
    similarity = results[0]["similarity"]
    matched_question = results[0]["question"]
    if similarity > 0.6:
        summarized = summarize_text(results[0]["answer"], sentence_count=2)
        return {
            "answer": summarized,
            "matched_question": matched_question,
            "similarity": similarity,
            "used_llm": False
        }
    else:
        improved = polish_with_llm(user_q, "")
        return {
            "answer": improved,
            "matched_question": matched_question,
            "similarity": similarity,
            "used_llm": True
        }

@app.get("/")
def root():
    return {"message": "Semantic Fitness Chatbot (LLM on low similarity only) is running!"}

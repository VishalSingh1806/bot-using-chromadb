from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional
from fuzzywuzzy import fuzz
import smtplib
import logging
import asyncio
import os
import redis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import traceback
from collections import Counter
from datetime import datetime
import smtplib
import socket  # Required for network error handling
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from db_import import collection, embedding_model, feedback_manager
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
import random
from spellchecker import SpellChecker
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
from intent import predict_intent
# Load environment variables
load_dotenv()


paraphrase_model_path = "./custom_paraphrase_model"
tokenizer = T5Tokenizer.from_pretrained(paraphrase_model_path)
model = T5ForConditionalGeneration.from_pretrained(paraphrase_model_path)

paraphraser = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # 🚀 Use GPU if available
)

# Email batch configuration
email_batch = []
batch_size = 1
SIMILARITY_THRESHOLD = 0.5
FUZZY_MATCH_THRESHOLD = 85

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Redis Configuration
def is_docker():
    return os.path.exists('/.dockerenv')

if is_docker():
    REDIS_HOST = os.getenv('REDIS_HOST_SERVER', '172.17.0.1')
    REDIS_PORT = int(os.getenv('REDIS_PORT_SERVER', '6379'))
    logger.info("🚀 Running inside Docker. Using server Redis config.")
else:
    REDIS_HOST = os.getenv('REDIS_HOST_LOCAL', '127.0.0.1')
    REDIS_PORT = int(os.getenv('REDIS_PORT_LOCAL', '6379'))
    logger.info("💻 Running locally. Using local Redis config.")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    logger.info(f"✅ Redis connected successfully at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.RedisError as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    redis_client = None  # fallback or handle accordingly


logger.info(f"SMTP Server: {SMTP_SERVER}")  
logger.info(f"SMTP Port: {SMTP_PORT}")
logger.info(f"SMTP Username: {SMTP_USERNAME}")
logger.info(f"SMTP Password: {SMTP_PASSWORD is not None}")
logger.info(f"Recipient Email: {RECIPIENT_EMAIL}")
logger.info(f"Redis Host: {REDIS_HOST}")
logger.info(f"Redis Port: {REDIS_PORT}")

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USERNAME, SMTP_PASSWORD)
    print("✅ SMTP Connection Successful!")
    server.quit()
except smtplib.SMTPAuthenticationError:
    print("❌ Authentication Error! Check username/password.")
except smtplib.SMTPConnectError:
    print("❌ Unable to connect to SMTP server. Check internet or firewall.")
except Exception as e:
    print(f"❌ Other SMTP Error: {e}")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# Pydantic models with updated validation
class Query(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, max_length=1000)
    n_results: Optional[int] = 5

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "What is machine learning?",
                "n_results": 5
            }
        }

class UserData(BaseModel):
    session_id: str
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., pattern=r'^\d{10}$')
    organization: str = Field(..., min_length=2, max_length=100)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @validator('organization')
    def validate_organization(cls, v):
        if not v.strip():
            raise ValueError('Organization cannot be empty')
        return v.strip()

class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    question_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "query": "What is AI?",
                "question_id": "q123",
                "relevance_score": 0.95
            }
        }

# Routes
@app.get("/")
async def read_root():
    return FileResponse("template/index.html")

def clean_query(text: str) -> str:
    spell = SpellChecker()
    words = text.split()
    corrected = [spell.correction(word) or word for word in words]
    return " ".join(corrected)

@app.post("/query")
async def query_database(query: Query):
    try:
        if not query.session_id:
            logger.info("No session found. Redirecting to form")
            return JSONResponse(content={
                "redirect_to": "/collect_user_data",
                "message": "Please complete the form first!"
            })

        session_key = f"session:{query.session_id}"
        user_data_collected = redis_client.hget(session_key, "user_data_collected")
        redis_client.hset(session_key, "last_interaction", datetime.utcnow().isoformat())

        if not user_data_collected or user_data_collected != "true":
            logger.info("⚠️ User data not collected, redirecting to form")
            return JSONResponse(content={
                "redirect_to": "/collect_user_data",
                "message": "Please complete the form first!"
            })

        logger.info(f"📥 Received raw query: {query.text}")
        paraphrases_used = []

        # ✅ Predict intent early so it's in scope below
        predicted_intent = predict_intent(query.text)
        logger.info(f"🎯 Predicted intent: {predicted_intent}")

        def run_similarity_search(text: str):
            query_embedding = embedding_model.encode([text])[0].tolist()
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=query.n_results,
                include=["metadatas", "embeddings"]
            )
            return query_embedding_np, results

        def filter_results(query_text: str, query_embedding_np, results):
            processed_results = []
            question_set = set()
            retrieved_embeddings_np = np.array(results["embeddings"][0])
            cos_similarities = cosine_similarity(query_embedding_np, retrieved_embeddings_np)[0]

            for i, result_id in enumerate(results['ids'][0]):
                similarity = cos_similarities[i]
                metadata = results['metadatas'][0][i]
                question = metadata.get('question', '').strip()
                answer = metadata.get('answer', '').strip()
                record_intent = metadata.get('intent', '').strip().lower()

                if not question:
                    continue

                is_cosine_match = similarity >= SIMILARITY_THRESHOLD
                is_fuzzy_match = fuzz.ratio(query_text.lower(), question.lower()) >= FUZZY_MATCH_THRESHOLD
                is_intent_match = record_intent == predicted_intent.lower()

                adjusted_similarity = similarity + (0.05 if is_intent_match else 0)

                if is_cosine_match or is_fuzzy_match:
                    if question not in question_set:
                        processed_results.append({
                            'id': result_id,
                            'question': question,
                            'answer': answer,
                            'similarity': round(adjusted_similarity, 4),
                            'intent_match': is_intent_match,
                            'fuzzy_score': fuzz.ratio(query_text.lower(), question.lower())
                        })
                        question_set.add(question)

            processed_results.sort(key=lambda x: (x['similarity'] + x['fuzzy_score'] / 100), reverse=True)
            return processed_results

        # Step 1: Try raw query
        query_embedding_np, results = run_similarity_search(query.text)
        processed_results = filter_results(query.text, query_embedding_np, results)
        cache_key = f"query:{query.text}"

        # Step 2: If no result, try cleaned query
        if not processed_results:
            cleaned_query = clean_query(query.text)
            logger.info(f"🧽 Cleaned query: {cleaned_query}")
            query_embedding_np, results = run_similarity_search(cleaned_query)
            processed_results = filter_results(cleaned_query, query_embedding_np, results)
            cache_key = f"query:{cleaned_query}"

        # Step 3: If still no result, paraphrase original query
        if not processed_results:
            logger.info("⚠ No confident results, paraphrasing query for fallback...")
            paraphrases_used = generate_paraphrases(query.text)
            logger.info(f"🌀 Paraphrased versions of '{query.text}': {paraphrases_used}")

            for para in paraphrases_used:
                para_embedding_np, para_results = run_similarity_search(para)
                para_filtered = filter_results(para, para_embedding_np, para_results)
                if para_filtered:
                    logger.info(f"✅ Paraphrase successful: {para}")
                    processed_results = para_filtered
                    query_embedding_np = para_embedding_np
                    break

        if not processed_results:
            logger.warning("⚠ Still no relevant results after fallback")
            return {
                "results": [ { "id": None, "answer": "I'm not confident I have a relevant answer. Could you please rephrase or ask something else?" } ],
                "similar_questions": fallback_random_suggestions(),
                "paraphrased_queries": paraphrases_used
            }

        best_answer = processed_results[0]
        refined_similar_questions = get_similar_questions(query_embedding_np, query.text)
        if not refined_similar_questions:
            logger.warning("⚠ No refined similar questions found, using fallback suggestions.")
            refined_similar_questions = fallback_random_suggestions()

        response_data = {
            "results": [best_answer],
            "query": query.text,
            "predicted_intent": predicted_intent,
            "similar_questions": refined_similar_questions,
            "paraphrased_queries": paraphrases_used
        }

        redis_client.setex(cache_key, 120, json.dumps(response_data))
        return response_data

    except Exception as e:
        logger.error(f"❌ Query error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred")



def generate_paraphrases(text: str, num_return_sequences: int = 3) -> list:
    try:
        outputs = paraphraser(
            f"paraphrase: {text}",
            max_length=64,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=120,
            top_p=0.95
        )
        results = list({o['generated_text'].strip() for o in outputs})
        logger.info(f"🌀 Paraphrased versions of '{text}': {results}")
        return results
    except Exception as e:
        logger.error(f"⚠️ Paraphrasing failed: {e}")
        return []



def fallback_random_suggestions():
    fallback_pool = [
        "What is EPR registration?",
        "How can I apply for a plastic waste certificate?",
        "What are the responsibilities of a brand owner?",
        "Do I need to submit monthly reports?",
        "How is EPR compliance verified?",
        "What documents are needed for CPCB registration?",
        "Who qualifies as a PIBO?",
        "How does ReCircle help with plastic credit?",
        "What is the penalty for non-compliance?",
        "Can you help me with recycling partners?"
    ]
    return random.sample(fallback_pool, 3)

def get_similar_questions(query_embedding, query_text):
    try:
        query_embedding_np = query_embedding if isinstance(query_embedding, np.ndarray) else np.array(query_embedding).reshape(1, -1)
        query_embedding_list = query_embedding_np.flatten().tolist()

        more_results = collection.query(query_embeddings=[query_embedding_list], n_results=15, include=["metadatas", "embeddings"])
        more_embeddings = np.array(more_results["embeddings"][0])
        more_similarities = cosine_similarity(query_embedding_np, more_embeddings)[0]

        candidate_list = sorted(
            zip(more_similarities, more_results['metadatas'][0]),
            key=lambda x: x[0],
            reverse=True
        )

        refined = []
        used = {query_text.lower()}

        for sim_score, metadata in candidate_list:
            q = metadata.get('question', '').strip()
            if not q or q.lower() in used:
                continue
            fuzzy_score = fuzz.ratio(query_text.lower(), q.lower())
            if sim_score >= 0.6 or fuzzy_score >= FUZZY_MATCH_THRESHOLD:
                refined.append(q)
                used.add(q.lower())
            if len(refined) >= 3:
                break

        return refined
    except Exception as e:
        logger.error(f"⚠ Error getting similar questions: {e}")
        return []


# Add new endpoint to get chat history
@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        chat_key = f"chat:{session_id}"
        messages = redis_client.lrange(chat_key, 0, -1)
        return {
            "messages": [json.loads(msg) for msg in messages]
        }
    except Exception as e:
        logger.error(f"❌ Error fetching chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch chat history"
        )

@app.post("/collect_user_data")
async def collect_user_data(user_data: UserData):
    try:
# Save to Redis
        session_key = f"session:{user_data.session_id}"
        redis_client.hset(session_key, "user_data_collected", "true")
        redis_client.hset(session_key, "user_name", user_data.name)
        redis_client.hset(session_key, "email", user_data.email)
        redis_client.hset(session_key, "phone", user_data.phone)
        redis_client.hset(session_key, "organization", user_data.organization)
        redis_client.hset(session_key, "last_interaction", datetime.utcnow().isoformat())
        
        # Add to email batch
        email_batch.append(user_data.dict())
        
        # Send batch if size limit reached
        if len(email_batch) >= batch_size:
            try:
                await send_email_batch(email_batch)
                email_batch.clear()
            except Exception as e:
                logger.error(f"❌ Failed to send email batch: {e}")

        return JSONResponse(
            content={"message": "User data collected successfully"},
            status_code=200
        )

    except Exception as e:
        logger.error(f"❌ Error collecting user data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to collect user data"
        )

async def send_email_batch(batch):
    """Send a batch of emails using a single SMTP connection."""
    if not SMTP_SERVER or not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.error("❌ Missing SMTP configuration. Check environment variables.")
        return

    try:
        logger.info(f"📧 Connecting to SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.set_debuglevel(1)  # Enable debugging for SMTP connection
            server.starttls()
            
            try:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"❌ SMTP Authentication Error: {e}")
                return
            except smtplib.SMTPException as e:
                logger.error(f"❌ General SMTP Error: {e}")
                return
            
            for user_data in batch:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = SMTP_USERNAME
                    msg['To'] = RECIPIENT_EMAIL
                    msg['Subject'] = "New User Registration"

                    body = f"""
                    New User Registration:
                    Name: {user_data['name']}
                    Email: {user_data['email']}
                    Phone: {user_data['phone']}
                    Organization: {user_data['organization']}
                    Time: {datetime.utcnow().isoformat()}
                    """

                    msg.attach(MIMEText(body, 'plain'))
                    server.send_message(msg)
                    logger.info(f"✅ Email sent for user: {user_data['email']}")

                except smtplib.SMTPRecipientsRefused:
                    logger.error(f"❌ Recipient email refused: {RECIPIENT_EMAIL}")
                except smtplib.SMTPException as e:
                    logger.error(f"❌ Failed to send email for {user_data['email']}: {e}")
                except Exception as e:
                    logger.error(f"❌ Unexpected error: {e}")

    except (socket.gaierror, smtplib.SMTPConnectError) as e:
        logger.error(f"❌ Network error while connecting to SMTP server: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error in email batch: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
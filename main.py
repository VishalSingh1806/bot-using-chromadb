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

# Load environment variables
load_dotenv()

# Email batch configuration
email_batch = []
batch_size = 1
SIMILARITY_THRESHOLD = 0.5

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

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))

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
    allow_headers=["*"],
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


@app.post("/query")
async def query_database(query: Query):
    try:
        # 1️⃣ Session validation
        if not query.session_id:
            logger.info("No session found. Redirecting to form")
            return JSONResponse(content={
                "redirect_to": "/collect_user_data",
                "message": "Please complete the form first!"
            })

        session_key = f"session:{query.session_id}"
        user_data_collected = redis_client.hget(session_key, "user_data_collected")

        if not user_data_collected or user_data_collected != "true":
            logger.info("⚠️ User data not collected, redirecting to form")
            return JSONResponse(content={
                "redirect_to": "/collect_user_data",
                "message": "Please complete the form first!"
            })

        logger.info(f"📥 Received query: {query.text}")

        # 2️⃣ Cache check
        cache_key = f"query:{query.text}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info("✅ Cache hit")
            return json.loads(cached_result)

        # 3️⃣ Generate embedding
        query_embedding = embedding_model.encode([query.text])[0].tolist()
        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        # 4️⃣ Retrieve initial results
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            include=["metadatas", "embeddings"]
        )

        if not results['ids'][0]:
            logger.warning("⚠ No results found")
            return {
                "results": [{
                    "id": None,
                    "answer": "Sorry, I couldn't find an answer to your question."
                }],
                "similar_questions": []
            }

        # 5️⃣ Compute Cosine Similarity and filter
        retrieved_embeddings_np = np.array(results["embeddings"][0])
        cos_similarities = cosine_similarity(query_embedding_np, retrieved_embeddings_np)[0]

        processed_results = []
        question_set = set()  # track questions to avoid duplicates

        logger.info(f"Processing {len(results['ids'][0])} results with "
                    f"similarity threshold: {SIMILARITY_THRESHOLD}")

        for i, result_id in enumerate(results['ids'][0]):
            similarity = cos_similarities[i]
            question = results['metadatas'][0][i]['question']
            answer = results['metadatas'][0][i]['answer']

            if not question or similarity < SIMILARITY_THRESHOLD:
                # Either no question text or below threshold => skip
                continue

            # Avoid duplicates
            if question not in question_set:
                processed_results.append({
                    'id': result_id,
                    'question': question,
                    'answer': answer,
                    'similarity': round(similarity, 4),
                })
                question_set.add(question)

        # 6️⃣ Sort results by similarity
        processed_results.sort(key=lambda x: x['similarity'], reverse=True)
        best_answer = processed_results[0] if processed_results else None

        if not best_answer:
            logger.warning("No results met the similarity threshold, returning minimal suggestions.")
            return {
                "results": [{
                    "id": None,
                    "answer": (
                        "I'm not confident I have a relevant answer. "
                        "Could you please rephrase or ask something else?"
                    )
                }],
                "similar_questions": []
            }

        # 7️⃣ Prepare to fetch additional similar questions
        # Exclude the query itself and the best_answer to avoid duplicates
        excluded_questions = {query.text.lower(), best_answer['question'].lower()}

        # We'll store only the question text in our final suggestions
        refined_similar_questions = []

        try:
            # 8️⃣ Get more candidates for suggested questions
            more_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=15,
                include=["metadatas", "embeddings"]
            )

            more_embeddings = np.array(more_results["embeddings"][0])
            more_similarities = cosine_similarity(query_embedding_np, more_embeddings)[0]

            # Sort candidates by similarity (descending)
            candidate_list = sorted(
                zip(more_similarities, more_results['metadatas'][0]),
                key=lambda x: x[0],
                reverse=True
            )

            # Dynamically determine thresholds
            max_sim = candidate_list[0][0]  # highest similarity across candidates
            # a) First pass threshold is max(0.6, 0.8 * max_sim)
            first_pass_threshold = max(0.5, 0.7 * max_sim)
            # b) Second pass threshold is max(0.5, 0.6 * max_sim) => can be adjusted
            second_pass_threshold = max(0.4, 0.5 * max_sim)

            def is_near_duplicate(q1, q2, fuzz_threshold=75):
                """
                Check if q1 and q2 are near duplicates using fuzzy matching.
                Return True if they are too similar.
                """
                if not q1 or not q2:
                    return False
                return fuzz.ratio(q1.lower(), q2.lower()) >= fuzz_threshold

            def already_in_list(q, suggestions):
                """Check if 'q' is near-duplicate with any question in 'suggestions'."""
                for existing_q in suggestions:
                    if is_near_duplicate(q, existing_q):
                        return True
                return False

            # 9️⃣ First pass for suggestions
            for sim_score, metadata in candidate_list:
                q = metadata.get('question', '').strip()
                if not q:
                    continue

                # Check exclude list + near-duplicate to best_answer
                if q.lower() in excluded_questions or is_near_duplicate(q, best_answer['question']):
                    continue

                # Strict threshold (first_pass_threshold)
                if sim_score >= first_pass_threshold:
                    # Also ensure it's not near-duplicate to what's already in refined_similar_questions
                    if not already_in_list(q, refined_similar_questions):
                        refined_similar_questions.append(q)
                        excluded_questions.add(q.lower())

                if len(refined_similar_questions) >= 3:
                    break

            # 🔟 Second pass with relaxed threshold
            if len(refined_similar_questions) < 3:
                for sim_score, metadata in candidate_list:
                    q = metadata.get('question', '').strip()
                    if not q:
                        continue

                    if q.lower() in excluded_questions or is_near_duplicate(q, best_answer['question']):
                        continue

                    if sim_score >= second_pass_threshold:
                        if not already_in_list(q, refined_similar_questions):
                            refined_similar_questions.append(q)
                            excluded_questions.add(q.lower())

                    if len(refined_similar_questions) >= 3:
                        break

            logger.info(f"✅ Found {len(refined_similar_questions)} unique suggestions")

        except Exception as e:
            logger.error(f"⚠ Error fetching additional similar questions: {str(e)}")
            refined_similar_questions = []

        # 1️⃣1️⃣ Final response
        response_data = {
            "results": [best_answer],
            "query": query.text,
            "similar_questions": [q["question"] if isinstance(q, dict) else q for q in refined_similar_questions[:3]] # Ensure we sending just the question string
        }

        # 1️⃣2️⃣ Cache the response
        redis_client.setex(cache_key, 120, json.dumps(response_data))
        return response_data

    except Exception as e:
        logger.error(f"❌ Query error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )



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

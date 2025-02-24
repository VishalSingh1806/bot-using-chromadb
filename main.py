# from fastapi import FastAPI, Request, HTTPException, Form, Depends
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel, confloat
# from sentence_transformers import SentenceTransformer
# from db_import import collection, embedding_model, feedback_manager
# from typing import List, Optional

# app = FastAPI()

# # Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="template")

# # Request model for chatbot query
# class QueryRequest(BaseModel):
#     user_query: str

# # Request model for feedback submission
# class FeedbackRequest(BaseModel):
#     query: str
#     question_id: str
#     feedback: str
#     relevance_score: float  # Changed from confloat to float for simplicity

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "query": "What is machine learning?",
#                 "question_id": "q123",
#                 "feedback": "yes",
#                 "relevance_score": 0.9
#             }
#         }

# ## **2️⃣ Serve the Chatbot UI (HTML)**
# @app.get("/", response_class=HTMLResponse)
# async def serve_homepage(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# ## **3️⃣ Chatbot API Endpoint**
# @app.post("/query")
# async def query_chromadb(request: Request):
#     try:
#         # Parse and log the incoming request
#         body = await request.json()
#         print("Received Request:", body)

#         # Extract user query
#         user_query = body.get("user_query")
#         if not user_query:
#             raise HTTPException(status_code=400, detail="Missing 'user_query' field in request body.")

#         # Convert query to embeddings
#         query_embedding = embedding_model.encode([user_query]).tolist()

#         # Fetch the best matching answer with metadata and distances
#         result = collection.query(
#             query_embeddings=query_embedding, 
#             n_results=1,
#             include=["metadatas", "distances", "documents"]
#         )

#         # Check if any valid result is found
#         if not result["ids"]:
#             return JSONResponse(
#                 content={"answer": "No relevant answer found.", "suggestions": []}, 
#                 status_code=200
#             )

#         # Extract best answer and metadata
#         metadata = result["metadatas"][0][0]
#         best_answer = metadata.get("answer", "No answer available")
#         question_id = result["ids"][0][0]
#         similarity_score = 1 - (result["distances"][0][0] / 2)  # Convert distance to similarity
#         cluster = metadata.get("cluster_kmeans", None)

#         # Retrieve top 3 suggested questions if a valid cluster exists
#         suggested_questions = []
#         if cluster:
#             suggestions = collection.query(
#                 query_embeddings=query_embedding,
#                 n_results=3,  # 1 best match + 2 suggestions
#                 where={"cluster_kmeans": cluster},
#                 include=["metadatas"]
#             )

#             # Extract suggested questions (exclude the exact user query)
#             suggested_questions = [
#                 meta["question"] 
#                 for meta in suggestions["metadatas"][0] 
#                 if meta["question"] != user_query
#             ]

#         return JSONResponse(
#             content={
#                 "answer": best_answer,
#                 "question_id": question_id,
#                 "similarity_score": similarity_score,
#                 "suggestions": suggested_questions
#             }, 
#             status_code=200
#         )

#     except Exception as e:
#         print("Error processing request:", str(e))
#         return JSONResponse(
#             content={"error": "Internal server error. Check logs for details."}, 
#             status_code=500
#         )

# ## **4️⃣ Feedback API Endpoint**
# @app.post("/feedback")
# async def update_relevance(request: FeedbackRequest):
#     try:
#         print(f"Received Feedback: {request}")  # Debugging log

#         # Add feedback through feedback manager
#         await feedback_manager.add_feedback(
#             query=request.query,
#             question_id=request.question_id,
#             relevance_score=1.0 if request.feedback.lower() == "yes" else 0.0
#         )

#         return JSONResponse(
#             content={"message": "Feedback recorded successfully"}, 
#             status_code=200
#         )

#     except Exception as e:
#         print(f"Error processing feedback: {e}")
#         return JSONResponse(
#             content={"error": "Failed to process feedback"}, 
#             status_code=500
#         )

# ## **5️⃣ Feedback Stats Endpoint**
# @app.get("/feedback/stats")
# async def get_feedback_stats():
#     """Get statistics about feedback and reweighting"""
#     try:
#         history = feedback_manager.feedback_history
#         return JSONResponse(content={
#             "total_feedbacks": len(history["feedbacks"]),
#             "last_reweight": history["last_reweight"],
#             "feedback_count_since_last_reweight": feedback_manager.feedback_count
#         })
#     except Exception as e:
#         return JSONResponse(
#             content={"error": f"Failed to get feedback stats: {str(e)}"}, 
#             status_code=500
#         )


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from db_import import collection, embedding_model, feedback_manager
import logging
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Enable Logging
logging.basicConfig(level=logging.INFO)

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

# Pydantic models
class Query(BaseModel):
    text: str
    n_results: Optional[int] = 5

class FeedbackRequest(BaseModel):
    query: str
    question_id: str
    relevance_score: float

# Routes
@app.get("/")
async def read_root():
    return FileResponse("template/index.html")

@app.post("/query")
async def query_database(query: Query):
    try:
        logging.info(f"📥 Received query: {query.text}")

        # ✅ Generate query embedding
        query_embedding = embedding_model.encode([query.text])[0].tolist()
        logging.info(f"🔎 Generated Embedding: {query_embedding[:5]}...")  # Log first 5 values

        # ✅ Query ChromaDB for relevant answers
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            include=["metadatas", "distances"]
        )

        # ✅ Check if results exist
        if not results['ids'][0]:  
            logging.warning("⚠ No matching results found in ChromaDB!")
            return {
                "results": [{"id": None, "answer": "Sorry, I couldn't find an answer to your question."}]
            }

        # ✅ Process results
        processed_results = []
        top_cluster = None  # Store cluster of the top-ranked result

        for i in range(len(results['ids'][0])):
            result_data = {
                'id': results['ids'][0][i],
                'question': results['metadatas'][0][i]['question'],
                'answer': results['metadatas'][0][i]['answer'],
                'cluster': results['metadatas'][0][i]['cluster_kmeans'],
                'similarity': round(1 - float(results['distances'][0][i]), 4),
            }
            processed_results.append(result_data)

            # Capture cluster of the first result
            if i == 0:
                top_cluster = result_data['cluster']

        # ✅ Fetch 3 similar questions from the same cluster
        if top_cluster is not None:
            similar_questions_results = collection.get(
                where={"cluster_kmeans": top_cluster},  # Filter by cluster
                include=["metadatas"]
            )

            # Extract 3 similar questions (excluding the original query result)
            similar_questions = [
                {"question": meta['question']} 
                for meta in similar_questions_results['metadatas']
                if meta['question'] != processed_results[0]['question']
            ][:3]  # Limit to 3 suggestions

        else:
            similar_questions = []

        # ✅ Log response
        logging.info(f"📤 Returning Answer: {processed_results[0]['answer']} with {len(similar_questions)} similar questions.")

        return {
            'results': processed_results,
            'query': query.text,
            'similar_questions': similar_questions
        }

    except Exception as e:
        logging.error(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def add_feedback(feedback: FeedbackRequest):
    try:
        await feedback_manager.add_feedback(
            query=feedback.query,
            question_id=feedback.question_id,
            relevance_score=feedback.relevance_score
        )
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logging.error(f"❌ Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clusters")
async def get_clusters():
    try:
        results = collection.get()
        clusters = {}

        for metadata in results['metadatas']:
            cluster = metadata['cluster_kmeans']
            if cluster not in clusters:
                clusters[cluster] = 0
            clusters[cluster] += 1

        return {
            'clusters': [
                {'name': k, 'count': v}
                for k, v in clusters.items()
            ]
        }
    except Exception as e:
        logging.error(f"❌ Cluster analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import chromadb
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# import scipy.cluster.hierarchy as sch
# import matplotlib.pyplot as plt

# # ✅ Load CSV File
# csv_file = r"C:\Users\visha\Downloads\ValidatedQA (2).csv"
# df = pd.read_csv(csv_file)

# # ✅ Load Embedding Model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Define ChromaDB Embedding Function
# class ChromaEmbeddingFunction:
#     def __call__(self, input):  
#         return embedding_model.encode(input).tolist()

# # ✅ Initialize ChromaDB with HNSW Indexing
# chroma_client = chromadb.PersistentClient(path="./chroma_db")

# collection = chroma_client.get_or_create_collection(
#     name="qa_collection",
#     embedding_function=ChromaEmbeddingFunction(),
#     metadata={"hnsw_space": "cosine", "hnsw_ef_construction": 200, "hnsw_M": 16}
# )

# print("✅ HNSW indexing enabled with optimized parameters!")

# # ✅ Check if data already exists
# existing_data = collection.get()

# if existing_data["ids"]:
#     print("ChromaDB already contains embeddings. Skipping re-insertion.")
# else:
#     # ✅ Generate Embeddings
#     question_texts = df["question"].tolist()
#     question_embeddings = embedding_model.encode(question_texts).tolist()
#     embeddings_array = np.array(question_embeddings)

#     # ✅ Step 1: Find Optimal K for K-Means
#     def find_optimal_k(embeddings):
#         silhouette_scores = []
#         distortions = []
#         K_range = range(2, 15)  # Test k from 2 to 15

#         for k in K_range:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             labels = kmeans.fit_predict(embeddings)
#             silhouette_scores.append(silhouette_score(embeddings, labels))
#             distortions.append(kmeans.inertia_)  # Add this line

#             # Plot Elbow & Silhouette Score
#         plt.figure(figsize=(10, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(K_range, distortions, 'bo-')
#         plt.xlabel('k')
#         plt.ylabel('Distortion (Inertia)')
#         plt.title('Elbow Method for Optimal k')

#         plt.subplot(1, 2, 2)
#         plt.plot(K_range, silhouette_scores, 'go-')
#         plt.xlabel('k')
#         plt.ylabel('Silhouette Score')
#         plt.title('Silhouette Score for Optimal k')

#         plt.show()
#         best_k = K_range[np.argmax(silhouette_scores)]
#         return best_k

#     optimal_k = find_optimal_k(embeddings_array)
#     print(f"✅ Optimal number of clusters: {optimal_k}")

#     # ✅ Step 2: K-Means Clustering
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#     df["cluster_kmeans"] = [f"cluster_{c}" for c in kmeans.fit_predict(embeddings_array)]

#     # ✅ Step 3: Hierarchical Clustering
#     linkage_matrix = sch.linkage(embeddings_array, method='ward')  # Ward minimizes variance

#     # ✅ Determine number of clusters dynamically
#     hierarchical_clusters = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
#     df["cluster_hierarchical"] = [f"hcluster_{c}" for c in hierarchical_clusters.fit_predict(embeddings_array)]

#     # ✅ Step 4: Store Relationships in ChromaDB
#     df["related_questions"] = [[] for _ in range(len(df))]  # Placeholder for related Qs

#     for i, question in enumerate(df["question"]):
#         # Find 3 closest hierarchical neighbors
#         distances = np.linalg.norm(embeddings_array - embeddings_array[i], axis=1)
#         closest_indices = distances.argsort()[1:4]  # Get 3 closest (excluding itself)
#         df.at[i, "related_questions"] = [df.iloc[idx]["question"] for idx in closest_indices]

#     # ✅ Insert Data into ChromaDB with Relationships
#     collection.add(
#         ids=[f"q{i}" for i in range(len(df))],
#         embeddings=question_embeddings,
#         documents=question_texts,
#         metadatas=[
#             {
#                 "question": q,
#                 "answer": a,
#                 "cluster_kmeans": clus_k,
#                 "cluster_hierarchical": clus_h,
#                 "related_questions": "|||".join(related_qs),  # Convert list to string
#                 "relevance_score": 1
#             }
#             for q, a, clus_k, clus_h, related_qs in zip(
#                 df["question"], 
#                 df["answer"], 
#                 df["cluster_kmeans"], 
#                 df["cluster_hierarchical"], 
#                 df["related_questions"]
#             )
#         ]
#     )


#     print("✅ Database rebuilt successfully with Enhanced Clustering!")


# import chromadb
# import pandas as pd
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm

# # ✅ Check GPU availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # ✅ Load CSV
# csv_file = r"C:\Users\visha\Downloads\ValidatedQA (2).csv"
# df = pd.read_csv(csv_file)

# # ✅ Initialize model on GPU
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model.to(device)

# class ChromaEmbeddingFunction:
#     def __call__(self, input):
#         return embedding_model.encode(input, device=device).tolist()

# # ✅ Initialize ChromaDB
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(
#     name="qa_collection",
#     embedding_function=ChromaEmbeddingFunction(),
#     metadata={
#         "hnsw_space": "cosine",
#         "hnsw_construction_ef": 200,
#         "hnsw_M": 16
#     }
# )

# # ✅ Function: Find Optimal K for K-Means (Using Silhouette Score)
# def find_optimal_k(embeddings):
#     silhouette_scores = []
#     K_range = range(2, 15)

#     for k in K_range:
#         try:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             labels = kmeans.fit_predict(embeddings)
#             silhouette_scores.append(silhouette_score(embeddings, labels))
#         except Exception as e:
#             print(f"⚠ Skipping k={k}: {e}")
    
#     if not silhouette_scores:
#         print("❌ No valid clustering found. Using default k=5.")
#         return 5
    
#     return K_range[np.argmax(silhouette_scores)]

# # ✅ K-Means Clustering on CPU (NumPy)
# def kmeans_clustering(embeddings, k):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     return kmeans.fit_predict(embeddings)

# # ✅ Find Related Questions Using NumPy Cosine Similarity
# def find_related_questions(embeddings, df, top_n=3):
#     similarities = cosine_similarity(embeddings)  # Compute cosine similarity matrix

#     related_questions = []
#     for i in tqdm(range(len(df)), desc="Finding related questions"):
#         closest_indices = np.argsort(-similarities[i])[1:top_n+1]  # Get top N closest questions (excluding itself)
#         related_qs = [df.iloc[idx]["question"] for idx in closest_indices]
#         related_questions.append(related_qs)

#     return related_questions

# # ✅ If Data Already Exists, Skip Re-insertion
# if collection.count() > 0:
#     print("ChromaDB already contains embeddings. Skipping re-insertion.")
# else:
#     # ✅ Generate Embeddings
#     question_texts = df["question"].tolist()
#     question_embeddings = embedding_model.encode(
#         question_texts,
#         show_progress_bar=True,
#         device=device,
#         batch_size=32
#     )
#     embeddings_array = np.array(question_embeddings)

#     # ✅ Find Optimal K and Cluster
#     optimal_k = find_optimal_k(embeddings_array)
#     print(f"✅ Optimal number of clusters: {optimal_k}")

#     df["cluster_kmeans"] = [f"cluster_{c}" for c in kmeans_clustering(embeddings_array, optimal_k)]
#     df["related_questions"] = find_related_questions(embeddings_array, df)

#     # ✅ Batch Insert into ChromaDB
#     collection.add(
#         ids=[f"q{i}" for i in range(len(df))],
#         embeddings=question_embeddings,  # ✅ No need for `.tolist()`
#         documents=question_texts,
#         metadatas=[{
#             "question": q,
#             "answer": a,
#             "cluster_kmeans": clus_k,
#             "related_questions": "|||".join(related_qs),  # ✅ Convert list to string
#             "relevance_score": 1
#         } for q, a, clus_k, related_qs in zip(
#             df["question"],
#             df["answer"],
#             df["cluster_kmeans"],
#             df["related_questions"]
#         )]
#     )

#     print("✅ Database rebuilt successfully with NumPy & Scikit-Learn!")

# Existing imports
# import chromadb
# import pandas as pd
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import logging
# from time import time
# from datetime import datetime
# from typing import List, Dict
# import json
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Check GPU availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

# # Add these constants here
# FEEDBACK_FILE = "feedback_history.json"
# FEEDBACK_THRESHOLD = 0.7  # Threshold for positive feedback
# REWEIGHT_INTERVAL = 100   # Number of feedbacks before triggering reweighting

# # Your existing CONFIG
# CONFIG = {
#     'batch_size': 32,
#     'pca_components': 50,
#     'min_clusters': 2,
#     'max_clusters': 15,
#     'related_questions_count': 3,
#     'chroma_batch_size': 500
# }

# # Rest of your code (FeedbackManager class, etc.)...

# def find_optimal_k(embeddings: np.ndarray, min_k: int = 2, max_k: int = 15) -> int:
#     """
#     Find optimal number of clusters using silhouette score.
    
#     Args:
#         embeddings: numpy array of embeddings
#         min_k: minimum number of clusters to try
#         max_k: maximum number of clusters to try
    
#     Returns:
#         optimal number of clusters
#     """
#     best_k = min_k
#     best_score = -1
    
#     # Use tqdm for progress bar
#     for k in tqdm(range(min_k, max_k + 1), desc="Finding optimal k"):
#         try:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             cluster_labels = kmeans.fit_predict(embeddings)
            
#             # Calculate silhouette score
#             score = silhouette_score(embeddings, cluster_labels)
#             logger.info(f"k={k}, silhouette score={score:.4f}")
            
#             if score > best_score:
#                 best_score = score
#                 best_k = k
                
#         except Exception as e:
#             logger.warning(f"Error calculating score for k={k}: {str(e)}")
#             continue
    
#     logger.info(f"✅ Optimal number of clusters: {best_k}")
#     return best_k

# class FeedbackManager:
#     def __init__(self, collection, embedding_model):
#         self.collection = collection
#         self.embedding_model = embedding_model
#         self.feedback_count = 0
#         self.feedback_history = self._load_feedback_history()

#     def _load_feedback_history(self) -> Dict:
#         """Load feedback history from file or create new one"""
#         if os.path.exists(FEEDBACK_FILE):
#             with open(FEEDBACK_FILE, 'r') as f:
#                 return json.load(f)
#         return {"feedbacks": [], "last_reweight": None}

#     def _save_feedback_history(self):
#         """Save feedback history to file"""
#         with open(FEEDBACK_FILE, 'w') as f:
#             json.dump(self.feedback_history, f)

#     async def add_feedback(self, query: str, question_id: str, relevance_score: float):
#         """Add new feedback and trigger reweighting if necessary"""
#         timestamp = datetime.now().isoformat()
        
#         # Add feedback to history
#         self.feedback_history["feedbacks"].append({
#             "query": query,
#             "question_id": question_id,
#             "relevance_score": relevance_score,
#             "timestamp": timestamp
#         })
#         self.feedback_count += 1
#         self._save_feedback_history()

#         # Check if reweighting is needed
#         if self.feedback_count >= REWEIGHT_INTERVAL:
#             await self.reweight_embeddings()
#             self.feedback_count = 0

#     async def reweight_embeddings(self):
#         """Reweight embeddings based on feedback history"""
#         logger.info("Starting embedding reweighting...")
        
#         try:
#             # Get all documents and their metadata
#             results = self.collection.get()
#             ids = results['ids']
#             embeddings = np.array(results['embeddings'])
#             metadatas = results['metadatas']

#             # Calculate weights based on feedback
#             weights = self._calculate_weights(ids)
            
#             # Apply weights to embeddings
#             weighted_embeddings = embeddings * weights[:, np.newaxis]

#             # Recalculate clusters if needed
#             if len(weighted_embeddings) > 0:
#                 new_clusters = self._recalculate_clusters(weighted_embeddings)
                
#                 # Update metadata with new clusters
#                 for i, metadata in enumerate(metadatas):
#                     metadata['cluster_kmeans'] = f"cluster_{new_clusters[i]}"
#                     metadata['last_reweight'] = datetime.now().isoformat()

#                 # Update collection with new embeddings and metadata
#                 self.collection.update(
#                     ids=ids,
#                     embeddings=weighted_embeddings.tolist(),
#                     metadatas=metadatas
#                 )

#             logger.info("Embedding reweighting completed")
#             self.feedback_history["last_reweight"] = datetime.now().isoformat()
#             self._save_feedback_history()

#         except Exception as e:
#             logger.error(f"Error during reweighting: {str(e)}")
#             raise

#     def _calculate_weights(self, ids: List[str]) -> np.ndarray:
#         """Calculate weights for each document based on feedback history"""
#         weights = np.ones(len(ids))
        
#         for feedback in self.feedback_history["feedbacks"]:
#             try:
#                 idx = ids.index(feedback["question_id"])
                
#                 # Adjust weight based on feedback score
#                 if feedback["relevance_score"] > FEEDBACK_THRESHOLD:
#                     weights[idx] *= 1.1  # Increase weight for positive feedback
#                 else:
#                     weights[idx] *= 0.9  # Decrease weight for negative feedback
#             except ValueError:
#                 logger.warning(f"Question ID {feedback['question_id']} not found in collection")
#                 continue
                
#         # Normalize weights
#         return weights / np.max(weights)  # Changed from mean to max for better scaling

#     def _recalculate_clusters(self, embeddings: np.ndarray) -> np.ndarray:
#         """Recalculate clusters based on weighted embeddings"""
#         try:
#             # Use CONFIG values for min and max clusters
#             optimal_k = find_optimal_k(
#                 embeddings,
#                 min_k=CONFIG['min_clusters'],
#                 max_k=CONFIG['max_clusters']
#             )
#             kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#             return kmeans.fit_predict(embeddings)
#         except Exception as e:
#             logger.error(f"Error in cluster recalculation: {str(e)}")
#             # Fallback to minimum number of clusters if error occurs
#             kmeans = KMeans(n_clusters=CONFIG['min_clusters'], random_state=42, n_init=10)
#             return kmeans.fit_predict(embeddings)

# # Initialize model and ChromaDB (existing code)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model.to(device)

# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(
#     name="qa_collection",
#     embedding_function=ChromaEmbeddingFunction(embedding_model),
#     metadata={"hnsw_space": "cosine", "hnsw_construction_ef": 200, "hnsw_M": 16}
# )

# # Initialize feedback manager
# feedback_manager = FeedbackManager(collection, embedding_model)

# # Rest of your existing code...


# # ✅ Configuration
# CONFIG = {
#     'batch_size': 32,
#     'pca_components': 50,
#     'min_clusters': 2,
#     'max_clusters': 15,
#     'related_questions_count': 3,
#     'chroma_batch_size': 500
# }

# class ChromaEmbeddingFunction:
#     def __init__(self, model):
#         self.model = model
    
#     def __call__(self, input):
#         return self.model.encode(input, device=device).tolist()

# def batch_add_to_chroma(collection, ids, embeddings, documents, metadatas, batch_size=500):
#     """Add data to ChromaDB in batches to prevent memory issues"""
#     for i in tqdm(range(0, len(ids), batch_size), desc="Adding to ChromaDB"):
#         end_idx = min(i + batch_size, len(ids))
#         collection.add(
#             ids=ids[i:end_idx],
#             embeddings=embeddings[i:end_idx],
#             documents=documents[i:end_idx],
#             metadatas=metadatas[i:end_idx]
#         )

# def reduce_dimensions(embeddings, n_components=50):
#     """Reduce dimensions using PCA"""
#     start_time = time()
#     pca = PCA(n_components=n_components)
#     reduced_embeddings = pca.fit_transform(embeddings)
#     logger.info(f"✅ PCA applied: Reduced from {embeddings.shape[1]}D to {n_components}D "
#                 f"(Time: {time()-start_time:.2f}s)")
#     return reduced_embeddings

# def find_optimal_k(embeddings):
#     """Find optimal number of clusters using silhouette score"""
#     silhouette_scores = []
#     K_range = range(CONFIG['min_clusters'], CONFIG['max_clusters'])

#     for k in tqdm(K_range, desc="Finding optimal k"):
#         try:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             labels = kmeans.fit_predict(embeddings)
#             score = silhouette_score(embeddings, labels)
#             silhouette_scores.append(score)
#             logger.info(f"k={k}, silhouette score={score:.4f}")
#         except Exception as e:
#             logger.warning(f"⚠ Skipping k={k}: {e}")
    
#     if not silhouette_scores:
#         logger.warning("❌ No valid clustering found. Using default k=5.")
#         return 5
    
#     return K_range[np.argmax(silhouette_scores)]

# def find_related_questions(embeddings, df, top_n=3):
#     """Find related questions using cosine similarity with batch processing"""
#     start_time = time()
#     batch_size = 1000
#     related_questions = []
    
#     for i in tqdm(range(0, len(df), batch_size), desc="Finding related questions"):
#         batch_end = min(i + batch_size, len(df))
#         batch_similarities = cosine_similarity(
#             embeddings[i:batch_end],
#             embeddings
#         )
        
#         for sim in batch_similarities:
#             closest_indices = np.argsort(-sim)[1:top_n+1]
#             related_qs = [df.iloc[idx]["question"] for idx in closest_indices]
#             related_questions.append(related_qs)
    
#     logger.info(f"Found related questions in {time()-start_time:.2f}s")
#     return related_questions

#         # ✅ Initialize model and ChromaDB
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model.to(device)
        
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(
#     name="qa_collection",
#     embedding_function=ChromaEmbeddingFunction(embedding_model),
#     metadata={"hnsw_space": "cosine", "hnsw_construction_ef": 200, "hnsw_M": 16}
# )

# def main():
#     try:
#         # ✅ Load CSV
#         csv_file = r"C:\Users\visha\Downloads\ValidatedQA (2).csv"
#         df = pd.read_csv(csv_file)
#         logger.info(f"Loaded {len(df)} questions from CSV")

#         if collection.count() > 0:
#             logger.info("ChromaDB already contains embeddings. Skipping re-insertion.")
#             return

#         # ✅ Generate Embeddings
#         logger.info("Generating embeddings...")
#         question_texts = df["question"].tolist()
#         question_embeddings = embedding_model.encode(
#             question_texts,
#             show_progress_bar=True,
#             device=device,
#             batch_size=CONFIG['batch_size']
#         )
#         embeddings_array = np.array(question_embeddings)

#         # ✅ Process Embeddings
#         embeddings_reduced = reduce_dimensions(embeddings_array, CONFIG['pca_components'])
#         optimal_k = find_optimal_k(embeddings_reduced)
#         logger.info(f"✅ Optimal number of clusters: {optimal_k}")

#         # ✅ Clustering and Related Questions
#         df["cluster_kmeans"] = [f"cluster_{c}" for c in KMeans(
#             n_clusters=optimal_k, 
#             random_state=42, 
#             n_init=10
#         ).fit_predict(embeddings_reduced)]
        
#         df["related_questions"] = find_related_questions(
#             embeddings_reduced, 
#             df, 
#             CONFIG['related_questions_count']
#         )

#         # ✅ Prepare metadata and add to ChromaDB
#         metadatas = [{
#             "question": q,
#             "answer": a,
#             "cluster_kmeans": clus_k,
#             "related_questions": "|||".join(related_qs),
#             "relevance_score": 1
#         } for q, a, clus_k, related_qs in zip(
#             df["question"],
#             df["answer"],
#             df["cluster_kmeans"],
#             df["related_questions"]
#         )]

#         batch_add_to_chroma(
#             collection,
#             ids=[f"q{i}" for i in range(len(df))],
#             embeddings=question_embeddings,
#             documents=question_texts,
#             metadatas=metadatas,
#             batch_size=CONFIG['chroma_batch_size']
#         )

#         logger.info("✅ Database rebuilt successfully!")

#     except Exception as e:
#         logger.error(f"❌ Error: {str(e)}", exc_info=True)
#         raise

# # Export these for use in other files
# __all__ = ['collection', 'embedding_model']

# if __name__ == "__main__":
#     main()



# 1. IMPORTS
import chromadb
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
from time import time
from datetime import datetime
from typing import List, Dict, Tuple
import json
import os
import csv
import sys

# 2. LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 3. DEVICE CONFIGURATION
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# 4. CONSTANTS
FEEDBACK_FILE = "feedback_history.json"
FEEDBACK_THRESHOLD = 0.7
REWEIGHT_INTERVAL = 100
CSV_FILE_PATH = r"C:\Users\visha\Downloads\DataBase.csv"  # Update this path

# 5. CONFIGURATION DICTIONARY
CONFIG = {
    'batch_size': 32,
    'pca_components': 50,
    'min_clusters': 2,
    'max_clusters': 15,
    'related_questions_count': 3,
    'chroma_batch_size': 500
}

# 6. EMBEDDING FUNCTION CLASS
class ChromaEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            if isinstance(input, str):
                input = [input]
            embeddings = self.model.encode(
                input,
                device=device,
                show_progress_bar=True,
                batch_size=CONFIG['batch_size']
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            raise

# 7. DATA IMPORT FUNCTIONS
def import_data_from_csv(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Import data from CSV and prepare it for ChromaDB."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows")
        
        # Ensure required columns exist
        required_columns = ['question', 'answer']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
        # Clean the data
        df['question'] = df['question'].fillna('').str.strip()
        df['answer'] = df['answer'].fillna('').str.strip()
        
        # Remove empty rows - Fixed the boolean operation
        mask = (df['question'].str.len() > 0) & (df['answer'].str.len() > 0)
        df = df[mask]
        
        # Combine question and answer for document text
        df['text'] = df['question'] + " " + df['answer']
        
        logger.info(f"Processed {len(df)} valid rows after cleaning")
        return df, df['text'].tolist()
    except Exception as e:
        logger.error(f"Error importing data from CSV: {str(e)}")
        raise

# Also, let's fix the logging setup to handle Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),  # Added encoding
        logging.StreamHandler()
    ]
)


def import_and_index_data(file_path: str, collection) -> None:
    """Import data from CSV and index it in ChromaDB."""
    try:
        # Check if collection already has data
        if collection.count() > 0:
            logger.info("Collection already contains data. Skipping import.")
            return

        # Import data
        df, documents = import_data_from_csv(file_path)
        
        logger.info("Generating embeddings...")
        embedding_function = ChromaEmbeddingFunction(embedding_model)
        original_embeddings = embedding_function(documents)
        
        logger.info("Adding data to ChromaDB...")
        batch_size = CONFIG['chroma_batch_size']
        for i in tqdm(range(0, len(df), batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, len(df))
            batch_df = df.iloc[i:end_idx]
            batch_embeddings = original_embeddings[i:end_idx]
            
            collection.add(
                ids=[f"q{j}" for j in range(i, end_idx)],
                embeddings=batch_embeddings,
                documents=batch_df['text'].tolist(),
                metadatas=[{
                    'question': row['question'],
                    'answer': row['answer'],
                    'cluster_kmeans': 'pending'
                } for _, row in batch_df.iterrows()]
            )
        
        # Initial clustering
        logger.info("Performing initial clustering...")
        
        try:
            # Get all embeddings at once
            all_embeddings = np.array(original_embeddings, dtype=np.float32)
            logger.info(f"Original embeddings shape: {all_embeddings.shape}")
            
            # Verify embeddings array
            if all_embeddings.size == 0:
                raise ValueError("Empty embeddings array")
            
            # Perform clustering
            optimal_k = find_optimal_k(all_embeddings)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(all_embeddings)
            
            # Update cluster information
            logger.info("Updating cluster information...")
            for i, label in enumerate(cluster_labels):
                collection.update(
                    ids=[f"q{i}"],
                    metadatas=[{
                        'cluster_kmeans': f"cluster_{label}",
                        'question': df.iloc[i]['question'],
                        'answer': df.iloc[i]['answer']
                    }]
                )
            
            logger.info(f"Successfully imported and indexed {len(df)} items with {optimal_k} clusters")
            
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            logger.info("Proceeding without clustering...")
            
    except Exception as e:
        logger.error(f"Error in import and indexing process: {str(e)}")
        raise

# 8. FEEDBACK MANAGER CLASS
class FeedbackManager:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.feedback_count = 0
        self.feedback_history = self._load_feedback_history()

    def _load_feedback_history(self) -> Dict:
        try:
            if os.path.exists(FEEDBACK_FILE):
                with open(FEEDBACK_FILE, 'r') as f:
                    return json.load(f)
            return {"feedbacks": [], "last_reweight": None}
        except Exception as e:
            logger.error(f"Error loading feedback history: {str(e)}")
            return {"feedbacks": [], "last_reweight": None}

    def _save_feedback_history(self):
        try:
            with open(FEEDBACK_FILE, 'w') as f:
                json.dump(self.feedback_history, f)
        except Exception as e:
            logger.error(f"Error saving feedback history: {str(e)}")

    async def add_feedback(self, query: str, question_id: str, relevance_score: float):
        try:
            timestamp = datetime.now().isoformat()
            
            self.feedback_history["feedbacks"].append({
                "query": query,
                "question_id": question_id,
                "relevance_score": relevance_score,
                "timestamp": timestamp
            })
            self.feedback_count += 1
            self._save_feedback_history()

            if self.feedback_count >= REWEIGHT_INTERVAL:
                await self.reweight_embeddings()
                self.feedback_count = 0
                
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            raise

    async def reweight_embeddings(self):
        try:
            logger.info("Starting embedding reweighting...")
            
            results = self.collection.get()
            ids = results['ids']
            embeddings = np.array(results['embeddings'])
            metadatas = results['metadatas']

            weights = self._calculate_weights(ids)
            weighted_embeddings = embeddings * weights[:, np.newaxis]

            if len(weighted_embeddings) > 0:
                new_clusters = self._recalculate_clusters(weighted_embeddings)
                
                for i, metadata in enumerate(metadatas):
                    metadata['cluster_kmeans'] = f"cluster_{new_clusters[i]}"
                    metadata['last_reweight'] = datetime.now().isoformat()

                self.collection.update(
                    ids=ids,
                    embeddings=weighted_embeddings.tolist(),
                    metadatas=metadatas
                )

            self.feedback_history["last_reweight"] = datetime.now().isoformat()
            self._save_feedback_history()
            logger.info("Embedding reweighting completed")
            
        except Exception as e:
            logger.error(f"Error during reweighting: {str(e)}")
            raise

    def _calculate_weights(self, ids: List[str]) -> np.ndarray:
        try:
            weights = np.ones(len(ids))
            
            for feedback in self.feedback_history["feedbacks"]:
                try:
                    idx = ids.index(feedback["question_id"])
                    if feedback["relevance_score"] > FEEDBACK_THRESHOLD:
                        weights[idx] *= 1.1
                    else:
                        weights[idx] *= 0.9
                except ValueError:
                    continue
                    
            return weights / np.max(weights)
            
        except Exception as e:
            logger.error(f"Error calculating weights: {str(e)}")
            return np.ones(len(ids))

    def _recalculate_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            optimal_k = find_optimal_k(embeddings)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            kmeans = KMeans(n_clusters=CONFIG['min_clusters'], random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)

# 9. UTILITY FUNCTIONS
def find_optimal_k(embeddings: np.ndarray) -> int:
    """Find optimal number of clusters using silhouette score."""
    try:
        # Ensure embeddings are the correct type and shape
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Finding optimal k for embeddings shape: {embeddings.shape}")
        
        best_k = CONFIG['min_clusters']
        best_score = -1
        
        for k in tqdm(range(CONFIG['min_clusters'], CONFIG['max_clusters'] + 1), 
                     desc="Finding optimal k"):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, cluster_labels)
                logger.info(f"k={k}, silhouette score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"Error for k={k}: {str(e)}")
                continue
                
        logger.info(f"Optimal clusters found: {best_k}")
        return best_k
        
    except Exception as e:
        logger.error(f"Error finding optimal k: {str(e)}")
        return CONFIG['min_clusters']

# Also, let's update the logging setup to handle Unicode properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # This will reset any existing handlers
)

# Update the logging setup to handle Unicode characters properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Add explicit stdout handler
    ]
)

# 10. INITIALIZATION
try:
    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model.to(device)
    logger.info("✅ Embedding model initialized")

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="qa_collection",
        embedding_function=ChromaEmbeddingFunction(embedding_model),
        metadata={
            "hnsw_space": "cosine",
            "hnsw_construction_ef": 200,
            "hnsw_M": 16
        }
    )
    logger.info("✅ ChromaDB collection initialized")

    # Initialize feedback manager
    feedback_manager = FeedbackManager(collection, embedding_model)
    logger.info("✅ Feedback manager initialized")

except Exception as e:
    logger.error(f"❌ Initialization error: {str(e)}")
    raise

# 11. MAIN EXECUTION
if __name__ == "__main__":
    try:
        # Import and index the data
        import_and_index_data(CSV_FILE_PATH, collection)
        logger.info("✅ Data import and indexing completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during data import and indexing: {str(e)}")

# 12. EXPORTS
__all__ = [
    'collection',
    'embedding_model',
    'feedback_manager',
    'import_and_index_data'
]

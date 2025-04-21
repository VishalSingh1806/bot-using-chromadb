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
CSV_FILE_PATH = r"~/bot-using-chroma/FAQ Database.csv"  # Update this path

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
            # ✅ Normalize embeddings
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return normalized_embeddings.tolist()
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
                ids=[f"q{j+i}" for j in range(len(batch_df))],  # ✅ Ensure unique IDs
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
                logger.info(f"✅ Updated q{i} → Cluster: cluster_{label}")

            
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
    """Find optimal number of clusters using PCA + KMeans + Silhouette Score"""
    try:
        # ✅ Reduce dimensions with PCA to avoid overfitting
        pca = PCA(n_components=CONFIG['pca_components'])
        reduced_embeddings = pca.fit_transform(embeddings)

        best_k = CONFIG['min_clusters']
        best_score = -1

        for k in tqdm(range(CONFIG['min_clusters'], CONFIG['max_clusters'] + 1), 
                     desc="Finding optimal k"):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(reduced_embeddings)
                score = silhouette_score(reduced_embeddings, cluster_labels)

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
            "hnsw_space": "cosine",  # ✅ Force cosine similarity
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

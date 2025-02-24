import chromadb
#This script is to clear the db if needed
# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# # Delete existing collection
chroma_client.delete_collection("qa_collection")

print("Existing ChromaDB collection deleted successfully.")

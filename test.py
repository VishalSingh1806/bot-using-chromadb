# import chromadb

# # Initialize ChromaDB Client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_collection(name="qa_collection")

# # Fetch schema & sample data (Ensure embeddings are included)
# sample_data = collection.get(
#     ids=["q0"],  # Get specific document
#     include=["embeddings", "documents", "metadatas"]  # ✅ Explicitly request embeddings
# )

# print("Stored Schema & Sample Data:")
# print(sample_data)


import chromadb

# ✅ Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Load the collection
collection = chroma_client.get_collection("qa_collection")

# ✅ Count total entries
print(f"Total records in ChromaDB: {collection.count()}")

# ✅ Retrieve a few sample entries
sample_data = collection.get(limit=10)  # Get first 5 records

# ✅ Display metadata of first 5 records
for i, metadata in enumerate(sample_data["metadatas"]):
    print(f"\n🔹 Record {i+1}:")
    print(f"Question: {metadata['question']}")
    print(f"Answer: {metadata['answer']}")
    print(f"Cluster: {metadata['cluster_kmeans']}")
    # print(f"Related Questions: {metadata['related_questions']}")


# import chromadb
# from sentence_transformers import SentenceTransformer

# # ✅ Load Embedding Model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Initialize ChromaDB
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_collection("qa_collection")

# def query_chromadb(user_query):
#     """Search for the most relevant answer"""
#     query_embedding = embedding_model.encode([user_query]).tolist()

#     # ✅ Retrieve the best match
#     results = collection.query(query_embeddings=query_embedding, n_results=1)

#     # ✅ Display response
#     if results["metadatas"]:
#         print("\n🔹 Query:", user_query)
#         print("✅ Best Answer:", results["metadatas"][0][0]["answer"])
#         print("🔗 Related Questions:", results["metadatas"][0][0]["related_questions"])
#     else:
#         print("❌ No matching answer found.")

# # ✅ Test Queries
# query_chromadb("What are the latest updates to plastic waste management?")
# query_chromadb("How do I apply for EPR certification?")
# query_chromadb("What are the penalties for non-compliance with plastic rules?")

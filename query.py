from db_import import embedding_model,collection

def query_chromadb(user_query, n_suggestions=3):
    # Convert user query to an embedding
    query_embedding = embedding_model.encode([user_query]).tolist()

    # Get the most relevant answer
    result = collection.query(query_embeddings=query_embedding, n_results=1)

    if not result["ids"]:
        return "No relevant answer found.", []

    best_answer = result["metadatas"][0][0]["answer"]
    cluster = result["metadatas"][0][0]["cluster"]

    # Retrieve top similar questions from the same category, sorted by relevance score
    suggestions = collection.query(
        query_embeddings=query_embedding,
        n_results=n_suggestions + 1,
        where={"cluster": cluster},  # Filter by auto-cluster
    )

    # Extract suggested questions
    suggested_questions = [meta["question"] for meta in suggestions["metadatas"][0] if meta["question"] != user_query]

    return best_answer, suggested_questions

# Example Query
user_input = input("What do you want to know? ")
answer, suggestions = query_chromadb(user_input)

print(f"\nBest Answer: {answer}")
print("\nSuggested Questions:")
for i, q in enumerate(suggestions, start=1):
    print(f"{i}. {q}")

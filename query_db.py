from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Connect to your existing database
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# --- QUERY TYPE 1: Similarity Search ---
# This converts your question into an embedding and finds the closest matching reviews.
print("--- Running Similarity Search ---")
query = "Are there any complaints about bad service or long wait times?"

# k=2 means "give me the top 2 closest matches"
results = vector_store.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\nMatch #{i+1}:")
    print(f"Review: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")


print("\n" + "="*50 + "\n")


# --- QUERY TYPE 2: Similarity Search with Scores ---
# This shows you HOW close the match is (lower distance = closer match).
print("--- Running Search with Scores ---")
query_text = "delicious pizza and Italian food"
results_with_scores = vector_store.similarity_search_with_score(query_text, k=1)

for doc, score in results_with_scores:
    print(f"Review: {doc.page_content}")
    print(f"Distance Score: {score}")  


print("\n" + "="*50 + "\n")


# --- QUERY TYPE 3: Filtering by Metadata ---
# You can filter results by specific columns from your original CSV (like rating).
print("--- Running Filtered Search (Only 5-star reviews) ---")
filtered_results = vector_store.similarity_search(
    "What do people say about the atmosphere?",
    k=2,
    filter={"rating": 5}  # Looks for exact match in your metadata dictionary
)

for doc in filtered_results:
    print(f"Review: {doc.page_content}")
    print(f"Rating: {doc.metadata.get('rating')}")
    
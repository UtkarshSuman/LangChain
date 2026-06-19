from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "A. P. J. Abdul Kalam was a renowned scientist...",
    "Ratan Tata is a respected business leader...",
    "Mahendra Singh Dhoni is one of India's greatest captains...",
    "Lionel Messi is widely regarded as one of football's greatest players..."
]

query = "tell me about ratan tata"


doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

index, score = max(
    enumerate(scores),
    key=lambda x: x[1]
)

print("Query:", query)
print("Best Match:")
print(documents[index])
print("Similarity Score:", score)
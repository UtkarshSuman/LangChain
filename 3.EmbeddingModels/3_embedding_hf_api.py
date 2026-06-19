from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token="HUGGINGFACEHUB_API_TOKEN"
)

vector = embeddings.embed_query(
    "What is LangChain?"
)

print(len(vector))
print(vector[:5])
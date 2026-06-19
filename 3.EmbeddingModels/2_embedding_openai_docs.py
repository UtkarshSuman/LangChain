from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
  "Delhi is capital of India",
  "Jaipur is capital of Rajasthan",
  "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))

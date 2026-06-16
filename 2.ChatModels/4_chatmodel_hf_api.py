from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #endpoint wala used for api mode
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="meta-llama/Llama-3.1-8B-Instruct",
  task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
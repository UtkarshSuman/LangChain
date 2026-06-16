# On running it will localy download the files on our system and will run on that 
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline #pipeline for local development

llm = HuggingFacePipeline.from_model_id(
  model_id="meta-llama/Llama-3.1-8B-Instruct",
  task="text-generation",
  pipeline_kwargs=dict(
    temperature=0.5,
    max_new_tokens=100
  )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")
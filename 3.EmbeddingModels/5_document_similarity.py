from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
 " A. P. J. Abdul Kalam was a renowned scientist and former President of India who inspired generations with his vision for education, innovation, and national development.",
"Ratan Tata is a respected business leader known for transforming the Tata Group into a global enterprise while championing philanthropy and ethical leadership.",
"Mahendra Singh Dhoni is one of India's greatest cricket captains, admired for his calm demeanor, strategic thinking, and remarkable finishing abilities.",
"Lionel Messi is widely regarded as one of football's greatest players, celebrated for his exceptional dribbling, vision, goal-scoring ability, and numerous records."
]

query = 'tell me about ratan tata'

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]


# scores might look like:
# [0.71, 0.95, 0.60, 0.45]

# Pair document index with score
# Example:
# [(0,0.71), (1,0.95), (2,0.60), (3,0.45)]
# lambda x:x[1] lets sort using 0.71, 0.91 values
# Python sorts tuples by the first element by default
# Sort by similarity score and take highest one
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]


print(query)
print(documents[index])
print("similarity score is:", score)

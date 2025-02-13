from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = "E:\\AIML\\N_langchain\\Langchain_models\\embeddings"


embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))
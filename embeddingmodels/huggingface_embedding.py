from langchain_huggingface import HuggingFaceEmbeddings

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "you are going to learn Gen AI",
    "Gen AI and its applications are fascinating",
    "The future of AI is promising and full of potential"
]

vectors = huggingface_embeddings.embed_documents(texts)

print(vectors)
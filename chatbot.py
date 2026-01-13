# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Load embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local("vector_store", embeddings)

# # Load free HuggingFace LLM
# llm = HuggingFaceHub(repo_id="google/flan-t5-base")

# # Create Retrieval QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=db.as_retriever()
# )

# def ask_bot(query):
#     result = qa_chain.run(query)
#     return result

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and chunks
index = faiss.read_index("vector_store/index.faiss")

with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load lightweight LLM
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_bot(question):
    # Convert question to embedding
    q_embedding = embed_model.encode([question])

    # Search nearest chunk
    distances, indices = index.search(q_embedding, k=1)
    best_chunk = chunks[indices[0][0]]

    # Ask LLM using retrieved context
    prompt = f"""
    Answer the question based on the context below.

    Context:
    {best_chunk}

    Question:
    {question}
    """

    result = generator(prompt, max_length=200)
    return result[0]['generated_text']


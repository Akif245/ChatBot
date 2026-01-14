# from pypdf import PdfReader
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import os

# # Load PDF
# reader = PdfReader("data/college_info_dummy.pdf")
# text = ""
# for page in reader.pages:
#     text += page.extract_text()

# # Split text into chunks
# chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Create embeddings
# embeddings = model.encode(chunks)

# # Store FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# os.makedirs("vector_store", exist_ok=True)
# faiss.write_index(index, "vector_store/index.faiss")

# # Save chunks
# with open("vector_store/chunks.pkl", "wb") as f:
#     pickle.dump(chunks, f)

# print(" Embeddings stored successfully!")


from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, pickle, os

reader = PdfReader("data/college_info_dummy.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

chunks = [text[i:i+500] for i in range(0, len(text), 500)]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Embeddings stored successfully!")

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Load data
loader = TextLoader("tnea_data.txt")
documents = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Convert to embeddings using a free, local model (no API key needed!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)

# Save database
db.save_local("faiss_index")

print("✅ Database created!")
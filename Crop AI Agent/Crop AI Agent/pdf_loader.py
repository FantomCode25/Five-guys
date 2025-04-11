import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # New recommended import
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # Fallback

def build_vector_index():
    # 1. PDF Loading
    pdf_files = [
        "Crop-Physiology.pdf",
        "Introduction-to-Soil-Science.pdf",
        "Introductory-Agriculture.pdf",
        "Soil-Chemistry-Soil-Fertility-Nutrient-Management.pdf",
        "100-agricultureQuestions.pdf"
    ]
    
    # Verify files exist with absolute paths
    missing_files = [f for f in pdf_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing files: {missing_files}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Files present: {os.listdir()}"
        )

    # 2. Process PDFs
    all_pages = []
    for file in pdf_files:
        try:
            print(f"Processing {file}...")
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            all_pages.extend(pages)
            print(f"Successfully processed {file} ({len(pages)} pages)")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if not all_pages:
        raise ValueError("No documents were loaded from any PDF files")

    # 3. Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Explicit separators for better splitting
    )
    docs = text_splitter.split_documents(all_pages)
    print(f"Split into {len(docs)} document chunks")

    # 4. Create Embeddings
    model_kwargs = {'device': 'cpu'}  # Force CPU usage
    encode_kwargs = {'normalize_embeddings': True}  # Better for similarity search
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 5. Build and Save Index
    print("Building FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    
    # Create directory if it doesn't exist
    os.makedirs("cropmind_db", exist_ok=True)
    db.save_local("cropmind_db")
    print(f"Index built successfully! Saved to cropmind_db with {db.index.ntotal} vectors")
    
    return db  # Return the database object for use in other modules

# At the bottom of pdf_loader.py (remove the if __name__ block)
db = build_vector_index()  # This will run whenever the module is imported # Now the db object is available after running
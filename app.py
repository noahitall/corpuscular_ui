import os
import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Initialize document storage
UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
DOCUMENTS_METADATA = 'documents_metadata.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize vector store
VECTOR_STORE_PATH = "vectorstore"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434"

def load_documents_metadata():
    if os.path.exists(DOCUMENTS_METADATA):
        with open(DOCUMENTS_METADATA, 'r') as f:
            return json.load(f)
    return []

def save_documents_metadata(metadata):
    with open(DOCUMENTS_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_loader_for_file(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return PyPDFLoader(file_path)
    elif ext == 'docx':
        return Docx2txtLoader(file_path)
    else:
        return TextLoader(file_path)

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags")
        if response.ok:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []

@app.route('/')
def serve_static():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process and index the document
        loader = get_loader_for_file(filepath)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create or update vector store using local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        vectorstore.persist()
        
        # Update documents metadata
        metadata = load_documents_metadata()
        metadata.append({
            'filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'size': os.path.getsize(filepath),
            'chunks': len(texts)
        })
        save_documents_metadata(metadata)
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'metadata': metadata[-1]
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/documents', methods=['GET'])
def list_documents():
    metadata = load_documents_metadata()
    return jsonify(metadata)

@app.route('/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    # Load current metadata
    metadata = load_documents_metadata()
    
    # Find and remove the document metadata
    metadata = [m for m in metadata if m['filename'] != filename]
    save_documents_metadata(metadata)
    
    # Remove the file if it exists
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Note: We don't remove from vector store as it would require reindexing
    # In a production system, you might want to rebuild the vector store
    
    return jsonify({'message': 'Document deleted successfully'})

@app.route('/models', methods=['GET'])
def list_models():
    models = get_available_models()
    return jsonify(models)

@app.route('/query', methods=['POST'])
def query_documents():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    model_name = data.get('model', 'mistral')  # Default to mistral if not specified
    
    try:
        # Load vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        
        # Create QA chain with Ollama
        llm = Ollama(model=model_name)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": question})
        
        return jsonify({
            'answer': result['result'],
            'sources': [doc.page_content for doc in result['source_documents']]
        })
    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

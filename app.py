import os
import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template
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

app = Flask(__name__, static_folder='.', static_url_path='')

# Initialize document storage
UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'html'}
DOCUMENTS_METADATA = 'documents_metadata.json'

# Initialize vector store
VECTOR_STORE_PATH = "vectorstore"

def ensure_directory(path):
    """Ensure a directory exists with proper permissions."""
    try:
        os.makedirs(path, mode=0o755, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {str(e)}")
        return False

# Initialize directories with proper permissions
ensure_directory(UPLOAD_FOLDER)
ensure_directory(VECTOR_STORE_PATH)

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
    elif ext == 'html':
        # For HTML files, we'll create a custom loader that extracts text content
        class HTMLLoader:
            def __init__(self, file_path):
                self.file_path = file_path
            
            def load(self):
                from bs4 import BeautifulSoup
                from langchain.docstore.document import Document
                
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    # Get text content
                    text = soup.get_text(separator='\n', strip=True)
                    # Create metadata
                    metadata = {"source": self.file_path}
                    return [Document(page_content=text, metadata=metadata)]
        
        return HTMLLoader(file_path)
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
def index():
    return app.send_static_file('index.html')

def process_file(filepath, filename):
    """Process a single file and add it to the vector store."""
    try:
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
        
        # Return metadata for the processed file
        return {
            'filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'size': os.path.getsize(filepath),
            'chunks': len(texts),
            'status': 'success'
        }
    except PermissionError as e:
        return {
            'filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'error': f"Permission denied: {str(e)}",
            'status': 'error'
        }
    except Exception as e:
        return {
            'filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'error': str(e),
            'status': 'error'
        }

def process_directory(directory_path, base_path=None):
    """Recursively process all files in a directory."""
    if base_path is None:
        base_path = directory_path
        
    results = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if allowed_file(file):
                # Calculate relative path for storage
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_path)
                target_path = os.path.join(UPLOAD_FOLDER, rel_path)
                
                # Create necessary subdirectories
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy file to upload folder and process it
                import shutil
                shutil.copy2(abs_path, target_path)
                result = process_file(target_path, rel_path)
                results.append(result)
    
    return results

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
        
        # Process the file using our new helper function
        result = process_file(filepath, filename)
        
        # Update documents metadata if successful
        if result['status'] == 'success':
            metadata = load_documents_metadata()
            metadata.append(result)
            save_documents_metadata(metadata)
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'metadata': result
            })
        else:
            return jsonify({
                'error': 'Failed to process file',
                'details': result
            }), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/upload_directory', methods=['POST'])
def upload_directory():
    if 'directory' not in request.files:
        return jsonify({'error': 'No directory uploaded'}), 400
    
    files = request.files.getlist('directory')
    if not files:
        return jsonify({'error': 'No files in directory'}), 400
    
    # Create a temporary directory for processing
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all files to temporary directory maintaining structure
            for file in files:
                if file.filename:
                    try:
                        # Create the full path for the file
                        temp_path = os.path.join(temp_dir, file.filename)
                        os.makedirs(os.path.dirname(temp_path), mode=0o755, exist_ok=True)
                        file.save(temp_path)
                    except PermissionError as e:
                        return jsonify({
                            'error': f'Permission denied while saving file {file.filename}: {str(e)}'
                        }), 403
                    except Exception as e:
                        return jsonify({
                            'error': f'Error saving file {file.filename}: {str(e)}'
                        }), 500
            
            # Process the entire directory
            results = process_directory(temp_dir)
            
            # Update documents metadata with successful uploads
            metadata = load_documents_metadata()
            successful_results = [r for r in results if r['status'] == 'success']
            metadata.extend(successful_results)
            
            try:
                save_documents_metadata(metadata)
            except PermissionError as e:
                return jsonify({
                    'error': f'Permission denied while saving metadata: {str(e)}'
                }), 403
            
            return jsonify({
                'message': 'Directory processed',
                'results': results,
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results)
            })
    except PermissionError as e:
        return jsonify({
            'error': f'Permission denied while creating temporary directory: {str(e)}'
        }), 403
    except Exception as e:
        return jsonify({
            'error': f'Error processing directory: {str(e)}'
        }), 500

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

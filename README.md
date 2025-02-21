# Document Q&A System

This is a web-based application that allows users to upload documents and ask questions about their content. The system uses Ollama's language models to provide accurate answers based on the document content, running completely locally without requiring any API keys.

## Features

- Upload PDF, DOCX, and TXT files
- Process and index documents for efficient searching
- Ask questions about document content
- Get AI-powered answers with source references
- Simple and intuitive web interface
- Runs completely locally - no API keys needed

## Prerequisites

1. Install Ollama:
   ```bash
   # macOS
   curl https://ollama.ai/install.sh | sh

   # Linux
   curl https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download
   ```

2. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

## Setup

1. Clone this repository
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Upload documents using the drag-and-drop interface
4. Ask questions about your documents in the question input field
5. View AI-generated answers and their source references

## Supported File Types

- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)

## How it Works

1. Documents are uploaded and processed into chunks
2. Document chunks are embedded using HuggingFace's all-MiniLM-L6-v2 model
3. Embeddings are stored in a Chroma vector database
4. When a question is asked, the system:
   - Finds the most relevant document chunks
   - Uses Ollama's Mistral model to generate an answer
   - Returns the answer along with source references

## Technical Details

- Backend: Flask (Python)
- Vector Database: ChromaDB
- Embeddings: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- LLM: Ollama (Mistral model)
- Document Processing: LangChain
- Frontend: HTML/CSS/JavaScript

## Troubleshooting

If you encounter any issues:

1. Ensure Ollama is running in the background
2. Make sure you've pulled the Mistral model using `ollama pull mistral`
3. Check that all Python dependencies are installed correctly
4. Verify that the documents directory and vector store directory exist and are writable

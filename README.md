# Document Q&A System

A web-based document management and question-answering system that allows users to upload documents and ask questions about their content using AI-powered analysis.

## Features

- **Document Management**
  - Upload individual files (PDF, TXT, DOCX, HTML)
  - Upload entire directories with automatic file type filtering
  - View document metadata (size, upload date, processing chunks)
  - Delete documents when no longer needed
  - Progress tracking for uploads and processing

- **Document Summaries**
  - Automatic generation of document summaries
  - Summaries are cached on disk for faster access
  - Collapsible summary view for each document

- **AI-Powered Q&A**
  - Ask questions about uploaded documents
  - Get answers with relevant source citations
  - Multiple AI model support through Ollama
  - Model selection interface

## Prerequisites

- Python 3.13 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Virtual environment (recommended)
- Sufficient disk space for document storage and embeddings

## Dependencies

Core dependencies include:
- Flask 3.0.0 - Web framework
- Langchain 0.1.0 - LLM framework
- Langchain-community 0.0.10 - Community extensions
- ChromaDB 0.4.22 - Vector store
- PyPDF 3.17.1 - PDF processing
- Docx2txt 0.8 - Word document processing
- Sentence-transformers 2.2.2 - Text embeddings
- BeautifulSoup4 4.12.3 - HTML processing

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd corpuscular_ui
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Ollama is running and install required models:
   ```bash
   ollama serve
   ollama pull mistral  # Install default model
   ```

5. Start the application:
   ```bash
   python app.py
   ```

The application will be available at `http://127.0.0.1:5000`

## Project Structure

```
corpuscular_ui/
├── app.py              # Main Flask application
├── index.html          # Frontend interface
├── requirements.txt    # Python dependencies
├── documents/         # Uploaded documents storage
├── summaries/         # Document summaries storage
├── vectorstore/       # Vector embeddings storage
└── documents_metadata.json  # Document metadata
```

## Usage

1. **Upload Documents**
   - Click "Select Files" to upload individual files
   - Click "Select Directory" to upload an entire directory
   - Drag and drop files or directories onto the upload area

2. **View Documents**
   - See all uploaded documents in the "Available Documents" section
   - Click the arrow icon to view/hide document summaries
   - Delete documents using the delete button

3. **Ask Questions**
   - Select an AI model from the dropdown (defaults to Mistral)
   - Type your question in the input field
   - Click "Ask" to get an answer with relevant sources

## Technical Details

- Uses `langchain` and `langchain-community` for document processing
- Embeddings generated using HuggingFace's `all-MiniLM-L6-v2` model
- Vector storage handled by ChromaDB
- Document chunking with RecursiveCharacterTextSplitter
- Frontend built with vanilla JavaScript and modern CSS

## File Type Support

- PDF (`.pdf`)
- Plain Text (`.txt`)
- Microsoft Word (`.docx`)
- HTML (`.html`)

## Data Storage

- Documents are stored in the `documents/` directory
- Summaries are cached in the `summaries/` directory
- Vector embeddings are stored in `vectorstore/`
- All storage directories are git-ignored

## Error Handling

- Permission errors are caught and reported
- Network errors during uploads are handled gracefully
- Progress tracking for large file uploads
- Proper cleanup of resources on application shutdown

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## System Requirements

- **Disk Space**: Ensure sufficient space for:
  - Document storage (varies with upload size)
  - Vector embeddings (~100MB per 1000 pages)
  - Model storage (varies by model, ~4GB for Mistral)
  
- **Memory**: 
  - Minimum: 8GB RAM
  - Recommended: 16GB RAM for better performance with large documents
  
- **GPU**: Optional but recommended for faster embeddings generation

## Troubleshooting

Common issues and solutions:

1. **Permission Errors**
   - Ensure write permissions for `documents/`, `summaries/`, and `vectorstore/` directories
   - Run `chmod -R 755 .` in the project directory if needed

2. **Model Loading Issues**
   - Verify Ollama is running with `ollama list`
   - Check model installation with `ollama pull mistral`

3. **Memory Issues**
   - Reduce chunk size in `app.py` if processing large documents
   - Close other memory-intensive applications

4. **Slow Performance**
   - Consider using a GPU for embeddings generation
   - Adjust chunk size and overlap in document processing
   - Ensure sufficient system resources

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

4. Ensure Ollama is running:
   ```bash
   ollama serve
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

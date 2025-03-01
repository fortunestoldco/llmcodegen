# Other Tales CodeMaker

A Streamlit-based application that generates code solutions using AI models while ensuring compliance with SDK documentation.

## Features

- **Multi-Model Support**: 
  - OpenAI GPT-4 variants (o1, o3, GPT-4.5, GPT-4o-mini)
  - Anthropic Claude models (3.5, 3.7)
  - Replicate models
  - HuggingFace models

- **Documentation Processing**:
  - Automatic SDK documentation scraping
  - Support for multiple documentation URLs
  - Intelligent parsing of documentation structure
  - Vector storage for efficient retrieval

- **Vector Storage Options**:
  - MongoDB Atlas Vector Search
  - Local Chroma vectorstore

- **Real-time Features**:
  - Streaming responses
  - Progress tracking
  - Interactive chat interface
  - Code highlighting

## Prerequisites

- Python 3.8+
- MongoDB Atlas cluster (optional)
- API keys for:
  - OpenAI (required for embeddings)
  - Anthropic (optional)
  - Replicate (optional)
  - HuggingFace (optional)
  - Firecrawl (optional)

## Environment Variables

# Required
OPENAI_API_KEY=your_openai_key

# Optional
ANTHROPIC_API_KEY=your_anthropic_key
REPLICATE_API_TOKEN=your_replicate_token
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
FIRECRAWL_API_KEY=your_firecrawl_key

# MongoDB Configuration (if using MongoDB)
MONGODB_URI=your_mongodb_uri
MONGODB_DATABASE=sdk_documentation
MONGODB_STRUCTURED_COLLECTION=structured_docs
MONGODB_VECTOR_COLLECTION=vector_docs
MONGODB_VECTOR_INDEX=vector_index
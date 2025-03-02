import streamlit as st
import os
import re
import json
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo.server_api import ServerApi
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Replicate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

# Set page config
st.set_page_config(page_title="Other Tales CodeMaker", page_icon="🧩", layout="wide")

# App title and description
st.title("Other Tales CodeMaker")
st.markdown("""
This tool helps you generate and verify code solutions, ensuring compliance with SDK documentation. 
Simply provide your task along with the SDK/API Documentation URL(s) you want to use.
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "code_solution" not in st.session_state:
    st.session_state.code_solution = ""
if "chroma_db_path" not in st.session_state:
    st.session_state.chroma_db_path = "./chroma_db"
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if "replicate_api_token" not in st.session_state:
    st.session_state.replicate_api_token = os.environ.get("REPLICATE_API_TOKEN", "")
if "huggingface_api_token" not in st.session_state:
    st.session_state.huggingface_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY", "")
if "progress_status" not in st.session_state:
    st.session_state.progress_status = ""
if "progress_details" not in st.session_state:
    st.session_state.progress_details = []

# Sidebar for configuration
with st.sidebar:
    st.header("Model Configuration")
    llm_provider = st.selectbox(
        "Generative Model",
        [
            "OpenAI GPT-4o", 
            "OpenAI GPT-4.5 Preview", 
            "OpenAI GPT-4o-mini", 
            "OpenAI o1", 
            "OpenAI o1-preview",
            "OpenAI o1-mini",
            "OpenAI o3-mini",
            "Anthropic Claude 3.7 Sonnet", 
            "Anthropic Claude 3.5 Latest", 
            "Replicate Model", 
            "HuggingFace Hub"
        ]
    )
    
    # Show model name input field only for Replicate and HuggingFace
    custom_model = None
    if llm_provider in ["Replicate Model", "HuggingFace Hub"]:
        custom_model = st.text_input("Model", 
                                    value="anthropic/claude-3.7-sonnet" if llm_provider == "Replicate Model" else "HuggingFaceH4/zephyr-7b-beta",
                                    key="custom_model")
    
    # Credentials section
    with st.expander("Credentials", expanded=False):
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for OpenAI models and embeddings"
        )
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Anthropic API Key
        anthropic_api_key = st.text_input(
            "Anthropic API Key", 
            value=st.session_state.anthropic_api_key,
            type="password",
            help="Required for Anthropic models"
        )
        if anthropic_api_key:
            st.session_state.anthropic_api_key = anthropic_api_key
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        
        # Replicate API Token
        replicate_api_token = st.text_input(
            "Replicate Access Token", 
            value=st.session_state.replicate_api_token,
            type="password",
            help="Required for Replicate models"
        )
        if replicate_api_token:
            st.session_state.replicate_api_token = replicate_api_token
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        
        # HuggingFace API Token
        huggingface_api_token = st.text_input(
            "HuggingFace Access Token", 
            value=st.session_state.huggingface_api_token,
            type="password",
            help="Required for HuggingFace models"
        )
        if huggingface_api_token:
            st.session_state.huggingface_api_token = huggingface_api_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
            
        # Firecrawl API Key
        firecrawl_api_key = st.text_input(
            "Firecrawl API Key", 
            value=st.session_state.firecrawl_api_key,
            type="password",
            help="Optional: Enhanced web crawling and scraping capabilities"
        )
        if firecrawl_api_key:
            st.session_state.firecrawl_api_key = firecrawl_api_key
            os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

# Function to determine optimal model parameters for code generation
def determine_optimal_parameters(provider, sdk_name):
    # Default parameters that will be used as a base
    default_params = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4000
    }
    
    # Remove repetition_penalty for models that don't support it
    if "OpenAI" in provider or "Anthropic" in provider:
        if "repetition_penalty" in default_params:
            del default_params["repetition_penalty"]
    
    # Get optimal parameters based on model provider and task
    prompt = f"""
    I need to determine the optimal parameters for using a {provider} language model for code generation with {sdk_name} SDK.
    Based on your knowledge of code generation tasks, what would be the ideal settings for:
    1. temperature
    2. top_p
    3. max_tokens or max_length
    4. repetition_penalty (if applicable)
    
    Please respond with a JSON object containing these parameters.
    """
    
    # Determine the base model to use for this parameter optimization task
    if "OpenAI" in provider and st.session_state.openai_api_key:
        # Use a reliable model to determine parameters
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=st.session_state.openai_api_key)
        messages = [
            SystemMessage(content="You are an expert in optimizing LLM parameters for code generation tasks."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages).content
    elif "Anthropic" in provider and st.session_state.anthropic_api_key:
        llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0, api_key=st.session_state.anthropic_api_key)
        messages = [
            SystemMessage(content="You are an expert in optimizing LLM parameters for code generation tasks."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages).content
    else:
        # For other providers or if keys are missing, return default parameters
        return default_params
    
    # Try to extract a JSON object from the response
    try:
        # Find JSON pattern in the response
        pattern = r'\{.*\}'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            params = json.loads(match.group())
            # Update the default parameters with the optimized ones
            default_params.update(params)
    except:
        # If parsing fails, use the default parameters
        pass
    
    return default_params

# Function to get the appropriate LLM based on user selection and optimized parameters
def get_llm(provider, task=None, sdk_name=None, temperature=0.2):
    # Get optimized parameters if task and sdk_name are provided
    params = {}
    if task and sdk_name:
        params = determine_optimal_parameters(provider, sdk_name)
    else:
        params = {"temperature": temperature}

    if "OpenAI" in provider:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API Key is required for OpenAI models. Please provide it in the sidebar.")
            return None
            
        model_name = ""
        if provider == "OpenAI GPT-4o":
            model_name = "gpt-4o-latest"
        elif provider == "OpenAI GPT-4.5 Preview":
            model_name = "gpt-4.5-preview"
        elif provider == "OpenAI GPT-4o-mini":
            model_name = "gpt-4o-mini"
        elif provider == "OpenAI o1":
            model_name = "o1"
        elif provider == "OpenAI o1-preview":
            model_name = "o1-preview"
        elif provider == "OpenAI o1-mini":
            model_name = "o1-mini"
        elif provider == "OpenAI o3-mini":
            model_name = "o3-mini"
            
        # Extract only supported parameters
        model_params = {
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.95),
            "max_tokens": params.get("max_tokens", 4000)
        }
            
        return ChatOpenAI(
            model=model_name,
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            max_tokens=model_params["max_tokens"],
            api_key=st.session_state.openai_api_key,
            streaming=True
        )
            
    elif "Anthropic" in provider:
        if not st.session_state.anthropic_api_key:
            st.error("Anthropic API Key is required for Claude models. Please provide it in the sidebar.")
            return None
            
        model_name = ""
        if provider == "Anthropic Claude 3.7 Sonnet":
            model_name = "claude-3-7-sonnet-20250219"
        elif provider == "Anthropic Claude 3.5 Latest":
            model_name = "claude-3-5-sonnet-latest"
            
        return ChatAnthropic(
            model=model_name, 
            model_kwargs=params, 
            api_key=st.session_state.anthropic_api_key,
            streaming=True  # Enable streaming for real-time updates
        )
        
    elif provider == "Replicate Model" and custom_model:
        if not st.session_state.replicate_api_token:
            st.error("Replicate Access Token is required for Replicate models. Please provide it in the sidebar.")
            return None
            
        # For Replicate, handle parameters in model_kwargs format
        model_kwargs = {
            "temperature": params.get("temperature", 0.2),
            "max_length": params.get("max_tokens", 500),
            "top_p": params.get("top_p", 1)
        }
        
        # Fix: Create a Replicate instance with streaming callback
        # This ensures the full response is properly captured
        return Replicate(
            model=custom_model,
            model_kwargs=model_kwargs,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            api_token=st.session_state.replicate_api_token  # Keep api_token separate from model_kwargs
        )
        
    elif provider == "HuggingFace Hub" and custom_model:
        if not st.session_state.huggingface_api_token:
            st.error("HuggingFace Access Token is required for HuggingFace Hub models. Please provide it in the sidebar.")
            return None
            
        # Move all parameters to model_kwargs to avoid conflicts
        model_kwargs = {
            "max_new_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.95),
            "do_sample": True
        }
            
        llm = HuggingFaceEndpoint(
            repo_id=custom_model,
            task="text-generation",
            model_kwargs=model_kwargs,
            token=st.session_state.huggingface_api_token,
            streaming=True
        )
        return ChatHuggingFace(llm=llm)
    else:
        # If nothing valid is selected, return None
        st.error("Please select a valid model provider and ensure you have the required API keys set.")
        return None

# Check if MongoDB URI is available, otherwise use Chroma
def get_vector_store_type():
    mongodb_uri = os.getenv("MONGODB_URI")
    if mongodb_uri:
        return "mongodb"
    else:
        if not os.path.exists(st.session_state.chroma_db_path):
            os.makedirs(st.session_state.chroma_db_path)
        return "chroma"

# Additional indexes for structured_docs collection
def create_structured_docs_indexes(db, collection_name):
    try:
        # Compound index for library and version
        db[collection_name].create_index(
            [("library", pymongo.ASCENDING), ("version", pymongo.ASCENDING)],
            unique=True,
            background=True
        )
        
        # Supporting indexes
        db[collection_name].create_index(
            [("last_updated", pymongo.ASCENDING)],
            expireAfterSeconds=7776000,  # 90 days
            background=True
        )
        
        # Supporting indexes
        db[collection_name].create_index(
            [("modules.name", pymongo.ASCENDING)],
            background=True
        )
        db[collection_name].create_index(
            [("imports.statement", pymongo.ASCENDING)],
            background=True
        )
        
        # Add index for examples
        db[collection_name].create_index(
            [("examples.module", pymongo.ASCENDING)],
            background=True
        )
        db[collection_name].create_index(
            [("examples.type", pymongo.ASCENDING)],
            background=True
        )
    except Exception as e:
        st.warning(f"Warning: Could not create all indexes for {collection_name}: {e}")

# Additional indexes for vector_docs collection
def create_vector_docs_indexes(db, collection_name):
    try:
        # Compound index for metadata
        db[collection_name].create_index(
            [
                ("metadata.library", pymongo.ASCENDING),
                ("metadata.version", pymongo.ASCENDING)
            ],
            background=True
        )
        
        # Full-text search index
        db[collection_name].create_index(
            [("page_content", "text")],
            background=True
        )
    except Exception as e:
        st.warning(f"Warning: Could not create all indexes for {collection_name}: {e}")

# MongoDB connection setup
def get_mongodb_connection():
    try:
        vector_store_type = get_vector_store_type()
        if vector_store_type == "mongodb":
            # Use environment variables for all MongoDB settings
            try:
                mongo_uri = os.getenv("MONGODB_URI", st.secrets["mongodb"]["uri"])
                # Get database and collection names from environment variables
                mongo_db_name = os.getenv("MONGODB_DATABASE", "sdk_documentation")
                mongo_structured_collection = os.getenv("MONGODB_STRUCTURED_COLLECTION", "structured_docs")
                mongo_vector_collection = os.getenv("MONGODB_VECTOR_COLLECTION", "vector_docs")
                mongo_vector_index = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
            except:
                mongo_uri = os.getenv("MONGODB_URI")
                mongo_db_name = os.getenv("MONGODB_DATABASE", "sdk_documentation")
                mongo_structured_collection = os.getenv("MONGODB_STRUCTURED_COLLECTION", "structured_docs")
                mongo_vector_collection = os.getenv("MONGODB_VECTOR_COLLECTION", "vector_docs")
                mongo_vector_index = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
                
            client = pymongo.MongoClient(mongo_uri, server_api=ServerApi('1'))
            
            # Store the configuration in the session state for reuse
            st.session_state.mongodb_config = {
                "database": mongo_db_name,
                "structured_collection": mongo_structured_collection,
                "vector_collection": mongo_vector_collection,
                "vector_index": mongo_vector_index
            }
            
            # Ping the database to confirm connection
            client.admin.command('ping')
            
            # Initialize collections with proper configurations
            db = client[mongo_db_name]
            
            # Check if collections exist, if not create them
            if mongo_structured_collection not in db.list_collection_names():
                # Structured docs collection - simple clustered index on _id
                db.create_collection(
                    mongo_structured_collection,
                    clusteredIndex={
                        "key": { "_id": 1 },
                        "unique": True
                    }
                )
                
                # Add compound index for efficient lookups
                db[mongo_structured_collection].create_index(
                    [
                        ("library", pymongo.ASCENDING),
                        ("version", pymongo.ASCENDING)
                    ],
                    unique=True,
                    background=True
                )
            
            if mongo_vector_collection not in db.list_collection_names():
                # Vector docs collection - simple clustered index on _id and TTL index
                db.create_collection(
                    mongo_vector_collection,
                    clusteredIndex={
                        "key": { "_id": 1 },
                        "unique": True
                    }
                )
                
                # Add compound index for metadata filtering
                db[mongo_vector_collection].create_index(
                    [
                        ("metadata.library", pymongo.ASCENDING),
                        ("metadata.version", pymongo.ASCENDING),
                        ("metadata.timestamp", pymongo.ASCENDING)
                    ],
                    background=True
                )
            
            # Create standard indexes for efficient querying
            create_structured_docs_indexes(db, mongo_structured_collection)
            create_vector_docs_indexes(db, mongo_vector_collection)
            
            st.session_state.db_connection_error = None
            return client
            
        else:
            st.session_state.db_connection_error = None
            return None
    except Exception as e:
        st.session_state.db_connection_error = str(e)
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to get embeddings based on available API keys
def get_embeddings():
    if st.session_state.openai_api_key:
        return OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    else:
        st.error("OpenAI API Key is required for generating embeddings. Please provide it in the sidebar.")
        return None

# Function to update progress with detailed information
def update_progress(message, level="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.progress_status = message
    st.session_state.progress_details.append({"time": timestamp, "message": message, "level": level})

# Parse URL path for SDK documentation
def parse_sdk_url_path(url):
    """Extract SDK path for proper documentation fetching."""
    parsed_url = urlparse(url)
    
    # Remove any potential prefix like /develop/api-reference
    path = parsed_url.path.lstrip('/')
    
    # Split by slashes to get path components
    path_components = path.split('/')
    
    # Check if there are path prefixes that aren't part of the actual SDK path
    prefix_patterns = ['develop', 'api-reference', 'docs', 'reference', 'api']
    
    # Filter out common prefixes
    sdk_path_components = [comp for comp in path_components if comp.lower() not in prefix_patterns]
    
    # Reconstruct the SDK path
    sdk_path = '/'.join(sdk_path_components)
    
    return sdk_path.strip('/')

# Function to scrape the SDK documentation using Firecrawl or fallback to BeautifulSoup
def scrape_documentation(url):
    update_progress(f"Starting documentation crawl for {url}...")
    
    # Parse the URL path to extract the SDK information
    sdk_path = parse_sdk_url_path(url)
    update_progress(f"Extracted SDK path: {sdk_path}")
    
    if st.session_state.firecrawl_api_key:
        try:
            # Determine base URL and allowed patterns
            parsed_url = urlparse(url)
            base_domain = parsed_url.netloc
            base_path = parsed_url.path.split('/')[1] if len(parsed_url.path.split('/')) > 1 else ''
            
            update_progress(f"Using Firecrawl to crawl {url}...")
            
            loader = FireCrawlLoader(
                api_key=st.session_state.firecrawl_api_key,
                url=url,
                mode="crawl",  # Use crawl mode to get all accessible pages
                config={
                    "max_pages": 100,
                    "max_depth": 4,
                    "follow_links": True,
                    "allowed_patterns": [
                        f"https?://{base_domain}/{base_path}/.*api.*",
                        f"https?://{base_domain}/{base_path}/.*reference.*",
                        f"https?://{base_domain}/{base_path}/.*docs.*",
                        f"https?://{base_domain}/{base_path}/.*guide.*"
                    ],
                    "extract_rules": {
                        "api_content": {
                            "selectors": [
                                "main", 
                                "article",
                                ".documentation",
                                ".api-documentation",
                                ".reference",
                                "#main-content"
                            ]
                        },
                        "code_blocks": {
                            "selectors": [
                                "pre code",
                                ".highlight",
                                ".example-code",
                                "[class*='language-python']"
                            ],
                            "attributes": ["class"]
                        },
                        "method_definitions": {
                            "selectors": [
                                ".method",
                                ".function",
                                "dl.py.method",
                                "dl.py.function"
                            ],
                            "surrounding_text": True
                        }
                    },
                    "custom_headers": {
                        "User-Agent": "Mozilla/5.0 (compatible; APIDocCrawler/1.0;)"
                    }
                }
            )

            update_progress("Starting API documentation crawl...")
            docs = loader.load()
            
            if docs:
                update_progress(f"FireCrawl found {len(docs)} pages, analyzing content...")
                # Log the first few URLs found
                for i, doc in enumerate(docs[:3]):
                    if i == 0:
                        update_progress(f"Sample URLs found: {doc.metadata.get('source', 'unknown')}")
            
            if not docs:
                update_progress("No documentation found, falling back to basic scraping", "warning")
                return fallback_scrape_documentation(url)
            
            # Process the crawled documentation
            processed_docs = []
            for doc in docs:
                # Clean and structure the content
                cleaned_content = process_api_content(doc.page_content)
                if cleaned_content:
                    processed_docs.append(Document(
                        page_content=cleaned_content,
                        metadata={
                            **doc.metadata,
                            "processed_at": datetime.now(timezone.utc).isoformat()
                        }
                    ))

            if processed_docs:
                update_progress(f"Successfully processed {len(processed_docs)} API documentation pages")
                # Implement the missing function
                documentation = create_structured_documentation(processed_docs)
                return documentation, processed_docs
            else:
                update_progress("No valid API documentation found, falling back to basic scraping", "warning")
                return fallback_scrape_documentation(url)

        except Exception as e:
            update_progress(f"Firecrawl error: {str(e)}", "error")
            return fallback_scrape_documentation(url)
    else:
        return fallback_scrape_documentation(url)

def process_api_content(content):
    """Clean and structure API documentation content."""
    if not content.strip():
        return None
        
    # Remove common noise
    content = re.sub(r'^\s*>>>\s*', '', content, flags=re.MULTILINE)  # Remove REPL prompts
    content = re.sub(r'^\s*\.\.\.\s*', '', content, flags=re.MULTILINE)  # Remove REPL continuation
    content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize whitespace
    
    # Identify and mark code blocks
    content = re.sub(
        r'```(?:python|py)?(.*?)```',
        lambda m: f'CODE_BLOCK_START\n{m.group(1).strip()}\nCODE_BLOCK_END',
        content,
        flags=re.DOTALL
    )
    
    return content.strip()

def create_structured_documentation(processed_docs):
    """Convert processed document chunks into structured documentation."""
    # Extract basic library info
    library_info = extract_library_info(processed_docs)
    
    # Extract modules, classes, functions
    modules = extract_modules(processed_docs)
    
    # Extract import statements
    imports = extract_imports(processed_docs)
    
    # Create the structured documentation
    documentation = {
        "library": library_info["name"],
        "version": library_info["version"],
        "description": library_info["description"],
        "modules": modules,
        "imports": imports,
        "examples": [],  # Will be populated with global examples
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    return documentation

def extract_library_info(docs):
    """Extract library name, version, and description from documentation."""
    info = {
        "name": "Unknown",
        "version": "Latest",
        "description": ""
    }
    
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        
        # Look for package info in common locations
        if "package" in metadata:
            info["name"] = metadata["package"]
        
        # Try to find version
        version_patterns = [
            r'version\s*[=:]\s*([\d\.]+)',
            r'v([\d\.]+)',
            r'Version\s+([\d\.]+)'
        ]
        for pattern in version_patterns:
            match = re.search(pattern, content)
            if match:
                info["version"] = match.group(1)
                break
                
        # Look for package description
        desc_patterns = [
            r'"""(.*?)"""',
            r'Description\s*[-:=]\s*([^\n]+)'
        ]
        for pattern in desc_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                info["description"] = match.group(1).strip()
                break
                
    return info

def extract_modules(docs):
    """Extract module information from documentation."""
    modules = []
    current_module = None
    
    for doc in docs:
        content = doc.page_content
        
        # Look for module definitions
        module_matches = re.finditer(r'(?:class|module)\s+(\w+)(?:\(([^)]+)\))?\s*:', content)
        for match in module_matches:
            module_name = match.group(1)
            parent_class = match.group(2) if match.groups() > 1 else None
            
            # Get module description
            desc_match = re.search(f'{module_name}\s*(?:\([^)]+\))?\s*:(.*?)(?:class|def|$)', content, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Extract methods and examples
            methods = extract_methods(content, module_name)
            examples = extract_examples(content, module_name)
            
            modules.append({
                "name": module_name,
                "parent_class": parent_class,
                "description": description,
                "methods": methods,
                "examples": examples,  # Add examples to module
                "classes": [],
                "source": doc.metadata.get("source", "")
            })
    
    return modules

def extract_methods(content, class_name):
    """Extract method information from a class definition."""
    methods = []
    
    # Look for method definitions
    method_pattern = r'def\s+(\w+)\s*\((self,?\s*[^)]*)\)[^:]*:(.*?)(?=\n\s*def|\Z)'
    for match in re.finditer(method_pattern, content, re.DOTALL):
        method_name = match.group(1)
        parameters = match.group(2)
        method_body = match.group(3)
        
        # Parse docstring if present
        docstring_match = re.search(r'"""(.*?)"""', method_body, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""
        
        # Parse parameters
        param_list = []
        for param in parameters.split(','):
            param = param.strip()
            if param and param != 'self':
                param_parts = param.split(':')
                param_list.append({
                    "name": param_parts[0].strip(),
                    "type": param_parts[1].strip() if len(param_parts) > 1 else "Any",
                    "description": ""  # Could be extracted from docstring if needed
                })
        
        methods.append({
            "name": method_name,
            "description": docstring,
            "parameters": param_list,
            "returns": extract_return_info(docstring)
        })
        
    return methods

def extract_return_info(docstring):
    """Extract return type and description from docstring."""
    return_info = {"type": "None", "description": ""}
    
    if not docstring:
        return return_info
    
    # Look for common return patterns in docstrings
    return_patterns = [
        r'(?:Returns|Return type):\s*(.*?)(?:$|\n)',
        r'@return:\s*(.*?)(?:$|\n)',
        r'@rtype:\s*(.*?)(?:$|\n)'
    ]
    
    for pattern in return_patterns:
        match = re.search(pattern, docstring, re.DOTALL)
        if match:
            return_text = match.group(1).strip()
            
            # Try to separate type and description
            parts = return_text.split(' ', 1)
            if len(parts) > 1:
                return_info["type"] = parts[0].strip()
                return_info["description"] = parts[1].strip()
            else:
                return_info["type"] = return_text
            
            break
    
    return return_info

def extract_examples(content, context_name=None):
    """Extract example code and usage patterns."""
    examples = []
    
    # Common patterns for finding examples
    example_patterns = [
        # Example block with heading
        r'(?:Example|Usage)[\s\-:]+\s*(```(?:python|py)?\s*[\s\S]*?```)',
        # Code block with context
        r'(?:class|def)\s+' + (context_name or r'\w+') + r'[\s\S]*?(```(?:python|py)?\s*[\s\S]*?```)',
        # General code blocks
        r'```(?:python|py)?\s*([\s\S]*?)```'
    ]
    
    for pattern in example_patterns:
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            example_text = match.group(1).strip()
            
            # Clean up the example
            example_text = re.sub(r'```(?:python|py)?\s*', '', example_text)
            example_text = example_text.replace('```', '').strip()
            example_text = re.sub(r'^\s*>>>\s*', '', example_text, flags=re.MULTILINE)
            
            # Verify it's a valid code example
            if any(keyword in example_text for keyword in ['import', 'def', 'class', '=', 'return']):
                examples.append({
                    "code": example_text,
                    "context": context_name or "global",
                    "type": "example"
                })
    
    return examples

def extract_imports(docs):
    """Extract import statements and their context."""
    imports = []
    seen_imports = set()
    
    for doc in docs:
        content = doc.page_content
        
        # Find all import statements
        import_patterns = [
            r'^import\s+([^;#\n]+)',
            r'^from\s+([^;#\n]+)\s+import\s+([^;#\n]+)'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if pattern.startswith(r'^import'):
                    import_stmt = f"import {match.group(1).strip()}"
                else:
                    import_stmt = f"from {match.group(1).strip()} import {match.group(2).strip()}"
                
                if import_stmt not in seen_imports:
                    seen_imports.add(import_stmt)
                    imports.append({
                        "statement": import_stmt,
                        "description": ""
                    })
    
    return imports

# Fallback scraping function using BeautifulSoup
def fallback_scrape_documentation(url):
    try:
        update_progress(f"Scraping documentation from {url} using BeautifulSoup...")
        response = requests.get(url)
        response.raise_for_status()
        
        update_progress(f"Successfully retrieved page content from {url}")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text content
        content = soup.get_text()
        update_progress(f"Extracted text content from {url}")
        
        # Process HTML content to extract structured information
        # This is a simplified version - a more robust implementation would parse 
        # the specific structure of the documentation site
        
        # For demonstration, we'll collect basic information
        library_name = "Unknown"
        library_version = "Latest"
        
        # Try to find a title that might contain the library name
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text()
            # Extract library name from title if possible
            match = re.search(r'([\w\-]+) Documentation', title_text)
            if match:
                library_name = match.group(1)
        
        # Parse modules, classes, methods, etc.
        modules = []
        imports = []
        
        update_progress(f"Looking for import statements and modules in {url}")
        # Look for import statements in code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        for block in code_blocks:
            text = block.get_text()
            if 'import' in text:
                for line in text.split('\n'):
                    if 'import' in line and not line.strip().startswith('#'):
                        cleaned_line = line.strip()
                        if cleaned_line not in [i["statement"] for i in imports if "statement" in i]:
                            imports.append({
                                "statement": cleaned_line,
                                "description": "Import found in documentation"
                            })
        
        # Extract modules and classes
        # This is highly dependent on the structure of the documentation
        # For a comprehensive solution, we would need to parse the specific HTML structure
        
        # Create sections based on heading hierarchy
        update_progress(f"Parsing document structure from {url}")
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        current_module = None
        
        for heading in headings:
            heading_text = heading.get_text().strip()
            
            # New module
            if heading.name == 'h1' or heading.name == 'h2':
                current_module = {
                    "name": heading_text,
                    "description": "",
                    "classes": [],
                    "functions": [],
                    "enums": []
                }
                
                # Get description from the next paragraph
                next_p = heading.find_next('p')
                if next_p:
                    current_module["description"] = next_p.get_text().strip()
                
                modules.append(current_module)
                
            # Class or function within a module
            elif heading.name == 'h3' and current_module:
                # Check if it looks like a class (typically capitalized)
                if heading_text[0].isupper() and not heading_text.startswith(('GET', 'POST', 'PUT', 'DELETE')):
                    new_class = {
                        "name": heading_text,
                        "description": "",
                        "attributes": [],
                        "methods": []
                    }
                    
                    # Get description from the next paragraph
                    next_p = heading.find_next('p')
                    if next_p:
                        new_class["description"] = next_p.get_text().strip()
                    
                    current_module["classes"].append(new_class)
                else:
                    # Assume it's a function
                    new_function = {
                        "name": heading_text,
                        "description": "",
                        "signature": "",
                        "is_async": "async" in heading_text,
                        "parameters": [],
                        "returns": {"type": "", "description": ""},
                        "examples": []
                    }
                    
                    # Get description from the next paragraph
                    next_p = heading.find_next('p')
                    if next_p:
                        new_function["description"] = next_p.get_text().strip()
                    
                    # Look for signature in next code block
                    next_code = heading.find_next(['pre', 'code'])
                    if next_code:
                        new_function["signature"] = next_code.get_text().strip()
                    
                    current_module["functions"].append(new_function)
        
        update_progress(f"Found {len(modules)} modules and {len(imports)} import statements")
        
        # Create the final documentation object
        documentation = {
            "library": library_name,
            "version": library_version,
            "modules": modules,
            "imports": imports,
            "last_updated": datetime.now().isoformat()
        }
        
        # Also create a list of text chunks for vector storage
        update_progress(f"Splitting content into chunks for vector storage")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_text(content)
        
        # Creating Document objects with metadata
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "library": library_name,
                    "version": library_version,
                    "source": url,
                    "chunk_id": i,
                    "timestamp": datetime.now(timezone.utc)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        update_progress(f"Created {len(docs)} document chunks from {url}")
        return documentation, docs
    
    except Exception as e:
        update_progress(f"Error scraping documentation: {e}", "error")
        return None, None

# Function to store documentation (MongoDB or Chroma)
def store_documentation(client, documentation, vector_docs):
    try:
        update_progress("Storing documentation in vector database...")
        vector_store_type = get_vector_store_type()
        
        # Get embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return False
            
        if vector_store_type == "mongodb":
            config = st.session_state.mongodb_config
            db = client[config["database"]]
            
            # Ensure vector search index exists
            try:
                db.command({
                    "createIndexes": config["vector_collection"],
                    "indexes": [
                        {
                            "name": config["vector_index"],
                            "key": {
                                "vector": "knnVector"
                            },
                            "definition": {
                                "dimensions": 1536,  # OpenAI embedding dimensions
                                "similarity": "cosine"
                            }
                        }
                    ]
                })
                update_progress("Vector search index created/verified")
            except Exception as e:
                update_progress(f"Warning: Vector index creation failed: {str(e)}", "warning")
                # Continue anyway as the index might already exist

            # Store the structured documentation
            doc_collection = db[config["structured_collection"]]
            
            # Check if we already have this library version
            existing = doc_collection.find_one({
                "library": documentation["library"],
                "version": documentation["version"]
            })
            
            if existing:
                update_progress(f"Updating existing documentation for {documentation['library']}")
                doc_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": documentation}
                )
            else:
                update_progress(f"Inserting new documentation for {documentation['library']}")
                doc_collection.insert_one(documentation)
            
            # Store vector documents
            vector_collection = db[config["vector_collection"]]
            
            # Create the vector store
            update_progress(f"Creating vector embeddings for {len(vector_docs)} documents")
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                vector_docs,
                embeddings,
                collection=vector_collection,
                index_name=config["vector_index"],
            )
        else:
            # Use Chroma for local vector storage
            # Store structured documentation
            structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
            os.makedirs(structured_docs_path, exist_ok=True)
            
            # Generate a unique filename based on library and version
            update_progress(f"Storing structured documentation for {documentation['library']}")
            filename = f"{documentation['library']}_{documentation['version']}.json"
            with open(os.path.join(structured_docs_path, filename), 'w') as f:
                json.dump(documentation, f)
            
            # Store vector documents in Chroma
            update_progress(f"Creating vector embeddings with Chroma for {len(vector_docs)} documents")
            vector_store = Chroma.from_documents(
                documents=vector_docs,
                embedding=embeddings,
                persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store")
            )
            vector_store.persist()
        
        update_progress("Documentation stored successfully!")
        return True
    
    except Exception as e:
        update_progress(f"Error storing documentation: {e}", "error")
        return False

# Function to search for documentation (MongoDB or Chroma)
def search_documentation(client, query, library=None):
    try:
        update_progress(f"Searching documentation for '{query}'...")
        vector_store_type = get_vector_store_type()
        
        # Get embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return [], []
            
        if vector_store_type == "mongodb":
            config = st.session_state.mongodb_config
            db = client[config["database"]]
            
            # Search in vector store
            vector_collection = db[config["vector_collection"]]
            
            # Create the vector store
            vector_store = MongoDBAtlasVectorSearch(
                collection=vector_collection,
                embedding=embeddings,
                index_name=config["vector_index"],
            )
            
            # Add library filter if provided
            search_filter = {"metadata.library": library} if library else None
            
            # Search for similar documents
            results = vector_store.similarity_search(query, k=10, filter=search_filter)
            update_progress(f"Found {len(results)} relevant document chunks")
            
            # Also get the structured documentation
            structured_collection = db[config["structured_collection"]]
            
            # If we have a library, get its specific documentation
            structured_docs = list(structured_collection.find(
                {"library": library} if library else {}
            ))
            update_progress(f"Found {len(structured_docs)} structured documentation entries")
        else:
            # Use Chroma for local vector search
            # Load vector store
            try:
                update_progress("Loading local Chroma vector store")
                vector_store = Chroma(
                    persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store"),
                    embedding_function=embeddings
                )
                
                # Search for similar documents, filter by library if provided
                if library:
                    update_progress(f"Searching for '{query}' with library filter: {library}")
                    results = vector_store.similarity_search(
                        query=query,
                        k=10,
                        filter={"metadata.library": library}
                    )
                else:
                    update_progress(f"Searching for '{query}' across all libraries")
                    results = vector_store.similarity_search(query, k=10)
                update_progress(f"Found {len(results)} relevant document chunks")
                
                # Get structured documentation
                structured_docs = []
                structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
                
                if os.path.exists(structured_docs_path):
                    update_progress("Loading structured documentation from local files")
                    for filename in os.listdir(structured_docs_path):
                        if filename.endswith('.json'):
                            if library and not filename.startswith(f"{library}_"):
                                continue
                            
                            with open(os.path.join(structured_docs_path, filename), 'r') as f:
                                doc = json.load(f)
                                structured_docs.append(doc)
                    update_progress(f"Found {len(structured_docs)} structured documentation entries")
            except Exception as e:
                update_progress(f"Error loading Chroma vector store: {e}. Creating a new one.", "warning")
                results = []
                structured_docs = []
        
        return results, structured_docs
    
    except Exception as e:
        update_progress(f"Error searching documentation: {e}", "error")
        return [], []

# Function to clean code from timestamp markers
def clean_code_from_timestamps(code_text):
    # Remove timestamp markers like [2025-03-01 19:54:57.110174]
    pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\]'
    clean_text = re.sub(pattern, '', code_text)
    
    # Extract code blocks if present
    code_pattern = r'''(?:python|py)(.*?)'''
    code_matches = re.findall(code_pattern, clean_text, re.DOTALL)
    
    if code_matches:
        return code_matches[0].strip()
    
    # If no code blocks found, try to identify and return just the code portion
    lines = clean_text.split('\n')
    code_lines = []
    in_code_section = False
    
    for line in lines:
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            in_code_section = True
        
        if in_code_section:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # If no obvious code sections found, return the cleaned text
    return clean_text.strip()

# Custom streaming callback handler for Streamlit
class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""
        self.code_detected = False
        self.in_code_block = False
        self.code_block = ""
        
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        
        # Look for code block markers
        if "'''python" in token or "'''py" in token:
            self.in_code_block = True
            self.code_detected = True
        elif self.in_code_block and "'''" in token:
            self.in_code_block = False
        
        # If we're in a code block, collect the code
        if self.in_code_block:
            self.code_block += token
        
        # Update the Streamlit container
        # If code is detected, show it properly formatted
        if self.code_detected:
            # Try to extract the code block
            code_pattern = r'''(?:python|py)(.*?)'''
            code_matches = re.findall(code_pattern, self.text, re.DOTALL)
            
            if code_matches:
                # Show the explanation text above the code block
                explanation_text = self.text.split("'''python")[0] if "'''python" in self.text else self.text.split("'''py")[0]
                self.container.markdown(explanation_text)
                
                # Show the code in a dedicated code block with proper formatting
                # FIX: Use code with language parameter to preserve formatting
                code_content = code_matches[-1].strip()
                self.container.code(code_content, language="python")
                
                # Show any text that follows the code block
                if "'''" in self.text:
                    after_code = self.text.split("'''")[-1]
                    if after_code.strip():
                        self.container.markdown(after_code)
            else:
                # If no complete code block yet, just show the text
                self.container.markdown(self.text)
        else:
            # No code detected yet, just show the text
            self.container.markdown(self.text)
        
    def on_llm_end(self, response, **kwargs):
        # Final cleanup and processing
        # Extract code block if present
        code_pattern = r'''(?:python|py)(.*?)'''
        code_matches = re.findall(code_pattern, self.text, re.DOTALL)
        
        if code_matches:
            # Return the last code block
            return code_matches[-1].strip()
        
        # If no code blocks, clean the text and try to extract code
        clean_text = clean_code_from_timestamps(self.text)
        return clean_text

# Function to verify code quality and format
def verify_code_quality(code_text):
    """Verify code quality and format."""
    issues = []
    
    # Remove code block markers if present
    if "'''python" in code_text or "'''py" in code_text:
        code_pattern = r'''(?:python|py)(.*?)'''
        code_matches = re.findall(code_pattern, code_text, re.DOTALL)
        if code_matches:
            code_text = code_matches[0].strip()
    
    # 1. Check indentation
    lines = code_text.split('\n')
    indent_pattern = r'^( {4})*[^ ]'
    for i, line in enumerate(lines, 1):
        if line.strip() and not re.match(indent_pattern, line):
            issues.append(f"Line {i}: Incorrect indentation. Use 4 spaces for indentation.")
    
    # 2. Check regex patterns
    regex_pattern = r'r[\'"].*?[\'"]'
    for i, line in enumerate(lines, 1):
        if 're.' in line or 'regex' in line.lower():
            # Check for raw string usage
            if 'r"' not in line and "r'" not in line:
                issues.append(f"Line {i}: Regex pattern should use raw string (prefix with r)")
            # Check for escaped characters
            if '\\' in line and not line.strip().startswith('r'):
                issues.append(f"Line {i}: Potential regex escape sequence issue")
    
    # 3. Check quotation consistency
    for i, line in enumerate(lines, 1):
        # Count different quote types
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        if single_quotes % 2 != 0:
            issues.append(f"Line {i}: Unmatched single quotes")
        if double_quotes % 2 != 0:
            issues.append(f"Line {i}: Unmatched double quotes")
        # Check mixed usage in same line
        if single_quotes > 0 and double_quotes > 0 and 'r"' not in line and "r'" not in line:
            issues.append(f"Line {i}: Mixed quote usage - stick to one style")
    
    # 4. Check encapsulation
    brackets = {'(': ')', '[': ']', '{': '}'}
    for i, line in enumerate(lines, 1):
        stack = []
        for char in line:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or char != brackets[stack.pop()]:
                    issues.append(f"Line {i}: Mismatched brackets")
                    break
        if stack:
            issues.append(f"Line {i}: Unclosed brackets")
    
    # 5. Check for unparsable characters
    unparsable_pattern = r'[^\x00-\x7F]+'
    for i, line in enumerate(lines, 1):
        if re.search(unparsable_pattern, line):
            issues.append(f"Line {i}: Contains non-ASCII characters")
    
    # 6. Check formatting
    for i, line in enumerate(lines, 1):
        # Check spacing around operators
        if re.search(r'\w[+\-*/=]=\w', line):
            issues.append(f"Line {i}: Missing spaces around operators")
        # Check comma spacing
        if re.search(r'\w,\w', line):
            issues.append(f"Line {i}: Missing space after comma")
        # Check line length
        if len(line) > 120:
            issues.append(f"Line {i}: Line too long (exceeds 120 characters)")
    
    return issues

# Function to generate code solution with proper formatting
def generate_code_solution(task, vector_results, structured_docs):
    try:
        update_progress("Generating code solution...")
        
        # Extract library name from structured docs for parameter optimization
        sdk_name = "Unknown"
        if structured_docs and len(structured_docs) > 0:
            if "library" in structured_docs[0]:
                sdk_name = structured_docs[0]["library"]
                update_progress(f"Optimizing model parameters for {sdk_name} SDK")

        # Format the vector chunks to be readable
        # Ensure they're properly formatted with clear sections
        vector_chunks = ""
        for i, doc in enumerate(vector_results):
            vector_chunks += f"--- DOCUMENTATION CHUNK {i+1} ---\n"
            vector_chunks += f"SOURCE: {doc.metadata.get('source', 'Unknown')}\n"
            vector_chunks += f"{doc.page_content}\n\n"
        
        update_progress(f"Prepared {len(vector_results)} documentation chunks for context")
        
        # Format the structured docs to be more readable and usable for the LLM
        structured_docs_formatted = []
        for doc in structured_docs:
            # Extract critical information for the LLM
            formatted_doc = {
                "library": doc.get("library", "Unknown"),
                "version": doc.get("version", "Latest"),
                "description": doc.get("description", ""),
                "imports": doc.get("imports", []),
                "modules": []
            }
            
            # Format modules to be more concise and focused on what the LLM needs
            for module in doc.get("modules", []):
                formatted_module = {
                    "name": module.get("name", ""),
                    "description": module.get("description", ""),
                    "methods": []
                }
                
                # Add key methods with their parameters
                for method in module.get("methods", []):
                    formatted_module["methods"].append({
                        "name": method.get("name", ""),
                        "description": method.get("description", ""),
                        "parameters": method.get("parameters", []),
                        "returns": method.get("returns", {"type": "None", "description": ""})
                    })
                
                formatted_doc["modules"].append(formatted_module)
            
            structured_docs_formatted.append(formatted_doc)
        
        # Format for readability instead of using json.dumps directly
        structured_docs_str = ""
        for i, doc in enumerate(structured_docs_formatted):
            structured_docs_str += f"=== SDK DOCUMENTATION {i+1}: {doc['library']} v{doc['version']} ===\n"
            structured_docs_str += f"Description: {doc['description']}\n\n"
            
            # Format imports in a readable way
            structured_docs_str += "IMPORT STATEMENTS:\n"
            for imp in doc.get("imports", []):
                structured_docs_str += f"- {imp.get('statement', '')}\n"
            structured_docs_str += "\n"
            
            # Format modules
            structured_docs_str += "MODULES AND CLASSES:\n"
            for module in doc.get("modules", []):
                structured_docs_str += f"## {module.get('name', '')}\n"
                structured_docs_str += f"   {module.get('description', '')}\n\n"
                
                # Format methods
                for method in module.get("methods", []):
                    structured_docs_str += f"   * {method.get('name', '')}("
                    
                    # Format parameters
                    params = []
                    for param in method.get("parameters", []):
                        param_type = param.get("type", "Any")
                        params.append(f"{param.get('name', '')}: {param_type}")
                    
                    structured_docs_str += ", ".join(params)
                    structured_docs_str += f") -> {method.get('returns', {}).get('type', 'None')}\n"
                    structured_docs_str += f"     {method.get('description', '')}\n\n"
            
            structured_docs_str += "\n\n"
        
        update_progress(f"Prepared structured documentation for {len(structured_docs)} libraries")
        
        # Get the optimized LLM with task-specific parameters
        update_progress(f"Initializing {llm_provider} for code generation")
        llm = get_llm(llm_provider, task=task, sdk_name=sdk_name)
        
        if not llm:
            update_progress("Could not initialize the language model. Please check your API keys.", "error")
            return "Error: Could not initialize the language model. Please check your API keys."
        
        # Create a clearer code generation prompt that isolates the task from URL
        # and provides documentation in a structured way
        code_prompt = f"""
        You are an expert Python developer tasked with generating code based on SDK documentation.
        
        USER TASK: {task}
        
        DOCUMENTATION CHUNKS:
        {vector_chunks}
        
        STRUCTURED DOCUMENTATION:
        {structured_docs_str}
        
        Your task is to:
        1. Understand the SDK's structure and available classes/functions
        2. Pay special attention to the correct import statements from the documentation 
        3. Use specifically the modules and classes as shown in the documentation, not from your training data
        4. Generate complete, working code for the user's task
        5. Include clear comments explaining your implementation
        6. Verify that all functions and classes used are correctly imported
        7. Explicitly take import statements and correct use of modules from the documentation, not your training data
        
        IMPORTANT: Always format your code with a proper '''python (code) ''' block for easier parsing.
        
        YOUR SOLUTION (complete Python code):
        """
        
        update_progress("Sending request to the model. This may take a moment...")
        
        # Create a placeholder for streaming output
        code_container = st.empty()
        
        # Initialize the custom streaming handler
        streaming_handler = StreamlitCallbackHandler(code_container)
        
        # Generate solution based on model type
        solution = ""
        
        if isinstance(llm, ChatHuggingFace):
            code_messages = [
                SystemMessage(content="You are an expert Python developer tasked with generating code based on SDK documentation or API Reference Material."),
                HumanMessage(content=code_prompt)
            ]
            # Set up callbacks for streaming
            llm.callbacks = [streaming_handler]
            update_progress("Generating code with HuggingFace model...")
            ai_msg = llm.invoke(code_messages)
            solution = ai_msg.content
            
        elif isinstance(llm, Replicate):
            # For Replicate, we'll use a different approach to handle streaming responses
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with generating code based on SDK documentation or API Reference Material."),
                HumanMessage(content=code_prompt)
            ]
            
            # Construct a chat-like prompt for Replicate models
            formatted_prompt = "System: You are an expert Python developer tasked with generating code based on SDK documentation or API Reference Material.\n\nHuman: " + code_prompt + "\n\nAssistant:"
            
            # Use invoke and collect the full response
            llm.callbacks = [streaming_handler]
            update_progress("Generating code with Replicate model...")
            response = llm.invoke(formatted_prompt)
            solution = response  # The response is already the string content
            
        else:
            # For OpenAI and Anthropic
            code_messages = [
                SystemMessage(content="You are an expert Python developer tasked with generating code based on SDK documentation or API Reference Material."),
                HumanMessage(content=code_prompt)
            ]
            # Enable streaming
            llm.callbacks = [streaming_handler]
            update_progress(f"Generating code with {llm_provider}...")
            response = llm.invoke(code_messages)
            solution = response.content
        
        # Extract clean code from solution
        final_code = clean_code_from_timestamps(solution)
        
        # Verify code quality
        issues = verify_code_quality(final_code)
        if issues:
            update_progress("Code generated, fixing formatting issues...", "warning")
            # Log issues for debugging
            for issue in issues:
                update_progress(f"Fixed: {issue}", "info")
            
            # Request the model to fix the issues
            fix_prompt = f"""
            The generated code has the following issues that need to be fixed:
            {chr(10).join(issues)}
            
            Please fix these issues while maintaining the same functionality.
            Original code:
            {final_code}
            
            Provide the corrected code with proper formatting, indentation, and consistent style.
            """
            
            # Get fixed version from the model
            messages = [
                SystemMessage(content="You are a Python code formatting expert."),
                HumanMessage(content=fix_prompt)
            ]
            
            fixed_response = llm.invoke(messages)
            final_code = clean_code_from_timestamps(fixed_response.content)
            
            # Verify fixes
            remaining_issues = verify_code_quality(final_code)
            if remaining_issues:
                update_progress("Some minor formatting issues remain, but code is functional", "warning")
            else:
                update_progress("Code formatting issues resolved successfully!")
        
        # Check if the solution is wrapped in a code block
        if not (final_code.startswith("'''python") or final_code.startswith("'''")):
            final_code = f"'''python\n{final_code}\n'''"
        
        # Update status
        update_progress("Code solution generated successfully!")
        
        return final_code
    
    except Exception as e:
        update_progress(f"Error generating solution: {e}", "error")
        return f"Error generating solution: {e}"

# Function to extract URLs from text
def extract_urls(text):
    # Regular expression pattern to find URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)

# Function to extract task from text
def extract_task(text, urls):
    # Remove URLs from text to get the task
    task_text = text
    for url in urls:
        task_text = task_text.replace(url, "")
    
    # Clean up the task text
    task_text = re.sub(r'\s+', ' ', task_text).strip()
    
    # Remove common prefixes like "Using the SDK" or "build a" if present
    task_text = re.sub(r'^(?:Using|using)\s+(?:the|latest)?\s*(?:documentation|SDK|Python SDK|API)(?:\s+at)?\s*,?\s*', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'^(?:build|create|implement|develop|code)(?:\s+a|\s+an)?\s*', '', task_text, flags=re.IGNORECASE)
    
    # Handle paths that were mistakenly added to the task
    task_text = re.sub(r'^/[\w-]+/[\w-]+,?\s*', '', task_text)
    
    return task_text.strip()

# Function to process a user request
def process_request(request):
    progress_placeholder = st.empty()
    
    def update_display_progress():
        # Display the current progress status
        progress_placeholder.info(st.session_state.progress_status)
        
    # Clear previous progress details
    st.session_state.progress_details = []
    
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        update_progress("Connecting to MongoDB database...")
        client = get_mongodb_connection()
        if not client:
            update_progress("Failed to connect to the database. Please try again.", "error")
            update_display_progress()
            return "Failed to connect to the database. Please try again."
    
    # Extract URLs from the request
    urls = extract_urls(request)
    
    if not urls:
        # Added proper URL extraction for paths like /develop/api-reference
        path_matches = re.findall(r'/[\w-]+/[\w-]+', request)
        if path_matches:
            update_progress(f"Found path reference: {path_matches[0]}, attempting to convert to full URL")
            # Attempt to construct a reasonable URL from the path
            if "langchain" in path_matches[0].lower():
                urls = ["https://python.langchain.com" + path_matches[0]]
                update_progress(f"Constructed URL: {urls[0]}")
            elif "react" in path_matches[0].lower():
                urls = ["https://react.dev" + path_matches[0]]
                update_progress(f"Constructed URL: {urls[0]}")
            else:
                # Generic domain construction
                urls = ["https://docs.example.com" + path_matches[0]]
                update_progress(f"Constructed URL: {urls[0]}")
    
    if not urls:
        update_progress("Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation.", "error")
        update_display_progress()
        return "Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation."
    
    # Extract task from the request
    task = extract_task(request, urls)
    
    if not task:
        update_progress("Could not identify the task in your request. Please describe what you want to build.", "error")
        update_display_progress()
        return "Could not identify the task in your request. Please describe what you want to build."
    
    update_progress(f"Processing task: '{task}' with {len(urls)} documentation URLs")
    update_display_progress()
    
    # Process each URL
    all_vector_results = []
    all_structured_docs = []
    
    for i, url in enumerate(urls):
        # Extract library name from URL if possible
        library_match = re.search(r'//([^/]+\.)?([^./]+)\.(io|com|org)', url)
        library = library_match.group(2) if library_match else None
        
        # Check if we already have documentation for this library
        update_progress(f"[URL {i+1}/{len(urls)}] Checking for existing documentation for {library if library else 'unknown library'}...")
        update_display_progress()
        
        vector_results, structured_docs = search_documentation(client, task, library)
        
        # If we don't have enough relevant results, scrape the documentation
        if len(vector_results) < 3 or not structured_docs:
            update_progress(f"[URL {i+1}/{len(urls)}] Insufficient existing documentation. Getting new documentation from {url}...")
            update_display_progress()
            
            # Use FireCrawl or fallback to BeautifulSoup
            documentation, vector_docs = scrape_documentation(url)
            update_display_progress()  # Update with the latest status
            
            if documentation and vector_docs:
                update_progress(f"[URL {i+1}/{len(urls)}] Storing documentation from {url}...")
                update_display_progress()
                
                store_result = store_documentation(client, documentation, vector_docs)
                
                if store_result:
                    update_progress(f"[URL {i+1}/{len(urls)}] Documentation from {url} processed and stored successfully!")
                else:
                    update_progress(f"[URL {i+1}/{len(urls)}] Documentation from {url} was processed but could not be stored completely.", "warning")
                update_display_progress()
                
                # Search again with the new documentation
                update_progress(f"[URL {i+1}/{len(urls)}] Searching newly stored documentation for '{task}'...")
                update_display_progress()
                
                vector_results, structured_docs = search_documentation(client, task, library)
        
        # Add results to our collections
        all_vector_results.extend(vector_results)
        all_structured_docs.extend(structured_docs)
        
        update_progress(f"[URL {i+1}/{len(urls)}] Successfully processed documentation for {library if library else 'unknown library'}")
        update_display_progress()
    
    # Generate the code solution
    update_progress(f"All documentation processed. Generating code solution for: {task}")
    update_display_progress()
    
    solution = generate_code_solution(task, all_vector_results, all_structured_docs)
    
    if client:
        client.close()
    
    update_progress("Task completed!")
    update_display_progress()
    
    return solution

# Function to handle feedback and corrections
def process_feedback(feedback, original_solution):
    # Clear previous progress details
    st.session_state.progress_details = []
    progress_placeholder = st.empty()
    
    def update_display_progress():
        # Display the current progress status
        progress_placeholder.info(st.session_state.progress_status)
    
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        update_progress("Connecting to MongoDB database...")
        client = get_mongodb_connection()
        if not client:
            update_progress("Failed to connect to the database. Please try again.", "error")
            update_display_progress()
            return "Failed to connect to the database. Please try again."
    
    # Re-search documentation based on the feedback
    update_progress(f"Processing feedback: '{feedback}'...")
    update_display_progress()
    
    vector_results, structured_docs = search_documentation(client, feedback)
    
    # Extract library name from structured docs for parameter optimization
    sdk_name = "Unknown"
    if structured_docs and len(structured_docs) > 0:
        if "library" in structured_docs[0]:
            sdk_name = structured_docs[0]["library"]
    
    # Get the selected model with optimal parameters
    update_progress(f"Initializing {llm_provider} for feedback processing...")
    update_display_progress()
    
    llm = get_llm(llm_provider, task=feedback, sdk_name=sdk_name)
    
    if not llm:
        update_progress("Could not initialize the language model. Please check your API keys.", "error")
        update_display_progress()
        return "Error: Could not initialize the language model. Please check your API keys."
    
    # Improve documentation formatting for feedback
    vector_chunks = ""
    for i, doc in enumerate(vector_results):
        vector_chunks += f"--- DOCUMENTATION CHUNK {i+1} ---\n"
        vector_chunks += f"SOURCE: {doc.metadata.get('source', 'Unknown')}\n"
        vector_chunks += f"{doc.page_content}\n\n"
    
    # Format structured documentation in a more readable way
    structured_docs_str = ""
    for i, doc in enumerate(structured_docs):
        structured_docs_str += f"=== SDK DOCUMENTATION {i+1}: {doc.get('library', 'Unknown')} v{doc.get('version', 'Latest')} ===\n"
        structured_docs_str += f"Description: {doc.get('description', '')}\n\n"
        
        # Format imports in a readable way
        structured_docs_str += "IMPORT STATEMENTS:\n"
        for imp in doc.get("imports", []):
            structured_docs_str += f"- {imp.get('statement', '')}\n"
        structured_docs_str += "\n"
        
        # Format modules
        structured_docs_str += "MODULES AND CLASSES:\n"
        for module in doc.get("modules", []):
            structured_docs_str += f"## {module.get('name', '')}\n"
            structured_docs_str += f"   {module.get('description', '')}\n\n"
            
            # Format methods
            for method in module.get("methods", []):
                structured_docs_str += f"   * {method.get('name', '')}("
                
                # Format parameters
                params = []
                for param in method.get("parameters", []):
                    param_type = param.get("type", "Any")
                    params.append(f"{param.get('name', '')}: {param_type}")
                
                structured_docs_str += ", ".join(params)
                structured_docs_str += f") -> {method.get('returns', {}).get('type', 'None')}\n"
                structured_docs_str += f"     {method.get('description', '')}\n\n"
        
        structured_docs_str += "\n\n"
    
    # Create a placeholder for streaming output
    feedback_container = st.empty()
    
    # Initialize the custom streaming handler
    streaming_handler = StreamlitCallbackHandler(feedback_container)
    
    # Create feedback processing prompt
    feedback_prompt = f"""
    You are an expert Python developer tasked with improving code based on user feedback.
    
    ORIGINAL CODE:
    {original_solution}
    
    USER FEEDBACK:
    {feedback}
    
    DOCUMENTATION CHUNKS:
    {vector_chunks}
    
    STRUCTURED DOCUMENTATION:
    {structured_docs_str}
    
    Your task is to:
    1. Understand the user's feedback
    2. Check the documentation or API Reference Material to verify the correct usage
    3. Fix any issues in the original code
    4. Ensure all imports and API usages are correct according to the documentation
    5. Provide the improved solution
    
    IMPORTANT: Always format your code with a proper '''python (code) ''' block for easier parsing.
    """
    
    # Handle different model types for feedback processing
    improved_solution = ""
    
    try:
        update_progress("Generating improved solution based on feedback...")
        update_display_progress()
        
        if isinstance(llm, ChatHuggingFace):
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with improving code based on user feedback."),
                HumanMessage(content=feedback_prompt)
            ]
            # Set up callbacks for streaming
            llm.callbacks = [streaming_handler]
            ai_msg = llm.invoke(messages)
            improved_solution = ai_msg.content
            
        elif isinstance(llm, Replicate):
            # For Replicate, use the same approach as in generate_code_solution
            formatted_feedback_prompt = "System: You are an expert Python developer tasked with improving code based on user feedback.\n\nHuman: " + feedback_prompt + "\n\nAssistant:"
            
            # Set up callbacks for streaming
            llm.callbacks = [streaming_handler]
            improved_solution = llm.invoke(formatted_feedback_prompt)
            
        else:
            # For OpenAI and Anthropic
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with improving code based on user feedback."),
                HumanMessage(content=feedback_prompt)
            ]
            # Enable streaming
            llm.callbacks = [streaming_handler]
            response = llm.invoke(messages)
            improved_solution = response.content
        
        # Clean and format the improved solution
        final_code = clean_code_from_timestamps(improved_solution)
        
        # Check if the solution is wrapped in a code block
        if not (final_code.startswith("'''python") or final_code.startswith("'''")):
            # Wrap it in a code block for consistency
            final_code = f"'''python\n{final_code}\n'''"
        
        update_progress("Feedback processed and solution improved successfully!")
        update_display_progress()
        
        if client:
            client.close()
        
        return final_code
        
    except Exception as e:
        update_progress(f"Error processing feedback: {e}", "error")
        update_display_progress()
        
        if client:
            client.close()
        return f"Error processing feedback: {e}"

# Display progress details
if st.session_state.progress_details:
    with st.expander("Progress Log", expanded=False):
        for detail in st.session_state.progress_details:
            if detail["level"] == "error":
                st.error(f"{detail['time']} - {detail['message']}")
            elif detail["level"] == "warning":
                st.warning(f"{detail['time']} - {detail['message']}")
            else:
                st.info(f"{detail['time']} - {detail['message']}")

# Chat message display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display progress status if exists
if st.session_state.progress_status:
    st.info(st.session_state.progress_status)

# Display code solution if available
if st.session_state.code_solution:
    with st.expander("Generated Code Solution", expanded=True):
        # Extract code from markdown code blocks if present
        code_pattern = r'''(?:python|py)(.*?)'''
        code_matches = re.findall(code_pattern, st.session_state.code_solution, re.DOTALL)
        
        if code_matches:
            # Fix: Properly format code with proper line breaks
            code_content = code_matches[0].strip()
            # Use st.code to preserve formatting
            st.code(code_content, language="python")
        else:
            # If no code block markers, display as is
            clean_code = clean_code_from_timestamps(st.session_state.code_solution)
            st.code(clean_code, language="python")

# User input area
if prompt := st.chat_input("What would you like me to build?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Set processing flag
    st.session_state.processing = True
    
    # Check if this is feedback on an existing solution
    is_feedback = st.session_state.code_solution != ""
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if is_feedback:
            # Process feedback
            message_placeholder.markdown("I'm processing your feedback...")
            response = process_feedback(prompt, st.session_state.code_solution)
            st.session_state.code_solution = response
            message_placeholder.markdown("I've updated the solution based on your feedback. Check the code panel above.")
        else:
            # Process new request
            message_placeholder.markdown("I'm working on your request...")
            response = process_request(prompt)
            st.session_state.code_solution = response
            message_placeholder.markdown("I've generated a solution for your task. Check the code panel above.")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": 
        "I've updated the solution based on your feedback. Check the code panel above." if is_feedback 
        else "I've generated a solution for your task. Check the code panel above."
    })
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Force a rerun to update the UI with the new code
    st.rerun()

# Show database connection error if any
if "db_connection_error" in st.session_state and st.session_state.db_connection_error:
    st.error(f"Database connection issue: {st.session_state.db_connection_error}")

# Add a clear button to reset the conversation
if st.button("Start New Task"):
    st.session_state.messages = []
    st.session_state.code_solution = ""
    st.session_state.progress_status = ""
    st.session_state.progress_details = []
    st.rerun()
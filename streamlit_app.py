import streamlit as st
import os
import datetime
import json
import re
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
import getpass

# Set page config
st.set_page_config(page_title="LLM Python Solution Builder", page_icon="🧩", layout="wide")

# App title and description
st.title("SDK Code Generator")
st.markdown("""
This tool helps you generate Python code solutions, checking it against the latest SDK documentation. 
Simply provide your task along with the SDK URL(s) you want to use.
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
                                    value="meta/meta-llama-3-8b-instruct" if llm_provider == "Replicate Model" else "HuggingFaceH4/zephyr-7b-beta",
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
        "max_tokens": 4000,
        "repetition_penalty": 1.03
    }
    
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
            
        return ChatOpenAI(
            model=model_name, 
            model_kwargs=params, 
            api_key=st.session_state.openai_api_key,
            streaming=True  # Enable streaming for real-time updates
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
            api_token=st.session_state.replicate_api_token
        )
        
    elif provider == "HuggingFace Hub" and custom_model:
        if not st.session_state.huggingface_api_token:
            st.error("HuggingFace Access Token is required for HuggingFace Hub models. Please provide it in the sidebar.")
            return None
            
        # For HuggingFace, use specific parameter structure
        llm = HuggingFaceEndpoint(
            repo_id=custom_model,
            task="text-generation",
            max_new_tokens=params.get("max_tokens", 512),
            do_sample=params.get("temperature", 0.2) > 0,
            temperature=params.get("temperature", 0.2),
            top_p=params.get("top_p", 0.95),
            repetition_penalty=params.get("repetition_penalty", 1.03),
            token=st.session_state.huggingface_api_token,
            streaming=True  # Enable streaming for real-time updates
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

# MongoDB connection setup
def get_mongodb_connection():
    try:
        # Check if MongoDB URI is available
        vector_store_type = get_vector_store_type()
        if vector_store_type == "mongodb":
            # Use environment variables for production or secrets for local development
            try:
                mongo_uri = os.getenv("MONGODB_URI", st.secrets["mongodb"]["uri"])
            except:
                mongo_uri = os.getenv("MONGODB_URI")
                
            client = pymongo.MongoClient(mongo_uri, server_api=ServerApi('1'))
            
            # Ping the database to confirm connection
            client.admin.command('ping')
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

# Function to scrape the SDK documentation using Firecrawl or fallback to BeautifulSoup
def scrape_documentation(url):
    # Update progress status
    st.session_state.progress_status = f"Using Firecrawl to crawl {url}..."
    
    # Try to use Firecrawl if API key is available
    if st.session_state.firecrawl_api_key:
        try:
            loader = FireCrawlLoader(
                api_key=st.session_state.firecrawl_api_key,
                url=url,
                mode="crawl"  # Use crawl mode to get all accessible subpages
            )
            
            docs = loader.load()
            
            if docs:
                # Extract library name and version from the first document's metadata
                library_name = "Unknown"
                library_version = "Latest"
                
                if "title" in docs[0].metadata:
                    title_text = docs[0].metadata["title"]
                    match = re.search(r'([\w\-]+)', title_text)
                    if match:
                        library_name = match.group(1)
                
                # Create the documentation object
                documentation = {
                    "library": library_name,
                    "version": library_version,
                    "modules": [],
                    "imports": [],
                    "last_updated": datetime.datetime.now().isoformat()
                }
                
                return documentation, docs
            else:
                st.session_state.progress_status = f"Firecrawl returned no documents for {url}. Falling back to basic scraping."
                # Fall back to basic scraping
                return fallback_scrape_documentation(url)
                
        except Exception as e:
            st.session_state.progress_status = f"Error using Firecrawl: {e}. Falling back to basic scraping."
            # Fall back to basic scraping
            return fallback_scrape_documentation(url)
    else:
        # Fall back to basic scraping if no Firecrawl API key
        return fallback_scrape_documentation(url)

# Fallback scraping function using BeautifulSoup
def fallback_scrape_documentation(url):
    try:
        st.session_state.progress_status = f"Scraping documentation from {url} using BeautifulSoup..."
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text content
        content = soup.get_text()
        
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
        
        # Create the final documentation object
        documentation = {
            "library": library_name,
            "version": library_version,
            "modules": modules,
            "imports": imports,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Also create a list of text chunks for vector storage
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
                    "chunk_id": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        return documentation, docs
    
    except Exception as e:
        st.session_state.progress_status = f"Error scraping documentation: {e}"
        return None, None

# Function to store documentation (MongoDB or Chroma)
def store_documentation(client, documentation, vector_docs):
    try:
        st.session_state.progress_status = "Storing documentation in vector database..."
        vector_store_type = get_vector_store_type()
        
        # Get embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return False
            
        if vector_store_type == "mongodb":
            db = client['sdk_documentation']
            
            # Store the structured documentation
            doc_collection = db['structured_docs']
            
            # Check if we already have this library version
            existing = doc_collection.find_one({
                "library": documentation["library"],
                "version": documentation["version"]
            })
            
            if existing:
                # Update existing document
                doc_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": documentation}
                )
            else:
                # Insert new document
                doc_collection.insert_one(documentation)
            
            # Store vector documents
            vector_collection = db['vector_docs']
            
            # Create the vector store
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                vector_docs,
                embeddings,
                collection=vector_collection,
                index_name="vector_index",
            )
        else:
            # Use Chroma for local vector storage
            # Store structured documentation
            structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
            os.makedirs(structured_docs_path, exist_ok=True)
            
            # Generate a unique filename based on library and version
            filename = f"{documentation['library']}_{documentation['version']}.json"
            with open(os.path.join(structured_docs_path, filename), 'w') as f:
                json.dump(documentation, f)
            
            # Store vector documents in Chroma
            vector_store = Chroma.from_documents(
                documents=vector_docs,
                embedding=embeddings,
                persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store")
            )
            vector_store.persist()
        
        return True
    
    except Exception as e:
        st.session_state.progress_status = f"Error storing documentation: {e}"
        return False

# Function to search for documentation (MongoDB or Chroma)
def search_documentation(client, query, library=None):
    try:
        st.session_state.progress_status = f"Searching documentation for '{query}'..."
        vector_store_type = get_vector_store_type()
        
        # Get embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return [], []
            
        if vector_store_type == "mongodb":
            db = client['sdk_documentation']
            
            # Search in vector store
            vector_collection = db['vector_docs']
            
            # Create the vector store
            vector_store = MongoDBAtlasVectorSearch(
                collection=vector_collection,
                embedding=embeddings,
                index_name="vector_index",
            )
            
            # Add library filter if provided
            search_filter = {"library": library} if library else None
            
            # Search for similar documents
            results = vector_store.similarity_search(query, k=10, pre_filter=search_filter)
            
            # Also get the structured documentation
            structured_collection = db['structured_docs']
            
            # If we have a library, get its specific documentation
            structured_docs = list(structured_collection.find(
                {"library": library} if library else {}
            ))
        else:
            # Use Chroma for local vector search
            # Load vector store
            try:
                vector_store = Chroma(
                    persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store"),
                    embedding_function=embeddings
                )
                
                # Search for similar documents, filter by library if provided
                if library:
                    results = vector_store.similarity_search(
                        query=query,
                        k=10,
                        filter={"library": library}
                    )
                else:
                    results = vector_store.similarity_search(query, k=10)
                
                # Get structured documentation
                structured_docs = []
                structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
                
                if os.path.exists(structured_docs_path):
                    for filename in os.listdir(structured_docs_path):
                        if filename.endswith('.json'):
                            if library and not filename.startswith(f"{library}_"):
                                continue
                            
                            with open(os.path.join(structured_docs_path, filename), 'r') as f:
                                doc = json.load(f)
                                structured_docs.append(doc)
            except Exception as e:
                st.session_state.progress_status = f"Error loading Chroma vector store: {e}. Creating a new one."
                results = []
                structured_docs = []
        
        return results, structured_docs
    
    except Exception as e:
        st.session_state.progress_status = f"Error searching documentation: {e}"
        return [], []

# Function to clean code from timestamp markers
def clean_code_from_timestamps(code_text):
    # Remove timestamp markers like [2025-03-01 19:54:57.110174]
    pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\]'
    clean_text = re.sub(pattern, '', code_text)
    
    # Clean up extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Extract code blocks if present
    code_pattern = r'```python(.*?)```'
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
        if "```python" in token or "```py" in token:
            self.in_code_block = True
            self.code_detected = True
        elif self.in_code_block and "```" in token:
            self.in_code_block = False
        
        # If we're in a code block, collect the code
        if self.in_code_block:
            self.code_block += token
        
        # Update the Streamlit container
        # If code is detected, show it properly formatted
        if self.code_detected:
            # Try to extract the code block
            code_pattern = r'```(?:python|py)(.*?)```'
            code_matches = re.findall(code_pattern, self.text, re.DOTALL)
            
            if code_matches:
                # Show the explanation text above the code block
                explanation_text = self.text.split("```python")[0] if "```python" in self.text else self.text.split("```py")[0]
                self.container.markdown(explanation_text)
                
                # Show the code in a dedicated code block
                self.container.code(code_matches[-1].strip(), language="python")
                
                # Show any text that follows the code block
                if "```" in self.text:
                    after_code = self.text.split("```")[-1]
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
        code_pattern = r'```(?:python|py)(.*?)```'
        code_matches = re.findall(code_pattern, self.text, re.DOTALL)
        
        if code_matches:
            # Return the last code block
            return code_matches[-1].strip()
        
        # If no code blocks, clean the text and try to extract code
        clean_text = clean_code_from_timestamps(self.text)
        return clean_text

# Function to generate code solution using the selected model
def generate_code_solution(task, vector_results, structured_docs):
    try:
        st.session_state.progress_status = "Generating code solution..."
        
        # Extract library name from structured docs for parameter optimization
        sdk_name = "Unknown"
        if structured_docs and len(structured_docs) > 0:
            if "library" in structured_docs[0]:
                sdk_name = structured_docs[0]["library"]

        # Format the vector chunks
        vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
        
        # Format the structured docs
        structured_docs_str = json.dumps(structured_docs, indent=2)
        
        # Get the optimized LLM with task-specific parameters
        llm = get_llm(llm_provider, task=task, sdk_name=sdk_name)
        
        if not llm:
            return "Error: Could not initialize the language model. Please check your API keys."
        
        # Create the code generation prompt
        code_prompt = f"""
        You are an expert Python developer tasked with generating code based on SDK documentation or API Reference Material.
        
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
        
        IMPORTANT: Always format your code with a proper ```python (code) ``` block for easier parsing.
        
        YOUR SOLUTION (complete Python code):
        """
        
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
            response = llm.invoke(code_messages)
            solution = response.content
        
        # Extract clean code from solution
        final_code = clean_code_from_timestamps(solution)
        
        # Check if the solution is wrapped in a code block
        if not (final_code.startswith("```python") or final_code.startswith("```")):
            # Wrap it in a code block for consistency
            final_code = f"```python\n{final_code}\n```"
        
        # Update status
        st.session_state.progress_status = "Code solution generated successfully!"
        
        return final_code
    
    except Exception as e:
        st.session_state.progress_status = f"Error generating solution: {e}"
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
    task_text = re.sub(r'^(?:Using|using)\s+(?:the|latest)?\s*(?:SDK|Python SDK|API)(?:\s+at)?\s*,?\s*', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'^(?:build|create|implement|develop|code)(?:\s+a|\s+an)?\s*', '', task_text, flags=re.IGNORECASE)
    
    return task_text.strip()

# Function to process a user request
def process_request(request):
    progress_placeholder = st.empty()
    
    def update_progress():
        progress_placeholder.info(st.session_state.progress_status)
    
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        client = get_mongodb_connection()
        if not client:
            st.session_state.progress_status = "Failed to connect to the database. Please try again."
            update_progress()
            return "Failed to connect to the database. Please try again."
    
    # Extract URLs from the request
    urls = extract_urls(request)
    
    if not urls:
        st.session_state.progress_status = "Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation."
        update_progress()
        return "Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation."
    
    # Extract task from the request
    task = extract_task(request, urls)
    
    if not task:
        st.session_state.progress_status = "Could not identify the task in your request. Please describe what you want to build."
        update_progress()
        return "Could not identify the task in your request. Please describe what you want to build."
    
    # Process each URL
    all_vector_results = []
    all_structured_docs = []
    
    for url in urls:
        # Extract library name from URL if possible
        library_match = re.search(r'//([^/]+\.)?([^./]+)\.(io|com|org)', url)
        library = library_match.group(2) if library_match else None
        
        # Check if we already have documentation for this library
        st.session_state.progress_status = f"Checking for existing documentation for {library if library else 'unknown library'}..."
        update_progress()
        vector_results, structured_docs = search_documentation(client, task, library)
        
        # If we don't have enough relevant results, scrape the documentation
        if len(vector_results) < 3 or not structured_docs:
            st.session_state.progress_status = f"Getting SDK documentation from {url}..."
            update_progress()
            
            # Use FireCrawl or fallback to BeautifulSoup
            documentation, vector_docs = scrape_documentation(url)
            update_progress()  # Update with the latest status
            
            if documentation and vector_docs:
                st.session_state.progress_status = f"Storing documentation from {url}..."
                update_progress()
                store_result = store_documentation(client, documentation, vector_docs)
                
                if store_result:
                    st.session_state.progress_status = f"Documentation from {url} processed and stored successfully!"
                else:
                    st.session_state.progress_status = f"Documentation from {url} was processed but could not be stored completely."
                update_progress()
                
                # Search again with the new documentation
                st.session_state.progress_status = f"Searching documentation for '{task}'..."
                update_progress()
                vector_results, structured_docs = search_documentation(client, task, library)
        
        # Add results to our collections
        all_vector_results.extend(vector_results)
        all_structured_docs.extend(structured_docs)
    
    # Generate the code solution
    st.session_state.progress_status = "Generating code solution..."
    update_progress()
    solution = generate_code_solution(task, all_vector_results, all_structured_docs)
    
    if client:
        client.close()
    return solution

# Function to handle feedback and corrections
def process_feedback(feedback, original_solution):
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        client = get_mongodb_connection()
        if not client:
            return "Failed to connect to the database. Please try again."
    
    # Re-search documentation based on the feedback
    st.session_state.progress_status = f"Processing feedback: '{feedback}'..."
    vector_results, structured_docs = search_documentation(client, feedback)
    
    # Extract library name from structured docs for parameter optimization
    sdk_name = "Unknown"
    if structured_docs and len(structured_docs) > 0:
        if "library" in structured_docs[0]:
            sdk_name = structured_docs[0]["library"]
    
    # Get the selected model with optimal parameters
    llm = get_llm(llm_provider, task=feedback, sdk_name=sdk_name)
    
    if not llm:
        return "Error: Could not initialize the language model. Please check your API keys."
    
    # Format the vector chunks and structured docs
    vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
    structured_docs_str = json.dumps(structured_docs, indent=2)
    
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
    
    IMPORTANT: Always format your code with a proper ```python (code) ``` block for easier parsing.
    """
    
    # Handle different model types for feedback processing
    improved_solution = ""
    
    try:
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
        if not (final_code.startswith("```python") or final_code.startswith("```")):
            # Wrap it in a code block for consistency
            final_code = f"```python\n{final_code}\n```"
        
        if client:
            client.close()
        
        return final_code
        
    except Exception as e:
        st.session_state.progress_status = f"Error processing feedback: {e}"
        if client:
            client.close()
        return f"Error processing feedback: {e}"

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
        code_pattern = r'```(?:python|py)(.*?)```'
        code_matches = re.findall(code_pattern, st.session_state.code_solution, re.DOTALL)
        
        if code_matches:
            # Display the code part only
            st.code(code_matches[0].strip(), language="python")
        else:
            # If no code block markers, display as is
            st.code(clean_code_from_timestamps(st.session_state.code_solution), language="python")

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
    st.rerun()

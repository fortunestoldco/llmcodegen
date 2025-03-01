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

# Set page config with improved layout and theme
st.set_page_config(
    page_title="Other Tales CodeMaker",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/othertales/codemaker',
        'Report a bug': 'https://github.com/othertales/codemaker/issues',
        'About': "# Other Tales CodeMaker\nGenerate code solutions based on SDK documentation."
    }
)

# Apply custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .fixed-height-code {
        height: 400px;
        overflow: auto;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .task-input {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .status-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f7ff;
        margin-bottom: 1rem;
    }
    .log-timestamp {
        color: #757575;
        font-size: 0.8rem;
    }
    .log-message {
        margin-left: 0.5rem;
    }
    .token-counter {
        font-size: 0.8rem;
        color: #757575;
        text-align: right;
    }
    .model-selection {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .api-key-warning {
        color: #f57f17;
        font-size: 0.9rem;
        font-style: italic;
    }
    .progress-section {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    /* Improve tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f5f5f5;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App title and description with improved styling
st.markdown('<h1 class="main-header">Other Tales CodeMaker</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Generate and verify code solutions based on SDK documentation.</p>', unsafe_allow_html=True)

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
if "current_progress" not in st.session_state:
    st.session_state.current_progress = 0
if "documentation_sources" not in st.session_state:
    st.session_state.documentation_sources = []
if "last_url_processed" not in st.session_state:
    st.session_state.last_url_processed = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Code Solution"

# Sidebar for configuration with improved styling
with st.sidebar:
    st.sidebar.image("https://raw.githubusercontent.com/othertales/brand/main/logo/othertales-logo-light.png", width=200)
    
    with st.expander("About CodeMaker", expanded=False):
        st.markdown("""
        ### What is CodeMaker?
        
        CodeMaker helps you generate code based on SDK documentation. Simply provide:
        
        1. A clear description of what you want to build
        2. URLs to the relevant SDK documentation
        
        The tool will scrape the documentation, index it, and generate working code specifically tailored to your requirements.
        
        ### How it works
        
        1. **Documentation Processing**: When you provide SDK docs URLs, CodeMaker scrapes and indexes them
        2. **Context Retrieval**: Your task is analyzed to retrieve relevant parts of the documentation
        3. **Code Generation**: An AI model generates code that correctly implements your requirements based on the documentation
        4. **Feedback Loop**: You can provide feedback to refine the solution
        """)
    
    st.markdown("## Model Configuration")
    
    with st.container():
        st.markdown('<div class="model-selection">', unsafe_allow_html=True)
        llm_provider = st.selectbox(
            "Select AI Model",
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
            ],
            help="Choose the AI model to generate your code solution"
        )
        
        # Show model name input field only for Replicate and HuggingFace
        custom_model = None
        if llm_provider in ["Replicate Model", "HuggingFace Hub"]:
            custom_model = st.text_input(
                "Model Identifier", 
                value="meta/meta-llama-3-8b-instruct" if llm_provider == "Replicate Model" else "HuggingFaceH4/zephyr-7b-beta",
                key="custom_model",
                help="Enter the specific model identifier"
            )
            
            # Add recommended models for reference
            if llm_provider == "Replicate Model":
                st.caption("Recommended: meta/meta-llama-3-8b-instruct, mistralai/mixtral-8x7b-instruct-v0.1")
            else:
                st.caption("Recommended: HuggingFaceH4/zephyr-7b-beta, mistralai/Mistral-7B-Instruct-v0.2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Credentials section with improved organization and security warnings
    with st.expander("API Credentials", expanded=False):
        st.info("API keys are stored in session state and not persisted after you close the browser.")
        
        # OpenAI API Key
        st.subheader("OpenAI")
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for OpenAI models and embeddings",
            key="openai_key_input"
        )
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ OpenAI API Key set", icon="✅")
        elif "OpenAI" in llm_provider:
            st.warning("⚠️ OpenAI API Key required for selected model", icon="⚠️")
        
        st.divider()
        
        # Anthropic API Key
        st.subheader("Anthropic")
        anthropic_api_key = st.text_input(
            "Anthropic API Key", 
            value=st.session_state.anthropic_api_key,
            type="password",
            help="Required for Anthropic Claude models",
            key="anthropic_key_input"
        )
        if anthropic_api_key:
            st.session_state.anthropic_api_key = anthropic_api_key
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            st.success("✅ Anthropic API Key set", icon="✅")
        elif "Anthropic" in llm_provider:
            st.warning("⚠️ Anthropic API Key required for selected model", icon="⚠️")
        
        st.divider()
        
        # Replicate API Token
        st.subheader("Replicate")
        replicate_api_token = st.text_input(
            "Replicate Access Token", 
            value=st.session_state.replicate_api_token,
            type="password",
            help="Required for Replicate models",
            key="replicate_token_input"
        )
        if replicate_api_token:
            st.session_state.replicate_api_token = replicate_api_token
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
            st.success("✅ Replicate API Token set", icon="✅")
        elif llm_provider == "Replicate Model":
            st.warning("⚠️ Replicate API Token required for selected model", icon="⚠️")
        
        st.divider()
        
        # HuggingFace API Token
        st.subheader("HuggingFace")
        huggingface_api_token = st.text_input(
            "HuggingFace Access Token", 
            value=st.session_state.huggingface_api_token,
            type="password",
            help="Required for HuggingFace models",
            key="huggingface_token_input"
        )
        if huggingface_api_token:
            st.session_state.huggingface_api_token = huggingface_api_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
            st.success("✅ HuggingFace API Token set", icon="✅")
        elif llm_provider == "HuggingFace Hub":
            st.warning("⚠️ HuggingFace API Token required for selected model", icon="⚠️")
        
        st.divider()
        
        # Firecrawl API Key
        st.subheader("Firecrawl (Optional)")
        firecrawl_api_key = st.text_input(
            "Firecrawl API Key", 
            value=st.session_state.firecrawl_api_key,
            type="password",
            help="Enhanced web crawling capabilities (optional)",
            key="firecrawl_key_input"
        )
        if firecrawl_api_key:
            st.session_state.firecrawl_api_key = firecrawl_api_key
            os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key
            st.success("✅ Firecrawl API Key set", icon="✅")
        
    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        # Database configuration
        st.subheader("Database Settings")
        
        vector_store_type = get_vector_store_type()
        if vector_store_type == "mongodb":
            st.success("Using MongoDB Atlas for vector storage (configured via environment variables)")
        else:
            st.info("Using local Chroma DB for vector storage")
            
            # Custom Chroma DB path
            chroma_db_path = st.text_input(
                "Chroma DB Path",
                value=st.session_state.chroma_db_path,
                help="Local directory to store vector embeddings"
            )
            if chroma_db_path != st.session_state.chroma_db_path:
                st.session_state.chroma_db_path = chroma_db_path
                if not os.path.exists(chroma_db_path):
                    try:
                        os.makedirs(chroma_db_path)
                        st.success(f"Created directory: {chroma_db_path}")
                    except Exception as e:
                        st.error(f"Error creating directory: {e}")
        
        # Model parameters
        st.subheader("Model Parameters")
        
        default_temp = 0.2
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=default_temp,
            step=0.05,
            help="Higher values make output more creative, lower values more deterministic"
        )
        
        st.caption("Note: For specific SDKs, optimal parameters may be determined automatically.")
        
        # Crawling settings
        st.subheader("Documentation Crawling")
        
        crawl_depth = st.number_input(
            "Max crawl depth",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum recursion level when crawling documentation"
        )
        
        # Reset data button
        st.subheader("Reset Data")
        if st.button("Clear Cached Documentation", help="Delete all stored documentation from local storage"):
            try:
                if os.path.exists(st.session_state.chroma_db_path):
                    import shutil
                    shutil.rmtree(st.session_state.chroma_db_path)
                    os.makedirs(st.session_state.chroma_db_path)
                    st.success("Documentation cache cleared successfully!")
                else:
                    st.info("No cached documentation to clear.")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
    
    # Links and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Links")
    col1, col2 = st.sidebar.columns(2)
    col1.link_button("GitHub Repo", "https://github.com/othertales/codemaker")
    col2.link_button("Documentation", "https://docs.othertales.com/codemaker")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 Other Tales. All rights reserved.")

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

# Function to update progress with detailed information
def update_progress(message, level="info", progress_value=None):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.progress_status = message
    st.session_state.progress_details.append({"time": timestamp, "message": message, "level": level})
    
    # Update progress bar if a value is provided
    if progress_value is not None:
        st.session_state.current_progress = progress_value

# Function to scrape the SDK documentation using Firecrawl or fallback to BeautifulSoup
def scrape_documentation(url):
    # Update progress status
    update_progress(f"Starting documentation crawl for {url}...", progress_value=10)
    st.session_state.last_url_processed = url
    
    # Try to use Firecrawl if API key is available
    if st.session_state.firecrawl_api_key:
        try:
            update_progress(f"Using Firecrawl to crawl {url}...", progress_value=15)
            
            loader = FireCrawlLoader(
                api_key=st.session_state.firecrawl_api_key,
                url=url,
                mode="crawl"  # Use crawl mode to get all accessible subpages
            )
            
            update_progress(f"Firecrawl initiated for {url}. Retrieving content...", progress_value=25)
            docs = loader.load()
            
            if docs:
                update_progress(f"Successfully retrieved {len(docs)} documents from Firecrawl", progress_value=40)
                
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
                
                # Add to documentation sources
                if url not in st.session_state.documentation_sources:
                    st.session_state.documentation_sources.append({
                        "url": url,
                        "library": library_name,
                        "version": library_version,
                        "document_count": len(docs),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                return documentation, docs
            else:
                update_progress(f"Firecrawl returned no documents for {url}. Falling back to basic scraping.", "warning", progress_value=20)
                # Fall back to basic scraping
                return fallback_scrape_documentation(url)
                
        except Exception as e:
            update_progress(f"Error using Firecrawl: {e}. Falling back to basic scraping.", "error", progress_value=20)
            # Fall back to basic scraping
            return fallback_scrape_documentation(url)
    else:
        # Fall back to basic scraping if no Firecrawl API key
        return fallback_scrape_documentation(url)

# Fallback scraping function using BeautifulSoup
def fallback_scrape_documentation(url):
    try:
        update_progress(f"Scraping documentation from {url} using BeautifulSoup...", progress_value=15)
        response = requests.get(url)
        response.raise_for_status()
        
        update_progress(f"Successfully retrieved page content from {url}", progress_value=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text content
        content = soup.get_text()
        update_progress(f"Extracted text content from {url}", progress_value=25)
        
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
        
        update_progress(f"Looking for import statements and modules in {url}", progress_value=30)
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
        update_progress(f"Parsing document structure from {url}", progress_value=35)
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
        
        update_progress(f"Found {len(modules)} modules and {len(imports)} import statements", progress_value=40)
        
        # Create the final documentation object
        documentation = {
            "library": library_name,
            "version": library_version,
            "modules": modules,
            "imports": imports,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Also create a list of text chunks for vector storage
        update_progress(f"Splitting content into chunks for vector storage", progress_value=45)
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
        
        update_progress(f"Created {len(docs)} document chunks from {url}", progress_value=50)
        
        # Add to documentation sources
        if url not in st.session_state.documentation_sources:
            st.session_state.documentation_sources.append({
                "url": url,
                "library": library_name,
                "version": library_version,
                "document_count": len(docs),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        return documentation, docs
    
    except Exception as e:
        update_progress(f"Error scraping documentation: {e}", "error", progress_value=20)
        return None, None

# Function to store documentation (MongoDB or Chroma)
def store_documentation(client, documentation, vector_docs):
    try:
        update_progress("Storing documentation in vector database...", progress_value=55)
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
                update_progress(f"Updating existing documentation for {documentation['library']}", progress_value=60)
                doc_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": documentation}
                )
            else:
                # Insert new document
                update_progress(f"Inserting new documentation for {documentation['library']}", progress_value=60)
                doc_collection.insert_one(documentation)
            
            # Store vector documents
            vector_collection = db['vector_docs']
            
            # Create the vector store
            update_progress(f"Creating vector embeddings for {len(vector_docs)} documents", progress_value=65)
            
            # Show embedding progress
            progress_bar = st.progress(0)
            total_docs = len(vector_docs)
            
            # Process in batches for visual feedback
            batch_size = min(100, max(10, total_docs // 10))  # Adjust batch size based on document count
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch = vector_docs[i:end_idx]
                
                # Create vector embeddings for this batch
                vector_store = MongoDBAtlasVectorSearch.from_documents(
                    batch,
                    embeddings,
                    collection=vector_collection,
                    index_name="vector_index",
                )
                
                # Update progress bar
                progress = (end_idx / total_docs)
                progress_bar.progress(progress)
                update_progress(f"Embedded {end_idx}/{total_docs} documents", progress_value=65 + int(25 * progress))
                
            # Ensure progress bar reaches 100%
            progress_bar.progress(1.0)
            
        else:
            # Use Chroma for local vector storage
            # Store structured documentation
            structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
            os.makedirs(structured_docs_path, exist_ok=True)
            
            # Generate a unique filename based on library and version
            update_progress(f"Storing structured documentation for {documentation['library']}", progress_value=60)
            filename = f"{documentation['library']}_{documentation['version']}.json"
            with open(os.path.join(structured_docs_path, filename), 'w') as f:
                json.dump(documentation, f)
            
            # Store vector documents in Chroma
            update_progress(f"Creating vector embeddings with Chroma for {len(vector_docs)} documents", progress_value=65)
            
            # Show embedding progress
            progress_bar = st.progress(0)
            total_docs = len(vector_docs)
            
            # Process in smaller batches for better visual feedback
            batch_size = min(50, max(5, total_docs // 20))  # Smaller batches for Chroma
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch = vector_docs[i:end_idx]
                
                # Create vector embeddings for this batch
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store")
                )
                vector_store.persist()
                
                # Update progress bar
                progress = (end_idx / total_docs)
                progress_bar.progress(progress)
                update_progress(f"Embedded {end_idx}/{total_docs} documents", progress_value=65 + int(25 * progress))
            
            # Ensure progress bar reaches 100%
            progress_bar.progress(1.0)
        
        update_progress("Documentation stored successfully!", progress_value=90)
        return True
    
    except Exception as e:
        update_progress(f"Error storing documentation: {e}", "error", progress_value=60)
        return False

# Function to search for documentation (MongoDB or Chroma)
def search_documentation(client, query, library=None):
    try:
        update_progress(f"Searching documentation for '{query}'...", progress_value=70)
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
            update_progress(f"Found {len(results)} relevant document chunks", progress_value=80)
            
            # Also get the structured documentation
            structured_collection = db['structured_docs']
            
            # If we have a library, get its specific documentation
            structured_docs = list(structured_collection.find(
                {"library": library} if library else {}
            ))
            update_progress(f"Found {len(structured_docs)} structured documentation entries", progress_value=85)
        else:
            # Use Chroma for local vector search
            # Load vector store
            try:
                update_progress("Loading local Chroma vector store", progress_value=75)
                vector_store = Chroma(
                    persist_directory=os.path.join(st.session_state.chroma_db_path, "vector_store"),
                    embedding_function=embeddings
                )
                
                # Search for similar documents, filter by library if provided
                if library:
                    update_progress(f"Searching for '{query}' with library filter: {library}", progress_value=80)
                    results = vector_store.similarity_search(
                        query=query,
                        k=10,
                        filter={"library": library}
                    )
                else:
                    update_progress(f"Searching for '{query}' across all libraries", progress_value=80)
                    results = vector_store.similarity_search(query, k=10)
                update_progress(f"Found {len(results)} relevant document chunks", progress_value=85)
                
                # Get structured documentation
                structured_docs = []
                structured_docs_path = os.path.join(st.session_state.chroma_db_path, "structured_docs")
                
                if os.path.exists(structured_docs_path):
                    update_progress("Loading structured documentation from local files", progress_value=87)
                    for filename in os.listdir(structured_docs_path):
                        if filename.endswith('.json'):
                            if library and not filename.startswith(f"{library}_"):
                                continue
                            
                            with open(os.path.join(structured_docs_path, filename), 'r') as f:
                                doc = json.load(f)
                                structured_docs.append(doc)
                    update_progress(f"Found {len(structured_docs)} structured documentation entries", progress_value=90)
            except Exception as e:
                update_progress(f"Error loading Chroma vector store: {e}. Creating a new one.", "warning", progress_value=70)
                results = []
                structured_docs = []
        
        return results, structured_docs
    
    except Exception as e:
        update_progress(f"Error searching documentation: {e}", "error", progress_value=70)
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
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update the UI every 0.1 seconds
        
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
        
        # Update the Streamlit container at regular intervals to avoid overwhelming the UI
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._update_ui()
            self.last_update_time = current_time
        
    def _update_ui(self):
        # If code is detected, show it properly formatted
        if self.code_detected:
            # Try to extract the code block
            code_pattern = r'```(?:python|py)(.*?)```'
            code_matches = re.findall(code_pattern, self.text, re.DOTALL)
            
            if code_matches:
                # Show the explanation text above the code block
                explanation_text = self.text.split("```python")[0] if "```python" in self.text else self.text.split("```py")[0]
                self.container.markdown(explanation_text)
                
                # Show the code in a dedicated code block with syntax highlighting
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
        # Make sure the final update is displayed
        self._update_ui()
        
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
        update_progress("Generating code solution...", progress_value=92)
        
        # Extract library name from structured docs for parameter optimization
        sdk_name = "Unknown"
        if structured_docs and len(structured_docs) > 0:
            if "library" in structured_docs[0]:
                sdk_name = structured_docs[0]["library"]
                update_progress(f"Optimizing model parameters for {sdk_name} SDK", progress_value=93)

        # Format the vector chunks
        vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
        update_progress(f"Prepared {len(vector_results)} documentation chunks for context", progress_value=94)
        
        # Format the structured docs
        structured_docs_str = json.dumps(structured_docs, indent=2)
        update_progress(f"Prepared structured documentation for {len(structured_docs)} libraries", progress_value=95)
        
        # Get the optimized LLM with task-specific parameters
        update_progress(f"Initializing {llm_provider} for code generation", progress_value=96)
        llm = get_llm(llm_provider, task=task, sdk_name=sdk_name)
        
        if not llm:
            update_progress("Could not initialize the language model. Please check your API keys.", "error", progress_value=92)
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
        
        update_progress("Sending request to the model. This may take a moment...", progress_value=97)
        
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
            update_progress("Generating code with HuggingFace model...", progress_value=98)
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
            update_progress("Generating code with Replicate model...", progress_value=98)
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
            update_progress(f"Generating code with {llm_provider}...", progress_value=98)
            response = llm.invoke(code_messages)
            solution = response.content
        
        # Extract clean code from solution
        final_code = clean_code_from_timestamps(solution)
        
        # Check if the solution is wrapped in a code block
        if not (final_code.startswith("```python") or final_code.startswith("```")):
            # Wrap it in a code block for consistency
            final_code = f"```python\n{final_code}\n```"
        
        # Update status
        update_progress("Code solution generated successfully!", progress_value=100)
        
        return final_code
    
    except Exception as e:
        update_progress(f"Error generating solution: {e}", "error", progress_value=92)
        return f"Error generating solution: {e}"

# Function to extract URLs from text
def extract_urls(text):
    # Regular expression pattern to find URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)

# Function to extract task from text
def extract_task(text, urls):
    # This is the key function that needs fixing
    # Remove URLs from text to get the task
    task_text = text
    for url in urls:
        task_text = task_text.replace(url, "")
    
    # Clean up the task text
    task_text = re.sub(r'\s+', ' ', task_text).strip()
    
    # Remove common prefixes like "Using the SDK" or "build a" if present
    task_text = re.sub(r'^(?:Using|using)\s+(?:the|latest)?\s*(?:documentation|SDK|Python SDK|API)(?:\s+at)?\s*,?\s*', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'^(?:build|create|implement|develop|code)(?:\s+a|\s+an)?\s*', '', task_text, flags=re.IGNORECASE)
    
    # If the task still contains "create a basic RAG agent graph" or similar, that's what we want
    return task_text.strip()

# Function to process a user request
def process_request(request):
    # Create a card for the progress section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Processing Status")
        
        # Create columns for structured display
        status_col, progress_col = st.columns([3, 1])
        
        with status_col:
            progress_status = st.empty()
        
        with progress_col:
            progress_bar = st.progress(0)
            progress_percentage = st.empty()
        
        progress_details_container = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Function to update the progress display
    def update_display_progress():
        # Update the progress bar
        progress_bar.progress(st.session_state.current_progress / 100)
        progress_percentage.markdown(f"**{st.session_state.current_progress}%**")
        
        # Update the status message
        progress_status.info(st.session_state.progress_status)
        
        # Show recent progress details
        if st.session_state.progress_details:
            details_md = ""
            # Show the last 5 progress details
            for detail in st.session_state.progress_details[-5:]:
                if detail["level"] == "error":
                    icon = "❌"
                    color = "#f44336"
                elif detail["level"] == "warning":
                    icon = "⚠️"
                    color = "#ff9800"
                else:
                    icon = "ℹ️"
                    color = "#2196f3"
                
                details_md += f"<span class='log-timestamp'>{detail['time']}</span> {icon} <span style='color:{color}'>{detail['message']}</span><br>"
            
            progress_details_container.markdown(details_md, unsafe_allow_html=True)
    
    # Reset progress
    st.session_state.current_progress = 0
    st.session_state.progress_details = []
    update_display_progress()
    
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        update_progress("Connecting to MongoDB database...", progress_value=5)
        client = get_mongodb_connection()
        if not client:
            update_progress("Failed to connect to the database. Please try again.", "error", progress_value=5)
            update_display_progress()
            return "Failed to connect to the database. Please try again."
    
    # Extract URLs from the request
    urls = extract_urls(request)
    
    if not urls:
        update_progress("Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation.", "error", progress_value=5)
        update_display_progress()
        return "Could not identify any SDK URLs in your request. Please include at least one URL to the SDK documentation."
    
    # Extract task from the request
    task = extract_task(request, urls)
    
    if not task:
        update_progress("Could not identify the task in your request. Please describe what you want to build.", "error", progress_value=5)
        update_display_progress()
        return "Could not identify the task in your request. Please describe what you want to build."
    
    update_progress(f"Processing task: '{task}' with {len(urls)} documentation URLs", progress_value=10)
    update_display_progress()
    
    # Process each URL
    all_vector_results = []
    all_structured_docs = []
    
    # Create a progress tracker for URLs
    url_progress_start = 10
    url_progress_end = 60
    url_progress_per_url = (url_progress_end - url_progress_start) / len(urls)
    
    for i, url in enumerate(urls):
        # Calculate progress value for this URL
        current_url_progress_start = url_progress_start + (i * url_progress_per_url)
        current_url_progress_end = current_url_progress_start + url_progress_per_url
        
        # Extract library name from URL if possible
        library_match = re.search(r'//([^/]+\.)?([^./]+)\.(io|com|org)', url)
        library = library_match.group(2) if library_match else None
        
        # Check if we already have documentation for this library
        update_progress(f"[URL {i+1}/{len(urls)}] Checking for existing documentation for {library if library else 'unknown library'}...", 
                      progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.1)
        update_display_progress()
        
        vector_results, structured_docs = search_documentation(client, task, library)
        
        # If we don't have enough relevant results, scrape the documentation
        if len(vector_results) < 3 or not structured_docs:
            update_progress(f"[URL {i+1}/{len(urls)}] Insufficient existing documentation. Getting new documentation from {url}...", 
                          progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.2)
            update_display_progress()
            
            # Use FireCrawl or fallback to BeautifulSoup
            documentation, vector_docs = scrape_documentation(url)
            update_display_progress()  # Update with the latest status
            
            if documentation and vector_docs:
                update_progress(f"[URL {i+1}/{len(urls)}] Storing documentation from {url}...", 
                              progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.6)
                update_display_progress()
                
                store_result = store_documentation(client, documentation, vector_docs)
                
                if store_result:
                    update_progress(f"[URL {i+1}/{len(urls)}] Documentation from {url} processed and stored successfully!", 
                                  progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.8)
                else:
                    update_progress(f"[URL {i+1}/{len(urls)}] Documentation from {url} was processed but could not be stored completely.", 
                                  "warning", 
                                  progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.7)
                update_display_progress()
                
                # Search again with the new documentation
                update_progress(f"[URL {i+1}/{len(urls)}] Searching newly stored documentation for '{task}'...", 
                              progress_value=current_url_progress_start + (current_url_progress_end - current_url_progress_start) * 0.9)
                update_display_progress()
                
                vector_results, structured_docs = search_documentation(client, task, library)
        
        # Add results to our collections
        all_vector_results.extend(vector_results)
        all_structured_docs.extend(structured_docs)
        
        update_progress(f"[URL {i+1}/{len(urls)}] Successfully processed documentation for {library if library else 'unknown library'}", 
                      progress_value=current_url_progress_end)
        update_display_progress()
    
    # Generate the code solution
    update_progress(f"All documentation processed. Generating code solution for: {task}", progress_value=70)
    update_display_progress()
    
    solution = generate_code_solution(task, all_vector_results, all_structured_docs)
    
    if client:
        client.close()
    
    update_progress("✅ Task completed!", progress_value=100)
    update_display_progress()
    
    return solution

# Function to handle feedback and corrections
def process_feedback(feedback, original_solution):
    # Create a card for the progress section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Processing Feedback")
        
        # Create columns for structured display
        status_col, progress_col = st.columns([3, 1])
        
        with status_col:
            progress_status = st.empty()
        
        with progress_col:
            progress_bar = st.progress(0)
            progress_percentage = st.empty()
        
        progress_details_container = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Function to update the progress display
    def update_display_progress():
        # Update the progress bar
        progress_bar.progress(st.session_state.current_progress / 100)
        progress_percentage.markdown(f"**{st.session_state.current_progress}%**")
        
        # Update the status message
        progress_status.info(st.session_state.progress_status)
        
        # Show recent progress details
        if st.session_state.progress_details:
            details_md = ""
            # Show the last 5 progress details
            for detail in st.session_state.progress_details[-5:]:
                if detail["level"] == "error":
                    icon = "❌"
                    color = "#f44336"
                elif detail["level"] == "warning":
                    icon = "⚠️"
                    color = "#ff9800"
                else:
                    icon = "ℹ️"
                    color = "#2196f3"
                
                details_md += f"<span class='log-timestamp'>{detail['time']}</span> {icon} <span style='color:{color}'>{detail['message']}</span><br>"
            
            progress_details_container.markdown(details_md, unsafe_allow_html=True)
    
    # Reset progress
    st.session_state.current_progress = 0
    st.session_state.progress_details = []
    
    update_progress("Processing feedback...", progress_value=10)
    update_display_progress()
    
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        update_progress("Connecting to MongoDB database...", progress_value=15)
        update_display_progress()
        client = get_mongodb_connection()
        if not client:
            update_progress("Failed to connect to the database. Please try again.", "error", progress_value=15)
            update_display_progress()
            return "Failed to connect to the database. Please try again."
    
    # Re-search documentation based on the feedback
    update_progress(f"Analyzing feedback: '{feedback}'...", progress_value=20)
    update_display_progress()
    
    vector_results, structured_docs = search_documentation(client, feedback)
    
    # Extract library name from structured docs for parameter optimization
    sdk_name = "Unknown"
    if structured_docs and len(structured_docs) > 0:
        if "library" in structured_docs[0]:
            sdk_name = structured_docs[0]["library"]
    
    # Get the selected model with optimal parameters
    update_progress(f"Initializing {llm_provider} for feedback processing...", progress_value=40)
    update_display_progress()
    
    llm = get_llm(llm_provider, task=feedback, sdk_name=sdk_name)
    
    if not llm:
        update_progress("Could not initialize the language model. Please check your API keys.", "error", progress_value=40)
        update_display_progress()
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
        update_progress("Generating improved solution based on feedback...", progress_value=60)
        update_display_progress()
        
        if isinstance(llm, ChatHuggingFace):
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with improving code based on user feedback."),
                HumanMessage(content=feedback_prompt)
            ]
            # Set up callbacks for streaming
            llm.callbacks = [streaming_handler]
            update_progress("Processing with HuggingFace model...", progress_value=70)
            update_display_progress()
            ai_msg = llm.invoke(messages)
            improved_solution = ai_msg.content
            
        elif isinstance(llm, Replicate):
            # For Replicate, use the same approach as in generate_code_solution
            formatted_feedback_prompt = "System: You are an expert Python developer tasked with improving code based on user feedback.\n\nHuman: " + feedback_prompt + "\n\nAssistant:"
            
            # Set up callbacks for streaming
            llm.callbacks = [streaming_handler]
            update_progress("Processing with Replicate model...", progress_value=70)
            update_display_progress()
            improved_solution = llm.invoke(formatted_feedback_prompt)
            
        else:
            # For OpenAI and Anthropic
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with improving code based on user feedback."),
                HumanMessage(content=feedback_prompt)
            ]
            # Enable streaming
            llm.callbacks = [streaming_handler]
            update_progress(f"Processing with {llm_provider}...", progress_value=70)
            update_display_progress()
            response = llm.invoke(messages)
            improved_solution = response.content
        
        # Clean and format the improved solution
        update_progress("Cleaning and formatting solution...", progress_value=90)
        update_display_progress()
        final_code = clean_code_from_timestamps(improved_solution)
        
        # Check if the solution is wrapped in a code block
        if not (final_code.startswith("```python") or final_code.startswith("```")):
            # Wrap it in a code block for consistency
            final_code = f"```python\n{final_code}\n```"
        
        update_progress("✅ Feedback processed and solution improved successfully!", progress_value=100)
        update_display_progress()
        
        if client:
            client.close()
        
        return final_code
        
    except Exception as e:
        update_progress(f"Error processing feedback: {e}", "error", progress_value=70)
        update_display_progress()
        
        if client:
            client.close()
        return f"Error processing feedback: {e}"

# Main container for the chat and code UI
main_container = st.container()

with main_container:
    # Create multiple tabs for different views
    tabs = st.tabs(["💬 Chat", "🖥️ Code Solution", "📚 Documentation Sources", "📊 Progress Log"])
    
    # Tab 1: Chat Interface
    with tabs[0]:
        # Chat message display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    # Tab 2: Code Solution
    with tabs[1]:
        code_container = st.container()
        with code_container:
            if st.session_state.code_solution:
                # Extract code from markdown code blocks if present
                code_pattern = r'```(?:python|py)(.*?)```'
                code_matches = re.findall(code_pattern, st.session_state.code_solution, re.DOTALL)
                
                if code_matches:
                    # Show explanation before code (if available)
                    explanation_text = st.session_state.code_solution.split("```python")[0] if "```python" in st.session_state.code_solution else st.session_state.code_solution.split("```py")[0]
                    if explanation_text.strip():
                        st.markdown(explanation_text)
                    
                    # Allow downloading the code as a file
                    code_content = code_matches[0].strip()
                    
                    # Create a download button for the code
                    st.download_button(
                        label="📥 Download Code",
                        data=code_content,
                        file_name="solution.py",
                        mime="text/plain",
                        help="Download the generated code as a Python file"
                    )
                    
                    # Determine a nice filename for the solution
                    if st.session_state.messages and len(st.session_state.messages) > 0:
                        first_message = st.session_state.messages[0]["content"]
                        # Extract a short name from the first message
                        words = first_message.split()[:5]  # Take first 5 words max
                        filename = "_".join(word.lower() for word in words if word.isalnum())
                        if filename:
                            filename = f"{filename}.py"
                        else:
                            filename = "solution.py"
                    else:
                        filename = "solution.py"
                    
                    # Display the code with nice styling
                    st.code(code_content, language="python")
                    
                    # Show any text that follows the code block
                    if "```" in st.session_state.code_solution:
                        after_code = st.session_state.code_solution.split("```")[-1]
                        if after_code.strip():
                            st.markdown(after_code)
                else:
                    # If no code block markers, display as is
                    st.code(clean_code_from_timestamps(st.session_state.code_solution), language="python")
            else:
                st.info("No code solution generated yet. Submit a task to generate code.")
    
    # Tab 3: Documentation Sources
    with tabs[2]:
        docs_container = st.container()
        with docs_container:
            if st.session_state.documentation_sources:
                st.subheader("SDK Documentation Sources")
                
                # Create a nice table of documentation sources
                docs_data = []
                for source in st.session_state.documentation_sources:
                    docs_data.append({
                        "SDK/Library": source.get("library", "Unknown"),
                        "Version": source.get("version", "Latest"),
                        "Documents": source.get("document_count", 0),
                        "URL": source.get("url", ""),
                        "Last Processed": datetime.datetime.fromisoformat(source.get("timestamp", datetime.datetime.now().isoformat())).strftime("%Y-%m-%d %H:%M")
                    })
                
                st.dataframe(docs_data, use_container_width=True)
                
                if st.session_state.last_url_processed:
                    st.markdown(f"Last processed URL: [{st.session_state.last_url_processed}]({st.session_state.last_url_processed})")
            else:
                st.info("No documentation sources processed yet. Submit a task with SDK documentation URLs to begin.")
    
    # Tab 4: Progress Log
    with tabs[3]:
        log_container = st.container()
        with log_container:
            if st.session_state.progress_details:
                st.subheader("Detailed Processing Log")
                
                # Create a formatted log display
                log_md = ""
                for detail in st.session_state.progress_details:
                    if detail["level"] == "error":
                        icon = "❌"
                        color = "#f44336"
                    elif detail["level"] == "warning":
                        icon = "⚠️" 
                        color = "#ff9800"
                    else:
                        icon = "ℹ️"
                        color = "#2196f3"
                    
                    log_md += f"<span class='log-timestamp'>{detail['time']}</span> {icon} <span style='color:{color}'>{detail['message']}</span><br>"
                
                st.markdown(log_md, unsafe_allow_html=True)
                
                # Add option to clear the log
                if st.button("Clear Log"):
                    st.session_state.progress_details = []
                    st.rerun()
            else:
                st.info("No processing logs available yet.")

# Show database connection error if any
if "db_connection_error" in st.session_state and st.session_state.db_connection_error:
    st.error(f"Database connection issue: {st.session_state.db_connection_error}")

# Action buttons for managing the conversation
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("📋 New Task", use_container_width=True, help="Clear the current conversation and start fresh"):
        st.session_state.messages = []
        st.session_state.code_solution = ""
        st.session_state.progress_status = ""
        st.session_state.progress_details = []
        st.session_state.current_progress = 0
        st.rerun()

with col2:
    if st.session_state.code_solution and st.button("✏️ Edit & Fix", use_container_width=True, help="Submit feedback to improve the current solution"):
        feedback_prompt = f"I'd like to make improvements to the code. Please provide feedback in the chat below."
        st.session_state.messages.append({"role": "assistant", "content": feedback_prompt})
        st.rerun()

with col3:
    # Option to show example prompts
    if st.button("💡 Example Tasks", use_container_width=True, help="Show example tasks to get started"):
        with st.expander("Example Tasks", expanded=True):
            example_tasks = [
                "Using the Streamlit API at https://docs.streamlit.io/library/api-reference, create a simple dashboard that displays stock price data with a date selector and multiple stock ticker selection.",
                "Create a web scraper using BeautifulSoup from https://www.crummy.com/software/BeautifulSoup/bs4/doc/ that extracts product information from an e-commerce site.",
                "Build a chatbot using the LangChain documentation at https://python.langchain.com/docs/get_started/introduction to create a conversational agent with memory.",
                "Create a data visualization app using Plotly from https://plotly.com/python/getting-started/ that shows interactive scatter plots and histograms.",
                "Create a FastAPI application based on https://fastapi.tiangolo.com/tutorial/ that provides a RESTful API for a todo list with CRUD operations."
            ]
            
            for i, task in enumerate(example_tasks):
                if st.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                    # Set the example as the user message
                    st.session_state.messages.append({"role": "user", "content": task})
                    st.rerun()

# User input area
st.markdown('<div class="card task-input">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    if prompt := st.chat_input("Describe what you want to build and include SDK documentation URLs..."):
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
                message_placeholder.markdown("I've updated the solution based on your feedback. Check the Code Solution tab above.")
                
                # Set active tab to Code Solution
                st.session_state.active_tab = "Code Solution"
            else:
                # Process new request
                message_placeholder.markdown("I'm working on your request...")
                response = process_request(prompt)
                st.session_state.code_solution = response
                message_placeholder.markdown("I've generated a solution for your task. Check the Code Solution tab above.")
                
                # Set active tab to Code Solution
                st.session_state.active_tab = "Code Solution"
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": 
            "I've updated the solution based on your feedback. Check the Code Solution tab above." if is_feedback 
            else "I've generated a solution for your task. Check the Code Solution tab above."
        })
        
        # Reset processing flag
        st.session_state.processing = False
        
        # Force a rerun to update the UI with the new code
        st.rerun()

with col2:
    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    if st.session_state.processing:
        with st.spinner("Processing..."):
            st.markdown("Working...")
    
st.markdown('</div>', unsafe_allow_html=True)

# Page footer with credits and version info
st.markdown("---")
cols = st.columns([1, 1, 1])
with cols[0]:
    st.markdown("**Other Tales CodeMaker v1.0.0**")
with cols[1]:
    st.markdown("⭐ [Star on GitHub](https://github.com/othertales/codemaker)")
with cols[2]:
    st.markdown("🐛 [Report an issue](https://github.com/othertales/codemaker/issues)")

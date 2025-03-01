import streamlit as st
import os
import datetime
import json
import re
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo.server_api import ServerApi
from langchain_community.vectorstores import MongoDBAtlasVectorSearch, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Replicate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
import time
import getpass

# Set page config
st.set_page_config(page_title="LLM Python Solution Builder", page_icon="🧩", layout="wide")

# App title and description
st.title("SDK Code Generator")
st.markdown("""
This tool helps you generate Python code solutions, checking it against the latest SDK documentation. 
Provide your task in the format: 'Using the Latest Python SDK at [URL], [task]'
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

# Sidebar for configuration
with st.sidebar:
    st.header("Model Configuration")
    llm_provider = st.selectbox(
        "Generative Model",
        ["OpenAI GPT-4o", "OpenAI o1-Preview", "Anthropic Claude 3.7 Sonnet", "Anthropic Claude 3.5 Latest", "Replicate Model", "HuggingFace Hub"]
    )
    
    # Show model name input field only for Replicate and HuggingFace
    custom_model = None
    if llm_provider in ["Replicate Model", "HuggingFace Hub"]:
        custom_model = st.text_input("Model", 
                                    value="meta/meta-llama-3-8b-instruct" if llm_provider == "Replicate Model" else "HuggingFaceH4/zephyr-7b-beta",
                                    key="custom_model")
        
        # Add API key input for the respective service
        if llm_provider == "Replicate Model" and not os.environ.get("REPLICATE_API_TOKEN"):
            replicate_api_key = st.text_input("Replicate API Key", type="password", key="replicate_api_key")
            if replicate_api_key:
                os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
        
        if llm_provider == "HuggingFace Hub" and not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
            hf_api_key = st.text_input("HuggingFace API Key", type="password", key="hf_api_key")
            if hf_api_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Function to get the appropriate LLM based on user selection
def get_llm(provider, temperature=0.2):
    if provider == "OpenAI GPT-4o":
        return ChatOpenAI(model="gpt-4o-latest", temperature=temperature)
    elif provider == "OpenAI o1-Preview":
        return ChatOpenAI(model="o1-preview", temperature=temperature)
    elif provider == "Anthropic Claude 3.7 Sonnet":
        return ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=temperature)
    elif provider == "Anthropic Claude 3.5 Latest":
        return ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=temperature)
    elif provider == "Replicate Model" and custom_model:
        return Replicate(
            model=custom_model,
            temperature=temperature
        )
    elif provider == "HuggingFace Hub" and custom_model:
        llm = HuggingFaceEndpoint(
            repo_id=custom_model,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        return ChatHuggingFace(llm=llm)
    else:
        # Default to GPT-4o if nothing valid is selected
        return ChatOpenAI(model="gpt-4o-latest", temperature=temperature)

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

# Function to scrape the SDK documentation
def scrape_documentation(url):
    try:
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
        st.error(f"Error scraping documentation: {e}")
        return None, None

# Function to store documentation (MongoDB or Chroma)
def store_documentation(client, documentation, vector_docs):
    try:
        vector_store_type = get_vector_store_type()
        embeddings = OpenAIEmbeddings()
        
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
        st.error(f"Error storing documentation: {e}")
        return False

# Function to search for documentation (MongoDB or Chroma)
def search_documentation(client, query, library=None):
    try:
        vector_store_type = get_vector_store_type()
        embeddings = OpenAIEmbeddings()
        
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
                st.warning(f"Error loading Chroma vector store: {e}. Creating a new one.")
                results = []
                structured_docs = []
        
        return results, structured_docs
    
    except Exception as e:
        st.error(f"Error searching documentation: {e}")
        return [], []

# Function to generate code solution using the selected model
def generate_code_solution(task, vector_results, structured_docs):
    try:
        # Create a prompt for the code generation
        prompt_template = """
        You are an expert Python developer tasked with generating code based on SDK documentation.
        
        USER TASK: {task}
        
        DOCUMENTATION CHUNKS:
        {vector_chunks}
        
        STRUCTURED DOCUMENTATION:
        {structured_docs}
        
        Your task is to:
        1. Understand the SDK's structure and available classes/functions
        2. Pay special attention to the correct import statements
        3. Generate complete, working code for the user's task
        4. Include clear comments explaining your implementation
        5. Verify that all functions and classes used are correctly imported
        
        YOUR SOLUTION (complete Python code):
        """
        
        # Format the vector chunks
        vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
        
        # Format the structured docs
        structured_docs_str = json.dumps(structured_docs, indent=2)
        
        # Get the selected model
        llm = get_llm(llm_provider, temperature=0.2)
        
        # Handle different model types
        if llm_provider == "HuggingFace Hub":
            # For ChatHuggingFace model
            messages = [
                SystemMessage(content="You are an expert Python developer tasked with generating code based on SDK documentation."),
                HumanMessage(content=f"""
                USER TASK: {task}
                
                DOCUMENTATION CHUNKS:
                {vector_chunks}
                
                STRUCTURED DOCUMENTATION:
                {structured_docs_str}
                
                Your task is to:
                1. Understand the SDK's structure and available classes/functions
                2. Pay special attention to the correct import statements
                3. Generate complete, working code for the user's task
                4. Include clear comments explaining your implementation
                5. Verify that all functions and classes used are correctly imported
                
                YOUR SOLUTION (complete Python code):
                """)
            ]
            
            ai_msg = llm.invoke(messages)
            solution = ai_msg.content
        else:
            # For other LLM types
            # Create the prompt
            prompt = PromptTemplate(
                input_variables=["task", "vector_chunks", "structured_docs"],
                template=prompt_template
            )
            
            # Create the chain with the selected model
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Generate the solution
            solution = chain.run(
                task=task,
                vector_chunks=vector_chunks,
                structured_docs=structured_docs_str
            )
        
        # Review the solution for accuracy
        review_prompt_template = """
        You are an expert code reviewer. You need to verify if the generated code correctly 
        uses the SDK according to its documentation.
        
        GENERATED CODE:
        {solution}
        
        DOCUMENTATION CHUNKS:
        {vector_chunks}
        
        STRUCTURED DOCUMENTATION:
        {structured_docs}
        
        Please review the code and check:
        1. Are the import statements correct?
        2. Are all classes and functions used correctly?
        3. Does the code fulfill the user's requirements?
        4. Are there any errors or improvements needed?
        
        If any issues are found, provide the corrected code. If no issues are found, simply return "VERIFIED" followed by the original code.
        
        YOUR REVIEW:
        """
        
        # Handle review based on model type
        if llm_provider == "HuggingFace Hub":
            # For ChatHuggingFace model
            review_messages = [
                SystemMessage(content="You are an expert code reviewer."),
                HumanMessage(content=f"""
                GENERATED CODE:
                {solution}
                
                DOCUMENTATION CHUNKS:
                {vector_chunks}
                
                STRUCTURED DOCUMENTATION:
                {structured_docs_str}
                
                Please review the code and check:
                1. Are the import statements correct?
                2. Are all classes and functions used correctly?
                3. Does the code fulfill the user's requirements?
                4. Are there any errors or improvements needed?
                
                If any issues are found, provide the corrected code. If no issues are found, simply return "VERIFIED" followed by the original code.
                """)
            ]
            
            review_ai_msg = llm.invoke(review_messages)
            review_result = review_ai_msg.content
        else:
            # For other LLM types
            review_prompt = PromptTemplate(
                input_variables=["solution", "vector_chunks", "structured_docs"],
                template=review_prompt_template
            )
            
            review_chain = LLMChain(llm=llm, prompt=review_prompt)
            
            review_result = review_chain.run(
                solution=solution,
                vector_chunks=vector_chunks,
                structured_docs=structured_docs_str
            )
        
        # If the review verifies the code, return the original solution
        if review_result.startswith("VERIFIED"):
            return solution
        
        # Otherwise, return the corrected code
        return review_result
    
    except Exception as e:
        st.error(f"Error generating solution: {e}")
        return f"Error generating solution: {e}"

# Function to process a user request
def process_request(request):
    vector_store_type = get_vector_store_type()
    client = None
    
    if vector_store_type == "mongodb":
        client = get_mongodb_connection()
        if not client:
            return "Failed to connect to the database. Please try again."
    
    # Extract URL and task from the request
    url_match = re.search(r'at\s+(https?://\S+),\s+build', request)
    if not url_match:
        return "Could not identify the SDK URL in your request. Please use the format 'Using the Latest Python SDK at [URL], build a [task]'."
    
    url = url_match.group(1)
    
    # Extract task
    task_match = re.search(r'build\s+(.+)', request)
    if not task_match:
        return "Could not identify the task in your request. Please use the format 'Using the Latest Python SDK at [URL], build a [task]'."
    
    task = task_match.group(1)
    
    # Extract library name from URL if possible
    library_match = re.search(r'//([^/]+\.)?([^./]+)\.(io|com|org)', url)
    library = library_match.group(2) if library_match else None
    
    # Check if we already have documentation for this library
    vector_results, structured_docs = search_documentation(client, task, library)
    
    # If we don't have enough relevant results, scrape the documentation
    if len(vector_results) < 3 or not structured_docs:
        st.info("Scraping SDK documentation... This may take a moment.")
        documentation, vector_docs = scrape_documentation(url)
        
        if documentation and vector_docs:
            store_result = store_documentation(client, documentation, vector_docs)
            if store_result:
                st.success("Documentation scraped and stored successfully!")
            else:
                st.warning("Documentation was scraped but could not be stored completely.")
            
            # Search again with the new documentation
            vector_results, structured_docs = search_documentation(client, task, library)
    
    # Generate the code solution
    st.info("Generating code solution...")
    solution = generate_code_solution(task, vector_results, structured_docs)
    
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
    vector_results, structured_docs = search_documentation(client, feedback)
    
    # Get the selected model
    llm = get_llm(llm_provider, temperature=0.2)
    
    # Handle different model types for feedback processing
    if llm_provider == "HuggingFace Hub":
        # Format the vector chunks
        vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
        
        # Format the structured docs
        structured_docs_str = json.dumps(structured_docs, indent=2)
        
        # For ChatHuggingFace model
        messages = [
            SystemMessage(content="You are an expert Python developer tasked with improving code based on user feedback."),
            HumanMessage(content=f"""
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
            2. Check the documentation to verify the correct usage
            3. Fix any issues in the original code
            4. Ensure all imports and API usages are correct
            5. Provide the improved solution
            """)
        ]
        
        ai_msg = llm.invoke(messages)
        improved_solution = ai_msg.content
    else:
        # Generate improved solution based on feedback
        feedback_prompt_template = """
        You are an expert Python developer tasked with improving code based on user feedback.
        
        ORIGINAL CODE:
        {original_solution}
        
        USER FEEDBACK:
        {feedback}
        
        DOCUMENTATION CHUNKS:
        {vector_chunks}
        
        STRUCTURED DOCUMENTATION:
        {structured_docs}
        
        Your task is to:
        1. Understand the user's feedback
        2. Check the documentation to verify the correct usage
        3. Fix any issues in the original code
        4. Ensure all imports and API usages are correct
        5. Provide the improved solution
        
        YOUR IMPROVED SOLUTION:
        """
        
        # Format the vector chunks
        vector_chunks = "\n\n".join([doc.page_content for doc in vector_results])
        
        # Format the structured docs
        structured_docs_str = json.dumps(structured_docs, indent=2)
        
        # Create the prompt
        prompt = PromptTemplate(
            input_variables=["original_solution", "feedback", "vector_chunks", "structured_docs"],
            template=feedback_prompt_template
        )
        
        # Create the chain with the selected model
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate the improved solution
        improved_solution = chain.run(
            original_solution=original_solution,
            feedback=feedback,
            vector_chunks=vector_chunks,
            structured_docs=structured_docs_str
        )
    
    if client:
        client.close()
    return improved_solution

# Chat message display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display code solution if available
if st.session_state.code_solution:
    with st.expander("Generated Code Solution", expanded=True):
        st.code(st.session_state.code_solution, language="python")

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

# Show database connection error if any
if "db_connection_error" in st.session_state and st.session_state.db_connection_error:
    st.error(f"Database connection issue: {st.session_state.db_connection_error}")

# Add a clear button to reset the conversation
if st.button("Start New Task"):
    st.session_state.messages = []
    st.session_state.code_solution = ""
    st.rerun()

import os
import json
from google import genai
from google.genai.errors import APIError

# LangChain V1.0+ Modular Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# 🛠️ NEW: Imports for the modern LCEL chain helpers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.documents import Document

# --- CONFIGURATION (Use your extracted JSON file and API Key) ---
API_KEY = "AIzaSyBHC9zkhP5gLgEHtqu84Aln8Zv4oKN8s6U" 
JSON_FILE_PATH = 'AI UNIT 2.json'  # <-- Ensure this JSON file exists!
MODEL_NAME = 'gemini-2.5-flash'
VECTOR_DB_PATH = 'chroma_db'

# --- 1. DATA PREPARATION FUNCTIONS ---

def load_text_from_json(json_path):
    """
    Reads the structured JSON file and converts its page content into 
    a list of LangChain Document objects.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n❌ ERROR: JSON file not found at {json_path}. Please run your extraction code first.")
        return []
    except json.JSONDecodeError:
        print(f"\n❌ ERROR: Failed to decode JSON from {json_path}.")
        return []
    
    documents = []
    source_file = data.get("source_file", "unknown_pdf")
    
    for page in data.get("pages", []):
        doc = Document(
            page_content=page["content"],
            metadata={
                "source": source_file,
                "page": page["page_number"],
            }
        )
        documents.append(doc)
        
    print(f"✅ Loaded {len(documents)} pages from {source_file} into memory.")
    return documents

def build_vector_store(documents):
    """
    Splits the documents and creates/updates a Chroma vector store.
    """
    if not documents:
        return None

    # 1. Split the document text into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    all_chunks = text_splitter.split_documents(documents)
    print(f"✅ Split data into {len(all_chunks)} chunks for the vector store.")

    # 2. Create the Gemini Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=API_KEY
    )

    # 3. Create and persist the Chroma Vector Store
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"✅ Vector store created and saved to '{VECTOR_DB_PATH}'.")
    return vector_store

# --- 2. RAG CHATBOT IMPLEMENTATION (Refactored to use LCEL Helpers) ---

def initialize_rag_chatbot():
    """
    Sets up the RAG chain using the vector store, Gemini model, and a custom prompt
    using the modern create_retrieval_chain helper.
    """
    
    # 1. Load Data and Build Vector Store
    documents = load_text_from_json(JSON_FILE_PATH)
    vector_store = build_vector_store(documents)
    
    if not vector_store:
        return None

    # 2. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, 
        google_api_key=API_KEY, 
        temperature=0.0
    )
    
    # 3. Define the custom prompt for the RAG chain (using ChatPromptTemplate is preferred)
    # Note: The new chain helpers use specific input keys (context, input)
    RAG_PROMPT_TEMPLATE = """
    You are an AI assistant specialized in answering questions ONLY based on the provided PDF content.
    If the answer is NOT found in the context provided below, you MUST respond with: 
    "I can only answer questions related to the provided PDF content. That topic is not covered in the document."

    Context:
    ---
    {context}
    ---

    Question: {input}
    Answer:
    """
    
    # Create the prompt structure
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # 4. Create the Retrieval-Augmented Generation (RAG) Chain using modern helpers
    
    # a. Create the document combination chain (combines retrieved docs into one prompt)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    # b. Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # c. Combine the retriever and the document chain into the final RAG chain
    # This automatically handles input/output mapping and returns sources
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain

def run_rag_chatbot(qa_chain):
    """
    Runs the interactive chat loop using the RAG chain.
    """
    print("-" * 70)
    print(f"🧠 RAG Chatbot Initialized (Knowledge Base: {os.path.basename(JSON_FILE_PATH)})")
    print("💡 Ask me about the content of the PDF! Type 'quit' or 'exit' to end.")
    print("-" * 70)

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye! Knowledge session ended.")
            break
        
        if not user_input.strip():
            continue

        try:
            # Run the query through the RAG chain
            print("GemBot: (Searching PDF knowledge base...)")
            
            # 🛠️ CHANGE: Input key is now 'input' for create_retrieval_chain
            result = qa_chain.invoke({"input": user_input})
            
            # Print the final answer
            # 🛠️ CHANGE: Output key is now 'answer' for create_retrieval_chain
            answer = result["answer"].strip() 
            print(f"GemBot: {answer}")
            
            # Optional: Display the source of the answer
            # 🛠️ CHANGE: Source key is now 'context' for create_retrieval_chain
            if result.get("context"):
                # Extract unique source pages
                sources = set(f'Page {doc.metadata["page"]}' for doc in result["context"])
                print(f"📚 Sources: {', '.join(sources)}")
            
        except Exception as e:
            print(f"\n❌ An error occurred during the RAG process: {e}")

# ======================================================================
# --- MAIN EXECUTION ---
# ======================================================================

if __name__ == "__main__":
    if not API_KEY or API_KEY == "YOUR_NEW_API_KEY_HERE":
        print("🛑 ERROR: Please set a valid GOOGLE_API_KEY in the script.")
    else:
        qa_chain = initialize_rag_chatbot()
        if qa_chain:
            run_rag_chatbot(qa_chain)

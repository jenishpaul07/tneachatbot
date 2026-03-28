import os
import streamlit as st

# PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="TNEA Assistant", page_icon="🎓", layout="centered")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline

load_dotenv()

# Cache the models so they aren't reloaded on every chat!
@st.cache_resource
def load_knowledge_base():
    embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    database = FAISS.load_local("faiss_index", embeds, allow_dangerous_deserialization=True)
    return database.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def load_llm():
    return HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        # Cutting max_new_tokens to 100 forces the AI to reply faster and more concisely!
        pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.1}
    )

# --- Custom UI Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #1E3A8A; text-align: center; font-weight: 800; margin-bottom: 0; padding-top: 20px;}
    .sub-header { font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 30px; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🎓 TNEA Counseling Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent Guide for Tamil Nadu Engineering Admissions</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 About the Bot")
    st.info(
        "This AI chatbot runs 100% locally and securely right on your machine! "
        "It acts as an expert on TNEA cutoffs, top colleges, and the eligibility process."
    )
    st.divider()
    st.caption("Powered by **Streamlit**, **LangChain**, and **Qwen AI**.")

# Loading the Brain
with st.spinner("Initializing AI Brain..."):
    retriever = load_knowledge_base()
    llm = load_llm()

# --- Chat Interface ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I am your TNEA Counseling expert. Ask me anything about cutoffs, colleges, or the admission steps!"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about cutoffs, eligibility, colleges..."):
    
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching the database..."):
            # 1. Retrieve documents
            docs = retriever.invoke(prompt)
            context = "\n".join([f"- {doc.page_content}" for doc in docs])
            
            # 2. Build the strict prompt
            ai_prompt = (
                f"You are a helpful TNEA admission assistant.\n"
                f"Answer the user's question accurately based ONLY on the following Context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Answer:"
            )
            
            # 3. Request answer
            raw_response = llm.invoke(ai_prompt)
            
            # 4. Clean up response (some local LLMs repeat the prompt)
            if "Answer:" in raw_response:
                clean_response = raw_response.split("Answer:")[-1].strip()
            else:
                clean_response = raw_response.strip()

            st.markdown(clean_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
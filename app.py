import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# List of supported models (Updated)
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",  # Fast Llama model
    "deepseek-r1-distill-qwen-32b"  # DeepSeek model
]

# Model descriptions for better user understanding
MODEL_DESCRIPTIONS = {
    "llama-3.1-8b-instant": "Fast, efficient model for quick responses",
    "deepseek-r1-distill-qwen-32b": "Advanced distilled model with excellent performance"
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'model' not in st.session_state:
        st.session_state.model = SUPPORTED_MODELS[0]
    if 'memory_length' not in st.session_state:
        st.session_state.memory_length = 5

def create_conversation(model, memory_length):
    """Create a new conversation with the specified model and memory"""
    memory = ConversationBufferWindowMemory(k=memory_length)
    
    # Preload existing chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})
    
    # Initialize Groq chat model
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )
    
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def handle_user_input(user_question):
    """Process user input and generate response"""
    if not user_question.strip():
        return
    
    # Display user message
    st.chat_message("user").write(user_question)
    
    # Display thinking indicator
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.text("Thinking...")
        
        try:
            # Get response from model
            response = st.session_state.conversation.invoke({'input': user_question})
            chatbot_reply = response['response']
            
            # Save to chat history
            message = {'human': user_question, 'AI': chatbot_reply}
            st.session_state.chat_history.append(message)
            
            # Replace thinking indicator with actual response
            thinking_placeholder.empty()
            st.write(chatbot_reply)
            
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"⚠️ Error: {str(e)}")

def reset_conversation():
    """Clear conversation history and reset the chat"""
    st.session_state.chat_history = []
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length
    )
    st.rerun()

def handle_model_change():
    """Handle model change and recreate conversation"""
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length
    )

def main():
    # Page configuration
    st.set_page_config(
        page_title=" Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    with st.sidebar:
        st.title('🔧 Chat Settings')
        
        # Model selection with descriptions
        st.subheader('🔍 Model Selection')
        model = st.selectbox(
            'Choose a model',
            SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(st.session_state.model),
            format_func=lambda x: f"{x} - {MODEL_DESCRIPTIONS.get(x, '')}"
        )
        
        # Update model if changed
        if model != st.session_state.model:
            st.session_state.model = model
            handle_model_change()
        
        # Memory settings
        st.subheader('🧠 Memory Settings')
        memory_length = st.slider(
            'Conversation memory (messages):',
            1, 15, value=st.session_state.memory_length
        )
        
        # Update memory length if changed
        if memory_length != st.session_state.memory_length:
            st.session_state.memory_length = memory_length
            handle_model_change()
        
        # Reset button
        st.button("🗑️ Reset Conversation", on_click=reset_conversation)
        
        # API key status indicator
        st.subheader("🔑 API Status")
        if groq_api_key:
            st.success("Groq API Key: Connected")
        else:
            st.error("❌ Groq API key is missing. Please check your .env file.")
        

    
    # Main chat interface
    st.title("Chat Assistant 🤖")
    st.markdown(f"**Current Model**: {st.session_state.model}")
    
    # Initialize conversation if not already done
    if st.session_state.conversation is None:
        st.session_state.conversation = create_conversation(
            st.session_state.model, 
            st.session_state.memory_length
        )
    
    # Display chat history
    for message in st.session_state.chat_history:
        st.chat_message("user").write(message['human'])
        st.chat_message("assistant").write(message['AI'])
    
    # Check API key
    if not groq_api_key:
        st.error("❌ Groq API key is missing. Please check your .env file.")
        return
    
    # Chat input
    user_question = st.chat_input("Ask a question...")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
import streamlit as st
from langchain_pinecone import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from src.helper import download_hugging_face_embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from src.prompt import system_prompt
import os
import uuid

# Load environment variables
load_dotenv()

# Configure API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('google_api_key')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize embeddings and vector store
@st.cache_resource
def initialize_resources():
    embeddings = download_hugging_face_embeddings()
    index_name = "schmemebot"
    
    docsearch = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.4)
    
    return retriever, llm

# Page configuration
st.set_page_config(page_title="Scheme Navigator", page_icon="🔍")
st.title("Scheme Navigator - Indian Government Schemes")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get resources
retriever, llm = initialize_resources()

# Chat input
if prompt := st.chat_input("Ask about any Indian government scheme..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Create history text from previous messages
        history_text = ""
        for msg in st.session_state.messages[:-1]:  # Exclude the current message
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['content']}\n\n"
        
        # Create custom prompt with history
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nPrevious conversation history:\n" + history_text if history_text else system_prompt),
            ("human", "{input}")
        ])
        
        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, custom_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # First try with vector database
        response = rag_chain.invoke({"input": prompt})
        assistant_response = response["answer"]
        
        # Check if the response indicates lack of information
        if ("I don't have information" in assistant_response or 
            "I don't have specific information" in assistant_response or
            "I'm not sure about" in assistant_response or
            "I don't have enough information" in assistant_response):
            try:
                with st.status("Searching the web for more information..."):
                    # Initialize TavilySearch
                    search = TavilySearch(max_results=10, topic="general")
                    
                    # Search the internet
                    search_results = search.run(f"Indian government scheme {prompt}")
                    
                    if search_results:
                        # Create a new prompt with search results - Fixed to use prompt variable directly, not as a template variable
                        web_search_prompt = ChatPromptTemplate.from_messages([
                            ("system", 
                                f"""You are Scheme Navigator, a specialized assistant that provides information ONLY about CURRENTLY ACTIVE Indian government schemes.

                                IMPORTANT INSTRUCTIONS:
                                1. The user asked about: {prompt}
                                2. ONLY discuss schemes that are explicitly confirmed as CURRENTLY ACTIVE in the search results
                                3. NEVER mention schemes unless you can verify they are currently active
                                4. For each scheme mentioned, include the name of the sponsoring ministry/department and implementation date
                                5. Clearly distinguish between central government and state government schemes
                                6. Include specific eligibility criteria and benefits when available
                                7. For each piece of information, cite the specific source (URL) it came from
                                8. If you cannot find clear evidence a scheme is currently active, DO NOT mention it
                                9. If no currently active schemes are found in the search results, clearly state "I cannot find any currently active schemes matching your query based on the search results."
                                10. Do not make assumptions about schemes' status - only report what is explicitly stated in the search results

                                Use ONLY the following search results to formulate your response:

                                {search_results}"""
                            ),
                            ("human", "{input}")
                        ])
                        
                        # Generate response with web search results
                        web_chain = web_search_prompt | llm
                        web_response = web_chain.invoke({"input": prompt})
                        
                        # Replace the original response with the web search response
                        assistant_response = f"I couldn't find detailed information about this in my primary knowledge base, but I found some information online:\n\n{web_response.content}"
            except Exception as e:
                st.error(f"Web search failed: {str(e)}")
                # If the web search fails, keep the original response
        
        # Update the message placeholder with the final response
        message_placeholder.markdown(assistant_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
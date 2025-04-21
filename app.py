from flask import Flask, render_template, jsonify, request, session
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from src.helper import download_hugging_face_embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.prompt import *
import os
import uuid
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for sessions

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('google_api_key')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "schmemebot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# User sessions dictionary to store conversation history
user_sessions = {}

@app.route("/")
def index():
    # Generate a unique session ID if not already present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        user_sessions[session['session_id']] = []
    
    return render_template('chat.html')

@app.route("/clear", methods=["POST"])
def clear_history():
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in user_sessions:
            user_sessions[session_id] = []
    return jsonify({"status": "success"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = session.get('session_id')
    
    # If no valid session, create one
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        user_sessions[session_id] = []
    
    # Get chat history
    history = user_sessions[session_id]
    
    # Create a custom prompt with chat history included
    history_text = ""
    if history:
        for entry in history:
            history_text += f"User: {entry['user']}\n"
            history_text += f"Assistant: {entry['assistant']}\n\n"
    
    # Create full prompt with history
    custom_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nPrevious conversation history:\n" + history_text if history else system_prompt),
            ("human", "{input}")
        ]
    )
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.4)
    question_answer_chain = create_stuff_documents_chain(llm, custom_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Process the user's message
    response = rag_chain.invoke({"input": msg})
    assistant_response = response["answer"]
    
    # Store conversation in history
    history.append({
        "user": msg,
        "assistant": assistant_response
    })
    
    # Limit history to last 10 exchanges to prevent context window issues
    if len(history) > 10:
        history = history[-10:]
    
    user_sessions[session_id] = history
    
    return str(assistant_response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
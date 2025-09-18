import tempfile
import fitz  # PyMuPDF
import streamlit as st
import os
import random
import string
import spacy
import pyttsx3
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import chromadb
from llama_index.core import Settings,VectorStoreIndex,Document,SimpleDirectoryReader, load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool,ToolMetadata
import google.generativeai as genai


st.set_page_config(
    page_title="SAVE ME",
    page_icon=":books:",  
    layout="wide",
)
st.title("SAVE ME")


# Initialize session state variables
if "query_engine" not in st.session_state:
    st.session_state.query_engine=None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "scores" not in st.session_state:
    st.session_state.scores = 0
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""  
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False
if "summary" not in st.session_state:
    st.session_state.summary = ""  # Store summary text

nlp = spacy.load("en_core_web_sm")
uploaded_files = st.sidebar.file_uploader("Upload PDF", accept_multiple_files=True, type="pdf")

# Function for text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def reset_quiz_state():
    st.session_state.quiz_mode = False
    st.session_state.current_question = 0
    st.session_state.user_answer = ""
    st.session_state.chat_history.append({"role": "assistant", "content": "Quiz mode stopped."})

def initialize_query_engine(text):
    documents = [Document(text=text)]
    load_client = chromadb.PersistentClient("./chromadb")
    chroma_collection = load_client.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize Gemini models with API key
    def get_api_key(file_name="keyyy.txt"):
        with open(file_name, 'r') as file:
            return file.read().strip()

    gemini_embed = GeminiEmbedding(api_key=get_api_key(), model_name="models/embedding-001")
    llm = Gemini(api_key=get_api_key(), model_name="models/gemini-pro")

    Settings.llm = llm
    Settings.embed_model = gemini_embed

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    return index.as_query_engine()


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    print(f"Extracted text: {text}")  # Debugging line
    return text

# Allow the user to select the number of questions they want
num_questions = st.sidebar.number_input("How many questions on the quiz would you like?", min_value=1, max_value=20, value=5, step=1)

# Function to generate multiple-choice questions from text
def generate_questions_from_text(text, num_questions):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    words = text.split()

    # Remove punctuation from words and make them lowercase
    words = [word.strip(string.punctuation).lower() for word in words if len(word) > 3]

    # Count frequency of each word
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Select the top 10 most frequent words as keywords
    keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

    questions = []
    for sentence in sentences:
        # Use keywords to generate multiple-choice questions
        for keyword in keywords:
            if keyword in sentence.lower():
                # Create the question by replacing the keyword with a blank
                question_text = sentence.replace(keyword, "______")

                # Generate answer choices, with one correct answer and 3 distractors
                correct_answer = keyword
                distractors = random.sample([w for w in words if w != correct_answer], 3)  # Random incorrect options
                options = [correct_answer] + distractors
                random.shuffle(options)  # Shuffle to randomize correct answer position

                questions.append({
                    "question": f"What word completes this sentence? {question_text}",
                    "options": options,  # The MCQ options
                    "answer": correct_answer
                })
                break  # Move to the next sentence after finding a keyword

    # Randomly sample questions if too many are generated
    return random.sample(questions, min(num_questions, len(questions)))



def handle_submit():
    if st.session_state.selected_option is not None:
        correct_answer = st.session_state.quiz_questions[st.session_state.current_question]['answer']
        
        # Check if answer is correct
        if st.session_state.selected_option == correct_answer:
            st.session_state.scores += 1
            feedback = "✅ Correct!"
        else:
            feedback = f"❌ Incorrect. The correct answer was: {correct_answer}"
        
        # Update chat history
        question_text = st.session_state.quiz_questions[st.session_state.current_question]['question']
        st.session_state.chat_history.append({"role": "assistant", "content": question_text})
        st.session_state.chat_history.append({"role": "user", "content": f"Your answer: {st.session_state.selected_option}"})
        st.session_state.chat_history.append({"role": "assistant", "content": feedback})
        
        # Move to next question
        st.session_state.current_question += 1
        st.session_state.selected_option = None

def display_question():
    if st.session_state.current_question < len(st.session_state.quiz_questions):
        question = st.session_state.quiz_questions[st.session_state.current_question]
        
        # Display question
        st.write(f"Question {st.session_state.current_question + 1}: {question['question']}")
        
        # Create radio buttons for options
        selected = st.radio(
            "Select your answer:",question['options'],key=f"radio_{st.session_state.current_question}"
        )
        
        # Store selected option in session state
        st.session_state.selected_option = selected
        
        # Submit button
        st.button("Submit Answer", on_click=handle_submit, key=f"submit_{st.session_state.current_question}")

def run_quiz_from_pdf(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
            
            try:
                text += extract_text_from_pdf(tmp_file_path)
            except Exception as e:
                print(f"Error processing file {tmp_file_path}: {e}")
            finally:
                # Ensure the temporary file is deleted properly after use
                if os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                        print(f"Successfully removed temporary file: {tmp_file_path}")
                    except OSError as e:
                        print(f"Error removing file {tmp_file_path}: {e}")
                else:
                    print(f"Temporary file does not exist: {tmp_file_path}")
    
    if not text:
        st.error("Couldn't extract text from the PDF(s). Please upload a valid PDF.")
        return
    
    st.session_state.quiz_questions = generate_questions_from_text(text, num_questions)
    st.session_state.documents_processed = True
    st.session_state.quiz_mode = True

# Process user input and handle quiz
user_query = st.chat_input("Enter your query:")
if user_query:
    if "stop quiz" in user_query.lower() or "exit quiz" in user_query.lower() or "pause quiz" in user_query.lower():
        if st.session_state.quiz_mode:
            reset_quiz_state()
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "No quiz is currently running."})
    elif "quiz" in user_query.lower() or "start quiz" in user_query.lower():
        if uploaded_files:
            if not st.session_state.documents_processed:
                with st.spinner("Processing documents..."):
                    run_quiz_from_pdf(uploaded_files)
            st.write("Quiz mode activated! Starting the quiz.")
            st.session_state.quiz_mode = True
        else:
            st.error("Please upload a PDF file to generate quiz questions.")
    # Summarize and recite summary logic
    elif user_query.lower() == "summarize" or user_query.lower() == "summarize this file" or user_query.lower() == "summary of the pdf":
        text = ""
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
                text += extract_text_from_pdf(tmp_file_path)
    
        if not text:
            st.error("Couldn't extract text from the PDF(s). Please upload a valid PDF.")
        else:
            if not st.session_state.query_engine:
                st.session_state.query_engine = initialize_query_engine(text)
            response = st.session_state.query_engine.query("summarize this document")
        
        # Store the summary in session state for recitation
        st.session_state.summary = response.response
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response.response})

# Recite the summary if available
    elif user_query.lower() == "recite the summary" or user_query.lower() == "recite":
        if st.session_state.summary:
            text_to_speech(st.session_state.summary)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": "Reciting the summary."})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "No summary available to recite. Please ask for a summary first."})

    elif user_query.lower() == "what can you do" or user_query.lower() == "help" or user_query.lower() == "hello what can you do":
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        response = "As a study-helping bot, I can assist you in the following ways:\n\n" \
                   "1. Summarize the contents of a PDF file you upload.\n" \
                   "2. Go into quiz mode and ask you multiple-choice questions based on the PDF content.\n" \
                   "3. Recite the summary of the PDF file.\n\n" \
                   "Just let me know what you'd like me to do!"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    elif user_query.lower() == "thank you" or user_query.lower() == "thankyou" or user_query.lower() == "thank you for your help":
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        response = "Sure! Just let me know what you'd like me to do!"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
    else:
        if st.session_state.query_engine:
            response = st.session_state.query_engine.query(user_query)
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response.response})
            with st.chat_message('user'):
                st.write(user_query)
            with st.chat_message('assistant'):
                st.write(response.response)
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": "I'm here to help! Ask me anything."})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display current question if in quiz mode
if st.session_state.quiz_mode and not (st.session_state.current_question >= len(st.session_state.quiz_questions)):
    display_question()
elif st.session_state.current_question >= len(st.session_state.quiz_questions) and st.session_state.quiz_mode:
    st.write(f"Quiz finished! Your score is: {st.session_state.scores} out of {len(st.session_state.quiz_questions)}")
    reset_quiz_state()

def get_index(docs,name):
    index=None
    if not os.path.exists(name):
        print("building index")
        index=VectorStoreIndex.from_documents(docs,show_progress=True)
        index.storage_content.persist(persist_dir=name)
    else:
        index=load_index_from_storage(StorageContext.from_default(persist_dir=name))
    return index
docs=SimpleDirectoryReader("Data").load_data()
#print(docs[0])


class LLMPromptTool:
    def query(self,prompt):
        model=genai.GenerativeModel("gemini-1.5-flash",
                              system_instruction=[
                                  '''
                                  As a study-helping bot, your 
                                  role is to assist users in 
                                  studying from a pdf uploaded by summarizing 
                                  contents from the pdf and going into quiz mode and
                                  asking smart and related questions from the pdf
                                  to user and displaying their marks at the end.
                                  '''
                              ])
        response=model.genrate_

general_llm_tool=LLMPromptTool()
tools=[
    QueryEngineTool(query_engine=general_llm_tool,
                    metadata=ToolMetadata(name="general_llm",
                                          description="Provides general information and responces using LLM ")
 )
]
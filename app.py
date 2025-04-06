import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import tempfile
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# Set up the Streamlit app layout
st.title("🤖 RAG-powered Chatbot and Data Analysis App")
st.subheader("Conversation with RAG and Data Analysis")

# Set Gemini API Key as a secret
if "GEMINI_API_KEY" not in st.secrets:
    gemini_api_key = st.text_input("Gemini API Key: ", placeholder="Type your API Key here...", type="password")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        st.success("Gemini API Key successfully configured.")
else:
    # If API key is stored in Streamlit secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    st.success("Using Gemini API Key from Streamlit secrets.")

# Initialize session state for storing chat history, data, and RAG components
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_csv_data" not in st.session_state:
    st.session_state.uploaded_csv_data = None
if "data_dict" not in st.session_state:
    st.session_state.data_dict = None
if "transactions" not in st.session_state:
    st.session_state.transactions = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []

# Display previous chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Create tabs for different file uploads
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Upload Data Dictionary", "Upload Transactions"])

with tab1:
    st.subheader("Upload CSV for Analysis")
    uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_csv is not None:
        try:
            st.session_state.uploaded_csv_data = pd.read_csv(uploaded_csv)
            st.success("CSV file successfully uploaded and read.")
            st.write("### CSV Data Preview")
            st.dataframe(st.session_state.uploaded_csv_data.head())
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")

with tab2:
    st.subheader("Upload Data Dictionary")
    uploaded_dict = st.file_uploader("Choose a JSON data dictionary file", type=["json"], key="dict_uploader")
    if uploaded_dict is not None:
        try:
            st.session_state.data_dict = json.load(uploaded_dict)
            st.success("Data dictionary file successfully uploaded and read.")
            st.write("### Data Dictionary Preview")
            st.json(st.session_state.data_dict)
        except Exception as e:
            st.error(f"An error occurred while reading the data dictionary file: {e}")

with tab3:
    st.subheader("Upload Transactions")
    uploaded_transactions = st.file_uploader("Choose a CSV or JSON transactions file", type=["csv", "json"], key="transactions_uploader")
    if uploaded_transactions is not None:
        try:
            file_extension = uploaded_transactions.name.split('.')[-1].lower()
            if file_extension == 'csv':
                st.session_state.transactions = pd.read_csv(uploaded_transactions)
                st.write("### Transactions Preview (CSV)")
                st.dataframe(st.session_state.transactions.head())
            elif file_extension == 'json':
                st.session_state.transactions = json.load(uploaded_transactions)
                st.write("### Transactions Preview (JSON)")
                st.json(st.session_state.transactions[:5] if isinstance(st.session_state.transactions, list) else st.session_state.transactions)
            st.success("Transactions file successfully uploaded and read.")
        except Exception as e:
            st.error(f"An error occurred while reading the transactions file: {e}")

# Build RAG system if all required components are present
build_rag = st.button("Build RAG System")
if build_rag and os.environ.get("GOOGLE_API_KEY"):
    with st.spinner("Building RAG system..."):
        try:
            # Initialize empty documents list
            documents = []
            
            # Process CSV data if available
            if st.session_state.uploaded_csv_data is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                    st.session_state.uploaded_csv_data.to_csv(temp_csv.name, index=False)
                    loader = CSVLoader(file_path=temp_csv.name)
                    documents.extend(loader.load())
                os.unlink(temp_csv.name)
            
            # Process data dictionary if available
            if st.session_state.data_dict is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_dict:
                    json.dump(st.session_state.data_dict, temp_dict)
                    temp_dict.flush()
                    # Use JSON loader with custom jq expression to handle different JSON structures
                    loader = JSONLoader(
                        file_path=temp_dict.name,
                        jq_schema='.',
                        text_content=False
                    )
                    documents.extend(loader.load())
                os.unlink(temp_dict.name)
            
            # Process transactions if available
            if st.session_state.transactions is not None:
                if isinstance(st.session_state.transactions, pd.DataFrame):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_trans:
                        st.session_state.transactions.to_csv(temp_trans.name, index=False)
                        loader = CSVLoader(file_path=temp_trans.name)
                        documents.extend(loader.load())
                    os.unlink(temp_trans.name)
                else:  # JSON data
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_trans:
                        json.dump(st.session_state.transactions, temp_trans)
                        temp_trans.flush()
                        loader = JSONLoader(
                            file_path=temp_trans.name,
                            jq_schema='.',
                            text_content=False
                        )
                        documents.extend(loader.load())
                    os.unlink(temp_trans.name)
            
            if documents:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings and vectorstore
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Create conversational chain
                llm = ChatGoogleGenerativeAI(model="gemini-pro")
                template = """
                You are a helpful assistant that has access to data and can answer questions based on that data.
                Use the following context to answer the question. If you don't know the answer, just say that you don't know.
                
                Context: {context}
                
                Question: {question}
                
                Conversation history: {chat_history}
                """
                
                PROMPT = PromptTemplate(
                    input_variables=["context", "question", "chat_history"],
                    template=template,
                )
                
                st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.retriever,
                    combine_docs_chain_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
                
                st.success("RAG system built successfully! You can now ask questions about your data.")
            else:
                st.warning("No documents were loaded. Please upload at least one file to build the RAG system.")
                
        except Exception as e:
            st.error(f"An error occurred while building the RAG system: {e}")

# Analysis options
analyze_choice = st.radio(
    "Choose analysis method:",
    ["Direct Question", "Data Analysis", "RAG-powered Question"],
    horizontal=True
)

# User input and response generation
if user_input := st.chat_input("Type your message here..."):
    # Store and display user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)
    
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("Please configure the Gemini API Key to enable responses.")
    else:
        try:
            if analyze_choice == "Direct Question":
                # Direct question to Gemini without data context
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(user_input)
                bot_response = response.text
                
            elif analyze_choice == "Data Analysis" and st.session_state.uploaded_csv_data is not None:
                # Analyze the CSV data with Gemini
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                model = genai.GenerativeModel("gemini-pro")
                
                # Create a description of the data for the model
                data_description = st.session_state.uploaded_csv_data.describe().to_string()
                columns_info = st.session_state.uploaded_csv_data.dtypes.to_string()
                sample_data = st.session_state.uploaded_csv_data.head(5).to_string()
                
                prompt = f"""
                Analyze the following dataset and provide insights based on the user's question: "{user_input}"
                
                Dataset information:
                Columns and types: {columns_info}
                
                Statistical summary:
                {data_description}
                
                Sample data:
                {sample_data}
                """
                
                response = model.generate_content(prompt)
                bot_response = response.text
                
            elif analyze_choice == "RAG-powered Question" and st.session_state.qa_chain is not None:
                # Use RAG to answer questions
                st.session_state.langchain_messages.append(user_input)
                result = st.session_state.qa_chain(
                    {"question": user_input, "chat_history": [(st.session_state.langchain_messages[i], st.session_state.langchain_messages[i+1]) for i in range(0, len(st.session_state.langchain_messages)-1, 2)] if len(st.session_state.langchain_messages) > 1 else []}
                )
                bot_response = result["answer"]
                st.session_state.langchain_messages.append(bot_response)
                
                # Display source documents if available
                if "source_documents" in result:
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}**")
                            st.code(doc.page_content)
            else:
                if analyze_choice == "RAG-powered Question" and st.session_state.qa_chain is None:
                    bot_response = "RAG system is not built yet. Please upload files and click 'Build RAG System'."
                else:
                    bot_response = "Please upload a CSV file first to enable analysis."
            
            # Store and display the bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
            
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

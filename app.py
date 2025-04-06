import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Set up the Streamlit app layout
st.title("🤖 Chatbot for Transaction Data Analysis")
st.subheader("Ask Questions About Your Transactions")

# Capture Gemini API Key
if "GEMINI_API_KEY" not in st.secrets:
    gemini_api_key = st.text_input("Gemini API Key: ", placeholder="Type your API Key here...", type="password")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
else:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    st.success("Using Gemini API Key from Streamlit secrets.")

# Initialize the Gemini Model
model = None
if os.environ.get("GOOGLE_API_KEY"):
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        st.success("Gemini API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")

# Initialize session state for storing chat history and data
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize with an empty list

if "data_dict" not in st.session_state:
    st.session_state.data_dict = None  # Placeholder for data dictionary (CSV)

if "transactions" not in st.session_state:
    st.session_state.transactions = None  # Placeholder for transactions data (CSV)

# Display previous chat history using st.chat_message
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Create tabs for file uploads
tab1, tab2 = st.tabs(["Upload Data Dictionary", "Upload Transactions"])

with tab1:
    st.subheader("Upload Data Dictionary (CSV)")
    uploaded_dict = st.file_uploader("Choose a CSV data dictionary file", type=["csv"], key="dict_uploader")
    if uploaded_dict is not None:
        try:
            st.session_state.data_dict = pd.read_csv(uploaded_dict)
            st.success("Data dictionary successfully uploaded and read as CSV.")
            st.write("### Data Dictionary Preview")
            st.dataframe(st.session_state.data_dict.head())
        except Exception as e:
            st.error(f"An error occurred while reading the data dictionary: {e}")

with tab2:
    st.subheader("Upload Transactions (CSV)")
    uploaded_trans = st.file_uploader("Choose a CSV file for transactions", type=["csv"], key="trans_uploader")
    if uploaded_trans is not None:
        try:
            st.session_state.transactions = pd.read_csv(uploaded_trans)
            st.write("### Transactions Preview")
            st.dataframe(st.session_state.transactions.head())
            st.success("Transactions file successfully uploaded and read.")
        except Exception as e:
            st.error(f"An error occurred while reading the transactions file: {e}")

# Capture user input and generate bot response
if user_input := st.chat_input("Ask a question about your transactions..."):
    # Store and display user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)
    
    if model:
        try:
            if st.session_state.transactions is not None:
                # Prepare RAG context from Transactions and Data Dictionary
                context = []
                
                if st.session_state.data_dict is not None:
                    context.append(f"Data Dictionary:\n{st.session_state.data_dict.to_string()}")
                
                if st.session_state.transactions is not None:
                    context.append(f"Transactions Sample (limited to first 10 rows):\n{st.session_state.transactions.head(10).to_string()}")
                
                # If no data dictionary, still provide transaction data
                if not context:
                    context.append("No additional context available.")
                
                prompt = f"""
                You are a chatbot designed to answer questions about transaction data. 
                Use the following context to provide accurate and concise answers to the user's question: "{user_input}"
                
                Context:
                {"".join(context)}
                
                If the question cannot be answered with the provided data, say so clearly.
                """
                
                # Generate AI response with RAG
                response = model.generate_content(prompt)
                bot_response = response.text
            else:
                bot_response = "Please upload a Transactions CSV file to proceed."
            
            # Store and display the bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    else:
        st.warning("Please configure the Gemini API Key to enable chat responses.")

# Add instructions in the sidebar
st.sidebar.markdown("## How to use this app")
st.sidebar.markdown("""
1. Enter your Gemini API Key
2. Upload your Data Dictionary (CSV) - optional, provides context
3. Upload your Transactions (CSV) - required
4. Ask questions about your transaction data
""")

# Display app status in the sidebar
st.sidebar.markdown("## App Status")
status_items = []
if os.environ.get("GOOGLE_API_KEY"):
    status_items.append("✅ API Key configured")
else:
    status_items.append("❌ API Key not configured")

if st.session_state.data_dict is not None:
    status_items.append("✅ Data dictionary loaded")
else:
    status_items.append("❌ No data dictionary")

if st.session_state.transactions is not None:
    status_items.append("✅ Transactions loaded")
else:
    status_items.append("❌ No transactions data")

for item in status_items:
    st.sidebar.markdown(item)

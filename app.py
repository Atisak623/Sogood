import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Set up the Streamlit app layout
st.title("🤖 My Chatbot and Data Analysis App")
st.subheader("Conversation and Data Analysis")

# Capture Gemini API Key
if "GEMINI_API_KEY" not in st.secrets:
    gemini_api_key = st.text_input("Gemini API Key: ", placeholder="Type your API Key here...", type="password")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
else:
    # Use API key from Streamlit secrets if available
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    st.success("Using Gemini API Key from Streamlit secrets.")

# Initialize the Gemini Model
model = None
if os.environ.get("GOOGLE_API_KEY"):
    try:
        # Configure Gemini with the provided API Key
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        st.success("Gemini API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")

# Initialize session state for storing chat history and data
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize with an empty list

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None  # Placeholder for uploaded CSV data

if "data_dict" not in st.session_state:
    st.session_state.data_dict = None  # Placeholder for data dictionary

if "transactions" not in st.session_state:
    st.session_state.transactions = None  # Placeholder for transactions data

# Display previous chat history using st.chat_message
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Create tabs for different file uploads
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Upload Data Dictionary", "Upload Transactions"])

with tab1:
    st.subheader("Upload CSV for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        try:
            # Load the uploaded CSV file
            st.session_state.uploaded_data = pd.read_csv(uploaded_file)
            st.success("CSV file successfully uploaded and read.")
            # Display the content of the CSV
            st.write("### Uploaded Data Preview")
            st.dataframe(st.session_state.uploaded_data.head())
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

with tab2:
    st.subheader("Upload Data Dictionary")
    uploaded_dict = st.file_uploader("Choose a JSON data dictionary file", type=["json"], key="dict_uploader")
    if uploaded_dict is not None:
        try:
            import json
            st.session_state.data_dict = json.load(uploaded_dict)
            st.success("Data dictionary successfully uploaded and read.")
            st.write("### Data Dictionary Preview")
            st.json(st.session_state.data_dict)
        except Exception as e:
            st.error(f"An error occurred while reading the data dictionary: {e}")

with tab3:
    st.subheader("Upload Transactions")
    uploaded_trans = st.file_uploader("Choose a CSV or JSON file for transactions", type=["csv", "json"], key="trans_uploader")
    if uploaded_trans is not None:
        try:
            file_extension = uploaded_trans.name.split('.')[-1].lower()
            if file_extension == 'csv':
                st.session_state.transactions = pd.read_csv(uploaded_trans)
                st.write("### Transactions Preview (CSV)")
                st.dataframe(st.session_state.transactions.head())
            else:  # JSON
                import json
                st.session_state.transactions = json.load(uploaded_trans)
                st.write("### Transactions Preview (JSON)")
                st.json(st.session_state.transactions)
            st.success("Transactions file successfully uploaded and read.")
        except Exception as e:
            st.error(f"An error occurred while reading the transactions file: {e}")

# Checkbox for indicating data analysis need
analyze_data_checkbox = st.checkbox("Analyze Data with AI")

# Analysis type selection
analysis_type = st.radio(
    "Choose analysis method:",
    ["Basic Analysis", "RAG-powered Analysis"],
    horizontal=True
)

# Capture user input and generate bot response
if user_input := st.chat_input("Type your message here..."):
    # Store and display user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)
    
    # Determine if user input is a request for data analysis and the checkbox is selected
    if model:
        try:
            if (st.session_state.uploaded_data is not None or 
                st.session_state.data_dict is not None or 
                st.session_state.transactions is not None) and analyze_data_checkbox:
                
                # Check if user requested data analysis or insights
                if "analyze" in user_input.lower() or "insight" in user_input.lower() or "data" in user_input.lower():
                    if analysis_type == "Basic Analysis" and st.session_state.uploaded_data is not None:
                        # Basic analysis of CSV data
                        data_description = st.session_state.uploaded_data.describe().to_string()
                        columns_info = st.session_state.uploaded_data.dtypes.to_string()
                        sample_data = st.session_state.uploaded_data.head(5).to_string()
                        
                        prompt = f"""
                        Analyze the following dataset and provide insights based on the user's question: "{user_input}"
                        
                        Dataset information:
                        Columns and types: {columns_info}
                        
                        Statistical summary:
                        {data_description}
                        
                        Sample data:
                        {sample_data}
                        """
                        
                        # Generate AI response for the data analysis
                        response = model.generate_content(prompt)
                        bot_response = response.text
                    elif analysis_type == "RAG-powered Analysis":
                        # Combine all available data for context
                        context = []
                        
                        if st.session_state.uploaded_data is not None:
                            context.append(f"CSV Data Summary:\n{st.session_state.uploaded_data.describe().to_string()}")
                            context.append(f"CSV Sample:\n{st.session_state.uploaded_data.head(5).to_string()}")
                        
                        if st.session_state.data_dict is not None:
                            import json
                            context.append(f"Data Dictionary:\n{json.dumps(st.session_state.data_dict, indent=2)}")
                        
                        if st.session_state.transactions is not None:
                            if isinstance(st.session_state.transactions, pd.DataFrame):
                                context.append(f"Transactions Sample:\n{st.session_state.transactions.head(5).to_string()}")
                            else:
                                import json
                                context.append(f"Transactions Data:\n{json.dumps(st.session_state.transactions, indent=2)[:1000]}")
                        
                        prompt = f"""
                        Based on the following data context, please answer this question: "{user_input}"
                        
                        Context:
                        {"".join(context)}
                        """
                        
                        # Generate AI response with combined context
                        response = model.generate_content(prompt)
                        bot_response = response.text
                    else:
                        bot_response = "Please upload a CSV file for basic analysis."
                else:
                    # Normal conversation with the bot
                    response = model.generate_content(user_input)
                    bot_response = response.text
            elif not analyze_data_checkbox:
                # Respond that analysis is not enabled if the checkbox is not selected
                bot_response = "Data analysis is disabled. Please select the 'Analyze Data with AI' checkbox to enable analysis."
            else:
                # Respond with a message to upload a file if not yet done
                bot_response = "Please upload data files first (CSV, Data Dictionary, or Transactions), then ask me to analyze them."
            
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
2. Upload your data files (CSV, Data Dictionary, Transactions)
3. Check the 'Analyze Data with AI' checkbox
4. Choose your analysis method
5. Ask questions about your data
""")

# Display app status in the sidebar
st.sidebar.markdown("## App Status")
status_items = []
if os.environ.get("GOOGLE_API_KEY"):
    status_items.append("✅ API Key configured")
else:
    status_items.append("❌ API Key not configured")

if st.session_state.uploaded_data is not None:
    status_items.append("✅ CSV data loaded")
else:
    status_items.append("❌ No CSV data")

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

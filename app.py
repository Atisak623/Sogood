import streamlit as st
import pandas as pd
import google.generativeai as genai

# ตั้งค่า API Key
GEMINI_API_KEY = "gemini_key_api"
genai.configure(api_key=GEMINI_API_KEY)

# ตั้งค่าโมเดล
model = genai.GenerativeModel('gemini-2.0-lite')  # ปรับชื่อโมเดลตามที่มีจริง

# ฟังก์ชันสำหรับเรียก Gemini API
def get_gemini_response(prompt):
    response = model.generate_content(prompt)
    return response.text

# ส่วนหัวของแอป
st.title("Transaction & Data Dictionary Analyzer")

# อัปโหลดไฟล์ CSV
st.subheader("Upload CSV Files")
data_dict_file = st.file_uploader("Upload Data Dictionary (CSV)", type="csv")
transaction_file = st.file_uploader("Upload Transaction Data (CSV)", type="csv")

# ตัวแปรสำหรับเก็บข้อมูล
data_dict_df = None
transaction_df = None

# อ่านไฟล์เมื่ออัปโหลด
if data_dict_file is not None:
    data_dict_df = pd.read_csv(data_dict_file)
    st.subheader("Data Dictionary Preview")
    st.write(data_dict_df)

if transaction_file is not None:
    transaction_df = pd.read_csv(transaction_file)
    st.subheader("Transaction Data Preview")
    st.write(transaction_df.head())

# วิเคราะห์ข้อมูลเมื่อมีทั้งสองไฟล์
if data_dict_df is not None and transaction_df is not None:
    # สร้าง prompt สำหรับ Gemini
    prompt = """
    You are python code provider.
    I have two CSV files:
    1. Data Dictionary: Contains column descriptions
    2. Transaction: Contains sales data
    
    Data Dictionary:
    {data_dict}
    
    Transaction sample:
    {transaction}
    
    Please provide:
    1. Summary of the data
    2. Your comments and observations
    """
    
    # แปลงข้อมูลเป็น string เพื่อใส่ใน prompt
    data_dict_str = data_dict_df.to_string()
    transaction_str = transaction_df.head().to_string()
    
    final_prompt = prompt.format(
        data_dict=data_dict_str,
        transaction=transaction_str
    )
    
    # เรียก Gemini API
    with st.spinner("Analyzing with Gemini..."):
        gemini_response = get_gemini_response(final_prompt)
    
    # แสดงผลลัพธ์
    st.subheader("Analysis Results")
    st.write("### Gemini Response:")
    st.write(gemini_response)

# เพิ่มคำแนะนำการใช้งาน
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload your Data Dictionary CSV file
2. Upload your Transaction CSV file
3. Wait for the analysis from Gemini model
""")

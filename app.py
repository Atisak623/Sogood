import streamlit as st
import pandas as pd
import google.generativeai as genai

# ตั้งค่าหน้าแอป
st.title('Chat with Database using Gemini')

# ตั้งค่า Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transaction_df" not in st.session_state:
    st.session_state.transaction_df = None
if "data_dict_df" not in st.session_state:
    st.session_state.data_dict_df = None

# ตั้งค่า Gemini API
try:
    key = st.secrets['gemini_api_key']
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    # แสดงประวัติการแชท
    for message in st.session_state.chat_history:
        role, content = message
        st.chat_message(role).markdown(content)

    # ส่วนอัพโหลดไฟล์ CSV
    with st.sidebar:
        st.header("อัพโหลดข้อมูล")
        
        # อัพโหลด Transaction CSV
        transaction_file = st.file_uploader("อัพโหลดไฟล์ Transaction CSV", type=["csv"])
        if transaction_file is not None:
            try:
                st.session_state.transaction_df = pd.read_csv(transaction_file)
                st.success("อัพโหลดไฟล์ Transaction สำเร็จ!")
                st.write("ตัวอย่างข้อมูล:")
                st.dataframe(st.session_state.transaction_df.head(2))
                st.session_state.example_record = st.session_state.transaction_df.head(2).to_string()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

        # อัพโหลด Data Dictionary CSV หรือสร้างอัตโนมัติ
        data_dict_file = st.file_uploader("อัพโหลดไฟล์ Data Dictionary CSV (ถ้ามี)", type=["csv"])
        if data_dict_file is not None:
            try:
                st.session_state.data_dict_df = pd.read_csv(data_dict_file)
                st.success("อัพโหลดไฟล์ Data Dictionary สำเร็จ!")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
        elif st.session_state.transaction_df is not None and st.button("สร้าง Data Dictionary อัตโนมัติ"):
            data = []
            for column in st.session_state.transaction_df.columns:
                data_type = str(st.session_state.transaction_df[column].dtype)
                data.append({
                    "column_name": column,
                    "data_type": data_type,
                    "description": f"Column {column} with type {data_type}"
                })
            st.session_state.data_dict_df = pd.DataFrame(data)
            st.success("สร้าง Data Dictionary อัตโนมัติสำเร็จ!")
            st.dataframe(st.session_state.data_dict_df)

        if st.session_state.data_dict_df is not None:
            try:
                st.session_state.data_dict_text = '\n'.join([
                    f"- {row['column_name']}: {row['data_type']}. {row['description']}" 
                    for _, row in st.session_state.data_dict_df.iterrows()
                ])
                st.success("พร้อมใช้งานแล้ว! ถามคำถามเกี่ยวกับข้อมูลได้เลย")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสร้าง data_dict_text: {e}")

    # ช่องสำหรับรับคำถามจากผู้ใช้
    if user_input := st.chat_input("ถามคำถามเกี่ยวกับข้อมูล..."):
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        if st.session_state.transaction_df is not None and hasattr(st.session_state, 'data_dict_text'):
            try:
                df_name = "transaction_df"
                prompt = f"""
                You are a helpful Python code generator.
                Your goal is to write Python code snippets based on the user's question
                and the provided DataFrame information.
                Here's the context:

                **User Question:**
                {user_input}

                **DataFrame Name:**
                {df_name}

                **DataFrame Details:**
                {st.session_state.data_dict_text}

                **Sample Data (Top 2 Rows):**
                {st.session_state.example_record}

                **Instructions:**
                1. Write Python code that addresses the user's question by querying or manipulating the DataFrame.
                2. **Crucially, use the `exec()` function to execute the generated code.**
                3. Do not import pandas
                4. Change date column type to datetime if needed
                5. **Store the result of the executed code in a variable named `ANSWER`.**
                This variable should hold the answer to the user's question (e.g., a filtered DataFrame, a calculated value, etc.).
                6. Assume the DataFrame is already loaded into a pandas DataFrame object named '{df_name}'. Do not include code to load the DataFrame.
                7. Keep the generated code concise and focused on answering the question.
                """

                # สร้างโค้ด Python ด้วย Gemini
                response = model.generate_content(prompt)
                
                # แก้ไขโค้ดโดยลบเครื่องหมาย ``` และจัดการให้เหมาะสม
                query = response.text.replace("```python", "").replace("```", "").strip()

                # แสดงโค้ดที่ AI สร้าง
                with st.expander("ดูโค้ด Python ที่ AI สร้าง"):
                    st.code(query)

                # รันโค้ดเพื่อหาคำตอบ
                transaction_df = st.session_state.transaction_df
                ANSWER = None
                exec(query)

                # สร้าง prompt เพื่ออธิบายผลลัพธ์
                explain_prompt = f"""
                The user asked: {user_input}
                Here is the result: {ANSWER}
                Please answer the question and summarize the answer in Thai language.
                Make it easy to understand.
                """

                explain_response = model.generate_content(explain_prompt)
                bot_response = explain_response.text

            except Exception as e:
                bot_response = f"เกิดข้อผิดพลาดในการวิเคราะห์หรือรันโค้ด: {e}"
        else:
            bot_response = "กรุณาอัพโหลดไฟล์ Transaction.csv และสร้าง Data Dictionary ก่อนถามคำถาม"

        st.session_state.chat_history.append(("assistant", bot_response))
        st.chat_message("assistant").markdown(bot_response)

except Exception as e:
    st.error(f'เกิดข้อผิดพลาดในการเริ่มต้น: {e}')

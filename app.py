import joblib
import pandas as pd

def load_trained_model():
    """Tải model đã huấn luyện từ file .pkl"""
    try:
        return joblib.load('loan_model.pkl')
    except:
        return None

def predict_eligibility(model, input_data):
    """Thực hiện dự báo và trả về kết quả + xác suất"""
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Lấy xác suất của nhãn dự đoán được (0 hoặc 1)
    conf = probability[0][prediction[0]]
    return prediction[0], conf
  import pandas as pd

def format_input(age, income, loan_amount, credit_score):
    """Chuyển đổi dữ liệu từ giao diện thành DataFrame cho model"""
    data = {
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score]
    }
    return pd.DataFrame(data)
  import streamlit as st
from model_utils import load_trained_model, predict_eligibility
from data_utils import format_input

# Load model
model = load_trained_model()

st.title("💳 SmartLoan AI")

# Sidebar
age = st.sidebar.slider("Tuổi", 18, 65, 30)
income = st.sidebar.number_input("Thu nhập (VNĐ)", 5000000)
loan_amount = st.sidebar.number_input("Số tiền vay (VNĐ)", 10000000)
credit_score = st.sidebar.slider("Điểm tín dụng CIC", 300, 850, 600)

if st.button("Phân tích"):
    if model is None:
        st.error("Chưa có model! Hãy chạy script train_model.py trước.")
    else:
        # Gọi hàm từ các module đã tách
        input_df = format_input(age, income, loan_amount, credit_score)
        result, confidence = predict_eligibility(model, input_df)
        
        if result == 1:
            st.success(f"Kết quả: PHÊ DUYỆT (Tin cậy: {confidence*100:.2f}%)")
        else:
            st.error(f"Kết quả: TỪ CHỐI (Tin cậy: {confidence*100:.2f}%)")
          

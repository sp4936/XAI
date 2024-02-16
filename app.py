import streamlit as st
import pickle

# Load the saved model
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Streamlit web app
st.title('Bank Customer Churn Prediction')

st.sidebar.header('User Input Features')

# Define input features
credit_score = st.sidebar.slider('Credit Score', 300, 850, 600)
age = st.sidebar.slider('Age', 18, 100, 30)
tenure = st.sidebar.slider('Tenure', 0, 10, 5)
balance = st.sidebar.number_input('Balance', 0.0, 250000.0, 50000.0)
products_number = st.sidebar.slider('Number of Products', 1, 4, 2)
credit_card = st.sidebar.selectbox('Has Credit Card', ('Yes', 'No'))
active_member = st.sidebar.selectbox('Is Active Member', ('Yes', 'No'))
estimated_salary = st.sidebar.number_input('Estimated Salary', 0.0, 250000.0, 50000.0)

# Map categorical inputs to numeric values
credit_card = 1 if credit_card == 'Yes' else 0
active_member = 1 if active_member == 'Yes' else 0

# Make sure you provide all 9 features in the input data
# Include the missing feature that the model expects
input_data = [[credit_score, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, 0]]

if st.button('Predict'):
    # Make predictions
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.subheader('Prediction Result')
    if prediction[0] == 0:
        st.write('The customer is not likely to churn.')
    else:
        st.write('The customer is likely to churn.')

    st.subheader('Prediction Probability')
    st.write(f'Churn Probability: {prediction_proba[0]:.2f}')

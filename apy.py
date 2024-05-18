import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model for diabetes prediction
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Load the trained model for skin disease classification
skin_disease_model = load_model('skinmod1.h5')
skin_disease_model_2 = load_model('skinmod2.h5')

# Define categories for skin disease classification
categories = ['normal', 'skin_disease', 'NotSkinImages']
categories_2 = ['Hives', 'Cold Sore', 'psoriasis', 'cellulitis',
                'ringworm', 'lupus', 'acne', 'eczema', 'dry skin', 'dermatitis']

if 'initial_screening_completed' not in st.session_state:
    st.session_state['initial_screening_completed'] = False
# Sidebar for navigation
with st.sidebar: #'''Health Prediction System'''
    selected = option_menu('AUTOMATED DERMATOLOGICAL CONDITION IDENTIFICATION SYSTEM', ['Initial Screening', 'Get to Know the Skin Disease', 'Diabetes Prediction', 'Skin Health Recommendations'], menu_icon='hospital-fill', icons=['check', 'lightbulb', 'activity', 'star'], default_index=0)

# Initial Screening Page
if selected == 'Initial Screening':
    st.title("Initial Screening Page")
    st.markdown("Upload an image of skin, and get a disease classification.")

    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # Resize the image to a smaller size
        resized_image = image.resize((300, 300))  # Adjust the dimensions as needed
        uploaded = st.image(resized_image, caption='Uploaded Image', use_column_width=True)  # Display the image
        col1, col2, col3 = st.columns([1, 2, 1])
        if col1.button('Clear'):
            uploaded_image = None  # Reset uploaded image
            uploaded.empty()  # Remove the displayed image
        if col3.button('Submit'):
            # Function to predict skin disease
            
            def predict_skin_disease(image_array):
                img_size = 112
                # Resize the image to the required size
                img_array = cv2.resize(image_array, (img_size, img_size))
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
                prediction = skin_disease_model.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                predicted_category = categories[predicted_class]
                return predicted_category, confidence

            predicted_category, confidence = predict_skin_disease(np.array(resized_image))
            result_text = f"<div style='text-align: center;'><b>Predicted category: {predicted_category}</b><br>Confidence: {confidence:.2f}</div>"
            st.markdown(result_text, unsafe_allow_html=True)
            st.session_state['initial_screening_completed'] = True

# Get to Know the Skin Disease Page
elif selected == 'Get to Know the Skin Disease':
    if not st.session_state['initial_screening_completed']:
    # Display a warning message if initial screening was not completed
        st.warning("Please complete the Initial Screening module first.")
    else:
        st.title("Get to Know the Skin Disease")
    st.write("Upload an image of skin, and get a disease classification.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # Resize the image to a smaller size
        resized_image = image.resize((100, 100))  # Adjust the dimensions as needed
        uploaded = st.image(resized_image, caption='Uploaded Image', use_column_width=True)  # Display the image
        col1, col2, col3 = st.columns([1, 2, 1])

        # Add Submit and Clear Buttons
        submit_button = col1.button("Submit")
        clear_button = col3.button("Clear")

        if submit_button:
            # Function to predict skin disease
            def predict_skin_disease_2(image_array):
                img_size = 112
                # Resize the image to the required size
                img_array = cv2.resize(image_array, (img_size, img_size))
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
                prediction = skin_disease_model_2.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                predicted_category = categories_2[predicted_class]
                return predicted_category, confidence

            # Process image and display result
            def process_image(image_array):
                predicted_category, confidence = predict_skin_disease_2(image_array)
    
                if confidence < 0.45:
                    result_text = "<div style='text-align: center;'><b>Predicted category:</b> Unknown skin disease<br>"
                    result_text += "Please contact a nearby dermatologist for further evaluation.</div>"
                else:
                    result_text = f"<div style='text-align: center;'><b>Predicted category:</b> {predicted_category}<br>"
                    result_text += f"<b>Confidence:</b> {confidence:.2f}</div>"
                return result_text


            result = process_image(np.array(image))
            st.write(result, unsafe_allow_html=True)

        if clear_button:
            uploaded_image = None  # Reset uploaded image
            uploaded.empty()
            

    

    # Add content for this page here

# Diabetes Prediction Page
elif selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to numerical values
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
                st.error(diab_diagnosis)  # Display in red if diabetic
            else:
                diab_diagnosis = 'The person is not diabetic'
                st.success(diab_diagnosis)  # Display in green if not diabetic

        except ValueError:
            st.warning("Please enter valid numerical values for all input fields.")

# Skin Health Recommendations Page
elif selected == 'Skin Health Recommendations':
    st.title("Skin Health Recommendations for Diabetics and Non-Diabetics")

    # Define the data for skin diseases, diabetic options, medical precautions, and food recommendations
    skin_diseases = [
        "Acne", "Cellulitis", "Cold Sore", "Dermatitis", "Dry Skin",
        "Eczema", "Hives", "Lupus", "Psoriasis", "Ringworm"
    ]

    diabetic_options = ["Yes", "No"]

    medical_precautions = {
        "Acne": {"No": "Avoid alcohol in skincare, Regular face wash.", "Yes": "Don't pop acne."},
        "Cellulitis": {"No": "Hand hygiene, wound care.", "Yes": "Address conditions such as chronic edema (swelling), vascular disease."},
        "Cold Sore": {"No": "skin contact with people, Don't share towels.", "Yes": "Keep the affected area clean, moisturized, and dry."},
        "Dermatitis": {"No": "Use moisturizer, Wear protective clothing.", "Yes": "Pat skin gently, focus."},
        "Dry Skin": {"No": "Warm water for bathing, Sufficient cleanser application.", "Yes": "Clean, dry skin regimen."},
        "Eczema": {"No": "Monitor water temperature, Daily moisturize habit.", "Yes": "Prevent rubbing, cover itching."},
        "Hives": {"No": "Sun protection, loose clothing.", "Yes": "Choose lukewarm water."},
        "Lupus": {"No": "7 hours of sleep each night, avoid sunlight.", "Yes": "Wear sunscreen, hats, clothing."},
        "Psoriasis": {"No": "Trim nails, avoid scratching.", "Yes": "Gentle skincare, moisturize often."},
        "Ringworm": {"No": "Keep your skin clean and dry , Wear Airy Footwear.", "Yes": "Avoid sharing items, Stay off moist surfaces."}
    }

    food_recommendations = {
        "Acne": {"No": "Carrots, Apricots, Tomatoes", "Yes": "Legumes, nuts, seeds."},
        "Cellulitis": {"No": "Baked beans, whole grains.", "Yes": "Broccoli and cabbage."},
        "Cold Sore": {"No": "milk, cheese.", "Yes": "juice,soup."},
        "Dermatitis": {"No": "wheat, oats, milk.", "Yes": "Broccoli and cauliflower."},
        "Dry Skin": {"No": "egg yolk, spinach.", "Yes": "Leafy Vegetables, Frozen Fruits."},
        "Eczema": {"No": "Drink more water, Cooking with olive oil.", "Yes": "Vegetable oils, Dry Fruits."},
        "Hives": {"No": "bread,pasta.", "Yes": "fish, chicken."},
        "Lupus": {"No": "eggs,pasta.", "Yes": "whole-wheat bread and brown rice."},
        "Psoriasis": {"No": "meat,eggs.", "Yes": "Whole grains, Olive oil."},
        "Ringworm": {"No": "Drink plenty of water,brown rice.", "Yes": "yogurt, kefir."},
    }

    # Select box for skin diseases
    selected_disease = st.selectbox("Select Skin Disease:", skin_diseases)

    # Select box for diabetic options
    diabetic_status = st.selectbox("Are you Diabetic?", diabetic_options)

    # Process the selections
    if st.button("Submit"):
        if selected_disease in medical_precautions and diabetic_status in medical_precautions[selected_disease]:
            st.subheader("Medical Precautions:")
            st.write(medical_precautions[selected_disease][diabetic_status])
        else:
            st.write("No medical precautions found for the selected disease and diabetic status.")

        if selected_disease in food_recommendations and diabetic_status in food_recommendations[selected_disease]:
            st.subheader("Food Recommendations:")
            st.write(food_recommendations[selected_disease][diabetic_status])
        else:
            st.write("No food recommendations found for the selected disease and diabetic status.")
# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (only errors)


# # Load the three models
# model_Alex = load_model('alexnet_model.h5')
# model_Alex.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

# model_VGG = load_model('vgg_model.h5')
# model_VGG.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

# model_Res = load_model('resnet_model.h5')
# model_Res.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
# class_labels = ['Other Activities', 'Safe Driving', 'Talking on Phone', 'Texting on Phone', 'Turning']

# st.title('Driver Behavior Detection')

# # Model selection
# model_choice = st.selectbox("Choose a model for prediction", ("AlexNet", "VGG", "ResNet"))

# # File uploader 
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# def predict_behavior(model, img):
#     img_resized = cv2.resize(img, (240, 240))  
#     img_scaled = img_resized / 255.0
#     img_expanded = np.expand_dims(img_scaled, axis=0)

#     prediction = model.predict(img_expanded)
#     predicted_class = class_labels[np.argmax(prediction)]
#     return predicted_class

# if uploaded_file is not None:
    
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, 1)

#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     if model_choice == "AlexNet":
#         st.write("Using **AlexNet** model for prediction...")
#         predicted_class = predict_behavior(model_Alex, img)
#     elif model_choice == "VGG":
#         st.write("Using **VGG** model for prediction...")
#         predicted_class = predict_behavior(model_VGG, img)
#     elif model_choice == "ResNet":
#         st.write("Using **ResNet** model for prediction...")
#         predicted_class = predict_behavior(model_Res, img)

#     st.write(f"Predicted Driver Activity: **{predicted_class}**")
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# Set the correct working directory
os.chdir(r"C:\Users\jaysh\Downloads\Driver-Behavior-Detection-main\Driver-Behavior-Detection-main")

# Define model paths
model_paths = {
    "AlexNet": r"C:\Users\jaysh\Downloads\Driver-Behavior-Detection-main\Driver-Behavior-Detection-main\alexnet_model.h5",
    "VGG": r"C:\Users\jaysh\Downloads\Driver-Behavior-Detection-main\Driver-Behavior-Detection-main\vgg_model.h5",
    "ResNet": r"C:\Users\jaysh\Downloads\Driver-Behavior-Detection-main\Driver-Behavior-Detection-main\resnet_model.h5"
}

# Load models dynamically
models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.error(f"Model file **{name}** not found at: {path}")

# Class labels
class_labels = ['Other Activities', 'Safe Driving', 'Talking on Phone', 'Texting on Phone', 'Turning']

# Streamlit App Title
st.title("🚗 Driver Behavior Detection")

# Model selection
model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))

# File uploader 
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function to preprocess and predict
def predict_behavior(model, img):
    img_resized = cv2.resize(img, (240, 240))  # Resize to match model input size
    img_scaled = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_scaled, axis=0)  # Add batch dimension

    prediction = model.predict(img_expanded)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # Make predictions
    if model_choice in models:
        st.write(f"🔍 Using **{model_choice}** model for prediction...")
        predicted_class = predict_behavior(models[model_choice], img)
        st.success(f"🚦 **Predicted Driver Activity:** {predicted_class}")
    else:
        st.error("❌ Model not found! Please check the file paths.")

# Run the app using: `streamlit run app.py`

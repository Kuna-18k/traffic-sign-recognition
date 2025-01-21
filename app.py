import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Define the labels (ensure they match the number of classes your model was trained on)
labels = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60", 
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No overtaking", "No entry", "Stop", "Yield", 
    "Pedestrian crossing", "Children crossing", "Bicycles crossing", "Beware of ice/snow", 
    "Traffic signal", "No parking", "One way", "Roundabout", "School zone", 
    "Zebra crossing", "Pedestrian crossing", "No U-turn", "No left turn", 
    "No right turn", "Parking", "Speed bump", "Sharp turn", "Slippery road", 
    "Crosswalk", "Dead end", "Intersection", "Bicycle lane", "Traffic light", 
    "Speed limit 40", "No overtaking trucks", "Warning sign", "Left lane ends", 
    "Bus stop", "Traffic light ahead", "Caution", "Pedestrian area"
]

# Streamlit app interface
st.title("🚦 **Traffic Sign Recognition System** 🚦")
st.markdown("""
    ## Welcome to the Traffic Sign Recognition App!  
    Upload an image of a traffic sign, and our model will predict the sign and show its confidence level.  
    The app uses deep learning to recognize various traffic signs.  
    **Simply upload an image and let the model do the rest!**
""")

# Add a custom logo or background image
st.image("path_to_logo_or_image.jpg", width=150)

# Upload image
uploaded_file = st.file_uploader("🖼️ **Choose an Image of a Traffic Sign**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the input size expected by the model (32x32 pixels)
    img_resized = cv2.resize(img, (32, 32)) / 255.0  # Normalize pixel values to range [0, 1]
    
    # Reshape the image for the model input
    img_resized = np.expand_dims(img_resized, axis=(0, -1))  # Add batch dimension and channel dimension

    # Model prediction
    prediction = model.predict(img_resized)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the result
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### **Prediction:** {labels[class_id]}")
    st.markdown(f"### **Confidence:** {confidence:.2f}")

# Footer with contact or additional info
st.markdown("""
    ---
    **Made with CNN by kuna,satyajit**  
    Visit the [GitHub repository](https://github.com/your-repository) for more details!  
    For any inquiries, contact us at [220301120222@cutm.ac.in](mailto:your-kunakandi26@gmail.com).
""")

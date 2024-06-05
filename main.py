import streamlit as st
from PIL import Image
from model import llava_model
from preprocessing import load_image, process_image, create_prompt, ask_image, to_markdown
from translation import transArabic

# Initialize model and tokenizer
tokenizer, model, image_processor, context_len = llava_model()

# Streamlit app
st.title("Image Description and Translation App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Processing...")

    # Process the image
    processed_image = process_image(image, image_processor, model)
    
    # Create prompt
    prompt, _ = create_prompt("Describe the image")
    
    # Get response from the model
    response = ask_image(image, "Describe the image", tokenizer, model)
    
    # Translate the response
    result = transArabic(response)
    
    # Display the response and its translation
    st.write("### Image Description:")
    st.write(response)
    st.write("### Translated Description (Arabic):")
    st.write(result)

    # Convert the result to markdown and display
    markdown_result = to_markdown(result)
    st.markdown(markdown_result)

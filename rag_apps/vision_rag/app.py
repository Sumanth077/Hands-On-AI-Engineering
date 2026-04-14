import streamlit as st
import os
import datetime
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from utils import get_cohere_embedding, gemini_vqa, pdf_to_images, image_to_bytes, find_most_similar

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="VisionRAG: Multimodal Search & VQA", layout="wide")
st.title("VisionRAG: Multimodal Search & Visual Question Answering")

# API Keys from environment variables
cohere_api = os.getenv("COHERE_API_KEY")
gemini_api = os.getenv("GEMINI_API_KEY")

# Initialize session state
if 'items' not in st.session_state:
    st.session_state['items'] = []  # List of loaded images/PDF pages with embeddings

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []  # Chat history

# Sidebar - Content Management
st.sidebar.header("Content Sources")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload Images or PDFs", 
    type=["png", "jpg", "jpeg", "pdf"], 
    accept_multiple_files=True,
    help="Upload images (PNG, JPG, JPEG) or PDF documents"
)

# Process uploaded files
if uploaded_files:
    new_items_count = 0
    existing_names = set((item['name'], item['type']) for item in st.session_state['items'])
    
    for file in uploaded_files:
        if file.type.startswith('image/'):
            key = (file.name, 'image')
            if key not in existing_names:
                img = Image.open(file).convert('RGB')
                st.session_state['items'].append({'type': 'image', 'name': file.name, 'img': img, 'emb': None})
                new_items_count += 1
        elif file.type == 'application/pdf':
            pdf_bytes = file.read()
            pages = pdf_to_images(pdf_bytes)
            for i, page_img in enumerate(pages):
                name = f"{file.name} - Page {i+1}"
                key = (name, 'pdf_page')
                if key not in existing_names:
                    st.session_state['items'].append({'type': 'pdf_page', 'name': name, 'img': page_img, 'emb': None})
                    new_items_count += 1

    st.success(f"Uploaded {len(uploaded_files)} file(s) with {new_items_count} total items.")

# Sidebar - Content Preview
st.sidebar.header("📁 Loaded Content")
if st.session_state['items']:
    st.sidebar.markdown(f"**Total items:** {len(st.session_state['items'])}")
    preview_items = st.session_state['items'][:6]  # Show first 6 items
    
    for item in preview_items:
        with st.sidebar.container():
            small_img = item['img'].resize((120, 120), Image.Resampling.LANCZOS)
            caption = item['name'][:15] + "..." if len(item['name']) > 15 else item['name']
            st.image(small_img, caption=caption, width=120)
    
    if len(st.session_state['items']) > 6:
        st.sidebar.markdown(f"*... and {len(st.session_state['items']) - 6} more items*")
else:
    st.sidebar.info("No content loaded yet.")

# Main Interface - Chat
st.subheader("Chat with Your Visual Data")

# Display conversation history
if st.session_state['conversation']:
    st.markdown("### Conversation History")
    for exchange in st.session_state['conversation']:
        with st.expander(f"Q: {exchange['question']} ({exchange['timestamp']})", expanded=False):
            st.markdown(f"**Question:** {exchange['question']}")
            st.markdown(f"**Answer:** {exchange['answer']}")
            st.markdown(f"**Relevant Image:** {exchange['relevant_image']}")
            if exchange.get('image_display'):
                st.image(exchange['image_display'], caption=exchange['relevant_image'], use_container_width=True)

# Clear conversation button
if st.session_state['conversation']:
    if st.button("🗑️ Clear Conversation"):
        st.session_state['conversation'] = []
        st.rerun()

# Question input form
with st.form("question_form"):
    question = st.text_input("Ask a question about your visual data:")
    submit_button = st.form_submit_button("Send", type="primary")

# Process question submission
if submit_button:
    if not cohere_api or not gemini_api:
        st.error("Please provide both Cohere and Gemini API keys.")
    elif not question:
        st.error("Please enter a question.")
    elif not st.session_state['items']:
        st.error("No content loaded to search.")
    else:
        # Compute embeddings and find relevant image
        with st.spinner("Computing embeddings and searching..."):
            # Generate embeddings for all items if not already done
            for item in st.session_state['items']:
                if item['emb'] is None:
                    if item['type'] in ['image', 'pdf_page']:
                        img_bytes = image_to_bytes(item['img'])
                        item['emb'] = get_cohere_embedding(cohere_api, img_bytes, input_type='image')
            
            # Get question embedding and find most similar image
            q_emb = get_cohere_embedding(cohere_api, question, input_type='text')
            emb_list = [item['emb'] for item in st.session_state['items']]
            idx, sim = find_most_similar(q_emb, emb_list)
            best_item = st.session_state['items'][idx]
            
            # Generate answer using Gemini
            st.info("Generating answer with Gemini...")
            img_bytes = image_to_bytes(best_item['img'])
            answer = gemini_vqa(gemini_api, img_bytes, question)
            
            # Add to conversation history
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            conversation_entry = {
                'question': question,
                'answer': answer,
                'relevant_image': best_item['name'],
                'timestamp': timestamp,
                'image_display': best_item['img'],
                'similarity': sim
            }
            st.session_state['conversation'].append(conversation_entry)
            
            st.rerun() 
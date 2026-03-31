"""
Utility functions for VisionRAG: Multimodal Search & VQA application.
Handles embeddings, VQA, PDF processing, and similarity calculations.
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cohere
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- Cohere Embed-4 ---
def get_cohere_embedding(api_key, input_data, input_type='text'):
    """
    Get embedding for text or image using Cohere Embed-4.
    
    Args:
        api_key (str): Cohere API key
        input_data (str or bytes): Text string or image bytes
        input_type (str): 'text' or 'image'
    
    Returns:
        np.ndarray: Embedding vector
    """
    co = cohere.Client(api_key)
    
    if input_type == 'text':
        response = co.embed(texts=[input_data], model="embed-v4.0")
        return np.array(response.embeddings[0])
    elif input_type == 'image':
        # Convert bytes to base64 data URI
        image_b64 = base64.b64encode(input_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{image_b64}"
        response = co.embed(images=[data_uri], model="embed-v4.0")
        return np.array(response.embeddings[0])
    else:
        raise ValueError("input_type must be 'text' or 'image'")

# --- Google Gemini 2.5 Flash VQA ---
def gemini_vqa(api_key, image_bytes, question):
    """
    Send image and question to Gemini 2.5 Flash for Visual Question Answering.
    
    Args:
        api_key (str): Google Gemini API key
        image_bytes (bytes): Image data
        question (str): Question about the image
    
    Returns:
        str: Generated answer
    """
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    image_b64 = base64.b64encode(image_bytes).decode()
    data = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    {"text": question}
                ]
            }
        ]
    }
    
    response = requests.post(endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} {response.text}"

# --- PDF to Images ---
def pdf_to_images(pdf_bytes):
    """
    Convert PDF bytes to a list of PIL Images (one per page).
    
    Args:
        pdf_bytes (bytes): PDF file data
    
    Returns:
        list: List of PIL Image objects
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    doc.close()
    return images

# --- Image to Bytes ---
def image_to_bytes(img, format='PNG'):
    """
    Convert PIL Image to bytes.
    
    Args:
        img (PIL.Image): Image to convert
        format (str): Image format (default: 'PNG')
    
    Returns:
        bytes: Image data
    """
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

# --- Cosine Similarity ---
def find_most_similar(query_emb, emb_list):
    """
    Find the most similar embedding in a list to the query embedding.
    
    Args:
        query_emb (np.ndarray): Query embedding vector
        emb_list (list): List of embedding vectors to compare against
    
    Returns:
        tuple: (index of most similar embedding, similarity score)
    """
    similarities = cosine_similarity([query_emb], emb_list)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(np.max(similarities))
    
    return best_idx, best_score 
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    st.write(f"Loading generation model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Adjust device_map and torch_dtype based on your hardware
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto", # Use 'cpu' if no GPU or driver issues
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        return None, None 
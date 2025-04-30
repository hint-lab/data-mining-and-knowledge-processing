import json
import streamlit as st

def load_data(filepath):
    """Loads data from the JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.write(f"Loaded {len(data)} articles from {filepath}")
        return data
    except FileNotFoundError:
        st.error(f"Data file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {filepath}")
        return []
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return [] 
import boto3
import torch
import tempfile
import streamlit as st
from safetensors.torch import load_file
from transformers import AutoModelForTokenClassification, AutoConfig

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"],
    )

def load_model_from_s3(model_cls="distilbert-base-uncased",t_type=torch.float16):
    bucket_name = "sounds-like"
    key = "data/model.safetensors"

    s3 = get_s3_client()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3.download_fileobj(bucket_name, key, temp_file)
        temp_path = temp_file.name

    # Load the state_dict from safetensors
    state_dict = load_file(temp_path)

    # Load model config
    config = AutoConfig.from_pretrained(model_cls, torch_dtype=t_type, num_labels=6)

    # Initialize model with config (no weights)
    model = AutoModelForTokenClassification.from_config(config)

    # Load weights manually
    model.load_state_dict(state_dict)

    return model
import boto3
import torch
import tempfile
import streamlit as st

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"],
    )

def load_model_from_s3(
    model_cls="distilbert-base-uncased",
):
    bucket_name = "sounds-like"
    key = "data/model.safetensors"

    s3 = get_s3_client()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3.download_fileobj(bucket_name, key, temp_file)
        temp_path = temp_file.name

    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        model_cls,
        local_files_only=True,
        state_dict=torch.load(temp_path),
    )
    return model
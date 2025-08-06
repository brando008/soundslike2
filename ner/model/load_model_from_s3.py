import boto3
import torch
import tempfile


def load_model_from_s3(
    bucket_name="sounds-like",
    key="models/model.safetensors",  # ← update this if needed
    model_cls="distilbert-base-uncased",
):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3 = boto3.client("s3")
        s3.download_fileobj(bucket_name, key, temp_file)
        temp_path = temp_file.name

    model = AutoModelForTokenClassification.from_pretrained(
        model_cls,
        local_files_only=True,
        state_dict=torch.load(temp_path),
    )
    return model
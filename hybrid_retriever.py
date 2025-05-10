import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.schema import Document

# Configuration via environment variables
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/stsb-roberta-base")


class HybridReranker:
    """
    Cross-encoder reranker that scores document relevance for a query.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device.startswith("cuda"):
            self.model.to(device)
        self.device = device
        print(f"Initialized reranker model: {model_name} on {device}")

    def score(self, query: str, docs: List[Document]) -> List[float]:
        combined = [f"{query}{self.tokenizer.sep_token}{doc.page_content}" for doc in docs]
        inputs = self.tokenizer(
            combined,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        return logits.cpu().tolist()
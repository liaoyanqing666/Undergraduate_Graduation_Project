import os
import base64
import numpy as np
import requests
from abc import ABC, abstractmethod
from typing import List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, Blip2Processor, Blip2Model
import torch

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, inputs: Union[str, List[str]]) -> np.ndarray:
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2", model_dir="model_para/embedding"):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            self.model = SentenceTransformer(model_name)
            self.model.save(model_path)
        else:
            self.model = SentenceTransformer(model_path)

    def embed(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        return self.model.encode(inputs, convert_to_numpy=True)


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, model_name="facebook/opt-2.7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state[:, 0, :]  # [CLS]向量或第一个token向量
        return outputs.cpu().numpy()


class APIEmbedder(BaseEmbedder):
    def __init__(self, api_url, embedding_type="text"):
        self.api_url = api_url
        self.embedding_type = embedding_type

    def embed(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if self.embedding_type == "text":
            payload = {"texts": inputs}
        elif self.embedding_type == "image":
            payload = {"images": [self._image_to_base64(p) for p in inputs]}
        else:
            raise ValueError("Unsupported embedding type")

        resp = requests.post(self.api_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"API Error: {resp.status_code}, {resp.text}")
        return np.array(resp.json()["embeddings"])

    def _image_to_base64(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()


class BLIP2Embedder(BaseEmbedder):
    def __init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        self.text_model = AutoModel.from_pretrained("facebook/opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.vision_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

    def embed(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]

        if os.path.exists(inputs[0]):
            # image embedding
            images = [Image.open(p).convert("RGB") for p in inputs]
            processed = self.processor(images=images, return_tensors="pt")
            with torch.no_grad():
                vision_out = self.vision_model(**processed).vision_model_output.last_hidden_state[:, 0, :]
            return vision_out.cpu().numpy()
        else:
            # text embedding
            encoded = self.text_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = self.text_model(**encoded).last_hidden_state[:, 0, :]
            return output.cpu().numpy()


import os
import pickle
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter


class MultimodalRAG:
    def __init__(self, embedding_model, pair_embed_mode="both"):
        assert pair_embed_mode in ["text", "image", "both", None]
        self.embedding_model = embedding_model
        self.pair_embed_mode = pair_embed_mode
        self.index = None
        self.documents = []  # [(embedding_content, return_content)]

    def _init_index(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def _add_embeddings(self, embeddings, return_contents):
        if self.index is None:
            self._init_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents.extend(return_contents)

    def add_texts(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_text(raw)
        embeds = self.embedding_model.embed(chunks)
        self._add_embeddings(embeds, chunks)

    def add_images(self, image_paths):
        embeds = self.embedding_model.embed(image_paths)
        self._add_embeddings(embeds, image_paths)

    def add_pairs(self, pairs):
        embedded_units = []
        return_units = []

        for img_path, caption in pairs:
            pair_data = (img_path, caption)
            if self.pair_embed_mode == "text":
                embedded_units.append(caption)
                return_units.append(pair_data)
            elif self.pair_embed_mode == "image":
                embedded_units.append(img_path)
                return_units.append(pair_data)
            elif self.pair_embed_mode == "both":
                embedded_units.extend([caption, img_path])
                return_units.extend([pair_data, pair_data])
            elif self.pair_embed_mode is None:
                continue

        if embedded_units:
            embeds = self.embedding_model.embed(embedded_units)
            self._add_embeddings(embeds, return_units)

    def search(self, query, top_k=3):
        query_vec = self.embedding_model.embed([query])
        _, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

    def load(self, path):
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))


class ModeSelector:
    def __init__(self, mode: int, embedding_model):
        self.mode = mode
        if mode == 1:
            self.rag = MultimodalRAG(embedding_model, pair_embed_mode=None)
        elif mode == 2:
            self.rag = MultimodalRAG(embedding_model, pair_embed_mode="text")
        elif mode == 3:
            self.rag = MultimodalRAG(embedding_model, pair_embed_mode="text")
        elif mode == 4:
            self.rag = MultimodalRAG(embedding_model, pair_embed_mode="both")
        else:
            raise ValueError("Invalid mode")

    def add_data(self, texts=None, images=None, pairs=None):
        if texts:
            self.rag.add_texts(texts)
        if images:
            self.rag.add_images(images)
        if pairs:
            self.rag.add_pairs(pairs)

    def search(self, query, top_k=3):
        results = self.rag.search(query, top_k=top_k)
        if self.mode == 2:
            # 只返回文字
            cleaned = []
            for r in results:
                if isinstance(r, tuple):
                    cleaned.append(r[1])  # caption
                else:
                    cleaned.append(r)
            return cleaned
        return results

    def save(self, path):
        self.rag.save(path)

    def load(self, path):
        self.rag.load(path)

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import pickle


class RAG():
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2",
                 embedding_model_dir="model_para/embedding"):
        self.embedding_model_path = os.path.join(embedding_model_dir, embedding_model_name)

        if not os.path.exists(self.embedding_model_path):
            print("Downloading embedding model...")
            self.model = SentenceTransformer(embedding_model_name)
            self.model.save(self.embedding_model_path)
        else:
            self.model = SentenceTransformer(self.embedding_model_path)
            
        self.documents = []
        self.index = None
        print("RAG initialized.")


    def add_txt_data(self, txt_path, separator="\n", chunk_size=1000, chunk_overlap=0): 
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        new_docs = splitter.split_text(raw_text)
        new_embeddings = self.model.encode(new_docs, convert_to_numpy=True)

        if self.index is None:
            dim = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        else:
            assert self.index.d == new_embeddings.shape[1], 'The dimension of the new embeddings does not match the dimension of the existing index.'

        self.index.add(new_embeddings)
        self.documents.extend(new_docs)


    def search(self, user_query, top_k=3):
        query_embedding = self.model.encode([user_query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        return [self.documents[idx] for idx in indices[0]]
            
            
    def clear_data(self):
        self.documents = []
        self.index = None
        
        
    def save(self, save_path):
        with open(os.path.join(save_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        faiss.write_index(self.index, os.path.join(save_path, "faiss.index"))
        print("Database saved.")


    def load(self, load_path):
        doc_path = os.path.join(load_path, "documents.pkl")
        index_path = os.path.join(load_path, "faiss.index")

        if os.path.exists(doc_path) and os.path.exists(index_path):
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
            self.index = faiss.read_index(index_path)
            print("Database loaded.")

    
if __name__ == "__main__":
    rag = RAG()
    rag.add_txt_data("RAG\dataset\《中药学》药品介绍.txt")
    context = rag.search("怎么治疗过敏")
    print(context)
    rag.save('RAG\dataset\\vector_database')
    
    rag.clear_data()
    rag.load('RAG\dataset\\vector_database')
    context = rag.search("萆薢是什么", top_k=1)
    print(context)
# src/inference/embedding_model.py

from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name="intfloat/e5-large-v2", device="cpu"):
        """
        모델 로드 및 장치 할당
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.device = device
        if device in ["cuda", "mps"]:
            self.model.to(self.device)

    def get_embedding(self, text: str):
        """
        단일 문장(text)에 대한 임베딩(1D list[float]) 반환
        """
        emb = self.model.encode([text], convert_to_numpy=True, device=self.device)[0]
        return emb.tolist()

    def get_embeddings(self, texts: list):
        """
        여러 문장(texts)에 대한 임베딩(2D numpy array) 반환
        """
        embs = self.model.encode(texts, convert_to_numpy=True, device=self.device)
        return embs

# 전역 인스턴스 예시

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
embedding_model = EmbeddingModel(model_name="intfloat/e5-large-v2", device=device)
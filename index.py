from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    link: str
    title: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    link: str
    score: float
    title: str
    text: str


def load_documents(path: str) -> List[Document]:
    """Загрузка документов из json файла"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            link = article["link"],
            title=article["title"],
            text=article["text"],
            embedding=None,
        )
        for article in data
    ]


class Indexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[Document]) -> None:
        """
        TODO: Реализовать индексацию документов
        1. Сохранить документы в self.documents
        2. Получить эмбеддинги для документов используя self.model.encode()
           Подсказка: для каждого документа нужно объединить title и text
        3. Сохранить эмбеддинги в self.embeddings
        """
        self.documents = docs
        self.embeddings = self.model.encode(
            [doc.title + " " + doc.text for doc in docs]
        )

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        1. Сохранить self.documents и self.embeddings в pickle файл
        """

        with open(path, "wb") as dump_out:
            pickle.dump(
                {"documents": self.documents, "embeddings": self.embeddings}, dump_out
            )

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        1. Загрузить self.documents и self.embeddings из pickle файла
        """
        with open(path, "rb") as dump_in:
            data = pickle.load(dump_in)
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]


class Searcher:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        TODO: Реализовать инициализацию поиска
        1. Загрузить индекс из index_path
        2. Инициализировать sentence-transformers
        """
        self.model = SentenceTransformer(model_name)
        self.index = Indexer(model_name)
        self.index.load(index_path)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск документов
        1. Получить эмбеддинг запроса через self.model.encode()
        2. Вычислить косинусное сходство между запросом и документами
        3. Вернуть top_k наиболее похожих документов
        """
        emb = self.model.encode(query)
        query_norm = np.linalg.norm(emb)
        s = (self.index.embeddings @ emb) / query_norm
        s = s / np.linalg.norm(self.index.embeddings, axis=1)
        arg_ans = np.argsort(s)
        ans = []
        for i in range(len(arg_ans) - 1, len(arg_ans) - 1 - top_k, -1):
            doc = self.index.documents[arg_ans[i]]
            ans.append(
                SearchResult(
                    link=doc.link, text=doc.text, title=doc.title, score=float(s[arg_ans[i]])
                )
            )
        return ans

def documents():
    return load_documents('data.json')

#docs = documents()

#indexer = Indexer()
#indexer.add_documents(docs)
#indexer.save('data.json')


def get_similar(query):
    searcher = Searcher(str("data.json"))
    results = searcher.search(query, top_k=3)
    return results
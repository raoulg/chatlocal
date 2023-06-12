from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger

from chatlocal.dataloader import DataLoader
from chatlocal.settings import VectorStoreSettings

if TYPE_CHECKING:
    from dataloader import DataStore

storesettings = VectorStoreSettings(
    chunk_size=1500, separator="\n", store_file=Path("vectorstore.pkl")
)


class VectorStore:
    def __init__(self, settings: VectorStoreSettings = storesettings):
        self.chunk_size = settings.chunk_size
        self.separator = settings.separator
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, separator=self.separator
        )
        self.cache = settings.cache
        self.store_path = settings.cache / settings.store_file
        self.index_path = settings.cache / "docs.index"
        self.initialized = False

    def build(self, datastore: DataStore) -> None:
        docs = []
        metadatas = []
        logger.info(f"Building vectorstore from {len(datastore)} documents")
        for doc in datastore:
            splits = self.text_splitter.split_text(doc.text)
            docs.extend(splits)
            metadatas.extend([{"source": doc.source}] * len(splits))
        logger.success("Processed datastore")

        self.store = FAISS.from_texts(
            texts=docs, embedding=OpenAIEmbeddings(), metadatas=metadatas  # type: ignore
        )
        self.initialized = True

    def add_documents(self, datastore: DataStore) -> None:
        """Adding additional documents to the vectorstore."""
        if not self.initialized:
            raise ValueError("Vectorstore not initialized. Run build() first.")
        docs = []
        metadatas = []
        logger.info(f"Adding {len(datastore)} new documents to vectorstore")
        for doc in datastore:
            splits = self.text_splitter.split_text(doc.text)
            docs.extend(splits)
            metadatas.extend([{"source": doc.source}] * len(splits))

        self.store.add_texts(texts=docs, metadatas=metadatas)

    def get(self) -> FAISS:
        return self.store

    def save(self) -> None:
        """Pickle the FAISS store"""
        faiss.write_index(self.store.index, str(self.index_path))
        self.store.index = None
        with open(self.store_path, "wb") as f:
            pickle.dump(self.store, f)
        logger.success(f"Saving vectorstore to {self.store_path}")

    def load(self, store_path: Path) -> None:
        """Load a pickled FAISS store"""
        logger.info(f"Loading vectorstore from {store_path}")
        index = faiss.read_index(str(self.index_path))
        with open(store_path, "rb") as f:
            self.store = pickle.load(f)
        self.store.index = index

    @classmethod
    def from_dataloader(
        cls, dataloader: DataLoader, settings: VectorStoreSettings = storesettings
    ) -> VectorStore:
        """Construct a VectorStore from a DataStore."""
        vectorstore = cls(settings=settings)  # type: ignore
        vectorstore.build(datastore=dataloader.get_datastore())
        return vectorstore

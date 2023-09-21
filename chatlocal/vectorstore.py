from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger

from chatlocal.settings import ModelType, VectorStoreSettings

if TYPE_CHECKING:
    from dataloader import DataStore
    from langchain.embeddings.base import Embeddings

    from chatlocal.dataloader import DataLoader

storesettings = VectorStoreSettings(
    chunk_size=1500,
    separator="\n",
    store_file=Path("scepa.pkl"),
    modeltype=ModelType.OPENAI,
)


class VectorStore:
    def __init__(self, settings: VectorStoreSettings = storesettings) -> None:
        self.chunk_size = settings.chunk_size
        self.separator = settings.separator
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            separator=self.separator,
        )
        self.modeltype = settings.modeltype
        self.cache = settings.cache
        self.store_path = settings.cache / settings.store_file
        self.index_path = settings.cache / settings.store_file.with_suffix(".index")
        self.initialized = False
        self.stepsize = 100

    def chunk_datastore(self, datastore: DataStore) -> dict[str, list[str]]:
        docs = []
        metadatas = []
        logger.info(f"Building vectorstore from {len(datastore)} documents")
        for doc in datastore:
            splits = self.text_splitter.split_text(doc.text)
            docs.extend(splits)
            metadatas.extend([{"source": doc.source}] * len(splits))
        logger.success(f"Processed datastore into {len(docs)} chunks.")
        return {"docs": docs, "metadatas": metadatas}

    def create_store(self, datastore: DataStore) -> None:
        embedding = self.get_embeddings()
        chunked = self.chunk_datastore(datastore)
        docs = chunked["docs"]
        metadatas = chunked["metadatas"]

        logger.info(f"Initailizing vectorstore for first {self.stepsize} documents")
        self.store = FAISS.from_texts(
            texts=docs[: self.stepsize],
            embedding=embedding,
            metadatas=metadatas[: self.stepsize],  # type: ignore
        )
        for i in range(self.stepsize, len(docs), self.stepsize):
            logger.info(f"Adding range {i}-{i+self.stepsize} to vectorstore")
            self.store.add_texts(
                texts=docs[i : i + self.stepsize],
                embedding=embedding,
                metadatas=metadatas[i : i + self.stepsize],
            )

        logger.success("Initialized vectorstore")
        self.initialized = True

    def add_documents(self, datastore: DataStore) -> None:
        """Adding additional documents to the vectorstore."""
        if not self.initialized:
            msg = "Vectorstore not initialized. Run build() first."
            raise ValueError(msg)
        docs = []
        metadatas = []
        logger.info(f"Adding {len(datastore)} new documents to vectorstore")
        for doc in datastore:
            splits = self.text_splitter.split_text(doc.text)
            docs.extend(splits)
            metadatas.extend([{"source": doc.source}] * len(splits))

        self.store.add_texts(texts=docs, metadatas=metadatas)

    def get_embeddings(self) -> Embeddings:
        if self.modeltype == ModelType.HUGGINGFACE:
            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": False}
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        if self.modeltype == ModelType.OPENAI:
            return OpenAIEmbeddings()
        msg = "Model type not supported"
        raise ValueError(msg)

    def get(self) -> FAISS:
        return self.store

    def save(self) -> None:
        """Pickle the FAISS store"""
        faiss.write_index(self.store.index, str(self.index_path))
        self.store.index = None
        with open(self.store_path, "wb") as f:
            pickle.dump(self.store, f)
        logger.success(f"Saved vectorstore to {self.store_path} and {self.index_path}")

    def load(self, store_path: Path) -> None:
        """Load a pickled FAISS store"""
        logger.info(f"Loading vectorstore from {store_path}")
        index = faiss.read_index(str(self.index_path))
        with open(store_path, "rb") as f:
            self.store = pickle.load(f)
        self.store.index = index

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        settings: VectorStoreSettings = storesettings,
    ) -> VectorStore:
        """Construct a VectorStore from a DataStore."""
        vectorstore = cls(settings=settings)  # type: ignore
        vectorstore.create_store(datastore=dataloader.get_datastore())
        return vectorstore

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

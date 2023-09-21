from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger

from chatlocal.settings import Chunk, ModelType, VectorStoreSettings

if TYPE_CHECKING:
    from dataloader import DataStore
    from langchain.embeddings.base import Embeddings

    from chatlocal.dataloader import DataLoader


class VectorStore:
    """
    A vectorstore is a wrapper around a FAISS vectorstore.
    It takes a DataStore as input, chunks the text into chunk_size chunks,
    turns them into embeddings and stores them in a FAISS vectorstore.

    Use either create_store to create a new vectorstore, or extend_store to
    extend an existing vectorstore created or to be loaded from disk.
    """
    def __init__(self, settings: VectorStoreSettings) -> None:
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

    def create_store(self, datastore: DataStore) -> None:
        """Creates a new vectorstore from a datastore

        Args:
            datastore (DataStore): _description_
        """
        chunked = self.chunk_datastore(datastore)

        logger.info(f"Initailizing vectorstore for first {self.stepsize} documents")
        self.init_store(chunked)
        self.add_chunks(chunked, self.stepsize)

        logger.success("Initialized vectorstore")
        self.initialized = True

    def extend_store(self, datastore: DataStore) -> None:
        """Extends an existing vectorstore with a datastore
        The existing vectorstore can either be initialized with .create_store
        or it can be loaded from disk (using the settings.store_file in the .cache)

        Args:
            datastore (DataStore): _description_
        """
        if not self.initialized:
            self.load(self.store_path)
        chunked = self.chunk_datastore(datastore)
        self.add_chunks(chunked, 0)
        logger.success("Extended vectorstore. Please save to disk to keep changes!")

    def chunk_datastore(self, datastore: DataStore) -> Chunk:
        """
        Takes in a datastore, chunks the documents into smaller chunks of texts
        and returns a Chunk object with chunks of text and metadata.

        Args:
            datastore (DataStore):

        Returns:
            Chunk:
        """

        docs = []
        metadatas = []
        logger.info(f"Building vectorstore from {len(datastore)} documents")
        for doc in datastore:
            splits = self.text_splitter.split_text(doc.text)
            docs.extend(splits)
            metadatas.extend([{"source": doc.source}] * len(splits))
        logger.success(f"Processed datastore into {len(docs)} chunks.")
        return Chunk(docs=docs, metadatas=metadatas)

    def init_store(self, chunked: Chunk) -> None:
        """start a new vectorstore

        Args:
            chunked (Chunk): _description_
        """
        if self.initialized:
            logger.warning("Store already initialized.")
        else:
            embedding = self.get_embeddings()
            self.store = FAISS.from_texts(
                texts=chunked.docs[: self.stepsize],
                embedding=embedding,
                metadatas=chunked.metadatas[: self.stepsize],  # type: ignore
            )
        self.initialized = True

    def add_chunks(self, chunked: Chunk, start: int) -> None:
        """add chunks to an existing vectorstore

        Args:
            chunked (Chunk): _description_
            start (int): 0 if starting from scratch, else the number of
                documents already in the store
            n_docs (int): total number of documents to add
        """
        embedding = self.get_embeddings()
        n_docs = len(chunked.docs)
        for i in range(start, n_docs, self.stepsize):
            logger.info(f"Adding range {i}-{i+self.stepsize} of {n_docs}")
            self.store.add_texts(
                texts=chunked.docs[i : i + self.stepsize],
                embedding=embedding,
                metadatas=chunked.metadatas[i : i + self.stepsize],
            )


    def get_embeddings(self) -> Embeddings:
        """Returns embeddings for the selected model

        Raises:
            ValueError: Raised if the modeltype is not supported

        Returns:
            Embeddings
        """
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
        with self.store_path.open(mode="wb") as f:
            pickle.dump(self.store, f)
        logger.success(f"Saved vectorstore to {self.store_path} and {self.index_path}")

    def load(self, store_path: Path) -> None:
        """Load a pickled FAISS store"""
        if not store_path.exists():
            logger.info(
                f"Store not found at {store_path}. "
                "Please first initialize the store with .create_store."
            )
            logger.error("Aborting because no store is found to extend.")
        logger.info(f"Loading vectorstore from {store_path}")
        index = faiss.read_index(str(self.index_path))
        with store_path.open(mode="rb") as f:
            self.store = pickle.load(f)  # noqa: S301
        self.store.index = index
        self.initialized = True

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        settings: VectorStoreSettings,
    ) -> VectorStore:
        """Construct a VectorStore from a DataStore."""
        vectorstore = cls(settings=settings)  # type: ignore
        vectorstore.create_store(datastore=dataloader.get_datastore())
        return vectorstore

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

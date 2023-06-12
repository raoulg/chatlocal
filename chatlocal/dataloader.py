from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterator, List

from loguru import logger
from pydantic import BaseModel


class FileType(Enum):
    TEXT = ".txt"
    MD = ".md"


class Document(BaseModel):
    text: str
    source: Path


class DataStore:
    data: List[Document] = []
    _index: int = 0

    def add(self, document: Document) -> None:
        self.data.append(document)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Document]:  # type: ignore
        self._index = 0
        return self

    def __next__(self) -> Document:
        if self._index < len(self.data):
            result = self.data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    class Config:
        arbitrary_types_allowed = True


class DataLoader:
    def __init__(self, filetypes: List[FileType] = [FileType.MD]) -> None:
        logger.info(f"Initializing Dataloader for files of type {filetypes}...")
        self.filetypes = filetypes
        self.datastore = DataStore()

    def load_files(self, path: Path) -> None:
        if path.exists() is False:
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_dir() is False:
            raise NotADirectoryError(f"{path} is not a directory")

        notepaths = self.walk_dir(path)
        num_docs = 0
        for source in notepaths:
            file_extension = source.suffix.lower()  # Get the file extension

            if not any(file_extension == ft.value for ft in self.filetypes):
                continue  # Skip the file if it doesn't match any of the specified file types

            with open(source, "r") as f:
                content = f.read()
            if len(content) > 0:
                doc = Document(text=content, source=source)
                self.datastore.add(doc)
            num_docs += 1
        logger.info(f"Added {num_docs} notes to the datastore.")

    def get_datastore(self) -> DataStore:
        return self.datastore

    def walk_dir(self, path: Path) -> Iterator[Path]:
        """loops recursively through a folder

        Args:
            path (Path): folder to loop trough. If a directory
                is encountered, loop through that recursively.

        Yields:
            Iterator: all paths in a folder and subdirs.
        """
        path = path.expanduser().resolve()
        for p in Path(path).iterdir():
            if p.is_dir():
                yield from self.walk_dir(p)
                continue
            yield p

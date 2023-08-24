from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from itertools import chain
from typing import Dict, Iterator, List

from loguru import logger
from pydantic import BaseModel, root_validator


class FileType(Enum):
    TEXT = ".txt"
    LATEX = ".tex"
    MD = ".md"
    DOCX = ".docx"
    PDF = ".pdf"
    JUPYTER = ".ipynb"


class FormattedBase(BaseModel):
    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


class ParserSettings(FormattedBase):
    textfiles: List[FileType]
    wordfiles: List[FileType]
    pdffiles: List[FileType]
    jupyterfiles: List[FileType]
    ignore_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.mypy_cache', '.vscode', 'venv', 'env']

    def __iter__(self) -> Iterator[FileType]:  # type: ignore
        return chain(self.textfiles, self.wordfiles, self.pdffiles, self.jupyterfiles)


    @classmethod
    def from_filetypes(cls, filetypes: List[FileType]) -> ParserSettings:
        defaulttextfiles: List[FileType] = [FileType.MD, FileType.TEXT, FileType.LATEX]
        defaultwordfiles: List[FileType] = [FileType.DOCX]
        defaultpdffiles: List[FileType] = [FileType.PDF]
        defaultjupyterfiles: List[FileType] = [FileType.JUPYTER]

        textfiles = [ft for ft in filetypes if ft in defaulttextfiles]
        wordfiles = [ft for ft in filetypes if ft in defaultwordfiles]
        pdffiles = [ft for ft in filetypes if ft in defaultpdffiles]
        jupyterfiles = [ft for ft in filetypes if ft in defaultjupyterfiles]

        return cls(textfiles=textfiles, wordfiles=wordfiles, pdffiles=pdffiles, jupyterfiles=jupyterfiles)


class Document(FormattedBase):
    text: str
    source: Path


class ModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class VectorStoreSettings(FormattedBase):
    """Summary
    This class is used to store the settings for the vectorstore.

    Args:
        chunk_size (int): The size of the chunks to split the text into. This is related
            to the max amount of tokens that can be processed by the model. Currently
            this is 4096 for the OpenAI GPT-4 model, and with a rough 1:4 ratio of
            words to token, this is about 1024 words.
        separator (str): The separator to use when splitting the text, eg '\\n'
        cache (Path, optional): The path to the cache directory. Defaults to $HOME/.cache/chatlocal
        store_file (Path, optional): The path to the picklefile to save the vectorstore to.

    Returns:
        _type_: _description_
    """

    chunk_size: int
    separator: str
    cache: Path = Path(os.getenv("CACHE_DIR", Path.home() / ".cache" / "chatlocal"))
    store_file: Path
    modeltype: ModelType

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        cache = values.get("cache").resolve()
        if not cache.exists():
            logger.info(f"cache did not exist. Creating at {cache}.")
            cache.mkdir(parents=True)
        return values

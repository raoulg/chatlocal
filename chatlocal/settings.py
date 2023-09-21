from __future__ import annotations

import os
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    textfiles: list[FileType]
    wordfiles: list[FileType]
    pdffiles: list[FileType]
    jupyterfiles: list[FileType]
    ignore_dirs: list[str] = [  # noqa: RUF012
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".vscode",
        "venv",
        "env",
    ]

    def __iter__(self) -> Iterator[FileType]:  # type: ignore
        return chain(self.textfiles, self.wordfiles, self.pdffiles, self.jupyterfiles)

    @classmethod
    def from_filetypes(cls, filetypes: list[FileType]) -> ParserSettings:
        defaulttextfiles: list[FileType] = [FileType.MD, FileType.TEXT, FileType.LATEX]
        defaultwordfiles: list[FileType] = [FileType.DOCX]
        defaultpdffiles: list[FileType] = [FileType.PDF]
        defaultjupyterfiles: list[FileType] = [FileType.JUPYTER]

        textfiles = [ft for ft in filetypes if ft in defaulttextfiles]
        wordfiles = [ft for ft in filetypes if ft in defaultwordfiles]
        pdffiles = [ft for ft in filetypes if ft in defaultpdffiles]
        jupyterfiles = [ft for ft in filetypes if ft in defaultjupyterfiles]

        return cls(
            textfiles=textfiles,
            wordfiles=wordfiles,
            pdffiles=pdffiles,
            jupyterfiles=jupyterfiles,
        )


class Document(FormattedBase):
    text: str
    source: Path

class Chunk(BaseModel):
    docs: list[str]
    metadatas: list[str]


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
        cache (Path, optional): The path to the cache directory.
            Defaults to $HOME/.cache/chatlocal
        store_file (Path, optional): The path to save the vectorstore.

    Returns:
        _type_: _description_
    """

    chunk_size: int
    separator: str
    cache: Path = Path(
        os.getenv("CACHE_DIR", Path.home() / ".cache" / "chatlocal"),  # noqa: PLW1508
    )
    store_file: Path
    modeltype: ModelType

    @model_validator(mode="after")
    def check_folder(self) -> VectorStoreSettings:
        cache = self.cache
        if not cache.exists():
            logger.info(f"cache did not exist. Creating at {cache}.")
            cache.mkdir(parents=True)
        return self

class UserConfig(FormattedBase):
    folder: Path
    filetypes: list[str]
    store_file: Path = Path("vectorstore.pkl")
    chunk_size: int = 1500

    @model_validator(mode="after")
    def check_folder(self) -> UserConfig:
        folder = self.folder
        if not folder.exists():
            logger.error(f"folder {folder} does not exist")
        logger.info(f"using folder {folder}")

        return self

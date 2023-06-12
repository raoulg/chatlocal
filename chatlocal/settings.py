from pathlib import Path
from typing import Dict
import os

from loguru import logger
from pydantic import BaseModel, root_validator


class VectorStoreSettings(BaseModel):
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
    cache: Path = Path(os.getenv('CACHE_DIR', Path.home() / ".cache" / "chatlocal"))
    store_file: Path

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        cache = values.get("cache").resolve()
        if not cache.exists():
            logger.info(f"cache did not exist. Creating at {cache}.")
            cache.mkdir(parents=True)
        return values

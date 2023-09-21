
from pathlib import Path

import click
from dataloader import read_toml
from loguru import logger

from chatlocal import DataLoader, FileType, ModelType, VectorStore, VectorStoreSettings


@click.command()
@click.option("-f", "--folder", help="Folder to parse")
@click.option("-t", "--types", multiple=True, help="Filetypes to parse")
def main(folder: str, types: list[str]) -> None:
    tomlconfig = read_toml()
    if folder is None:
        folder = tomlconfig.get("folder")
    docpath = Path(folder).expanduser().resolve()

    if not docpath.exists():
        logger.error(f"folder {docpath} does not exist")
    logger.info(f"using folder {docpath}")

    if len(types) == 0:
        types = tomlconfig["types"]

    filetypes = [FileType(ft) for ft in types]
    logger.info(f"using filetypes {[ft.value for ft in filetypes]}")

    dataloader = DataLoader(filetypes=filetypes)
    dataloader.load_files(docpath)

    storesettings = VectorStoreSettings(
        chunk_size=1500,
        separator="\n",
        store_file=tomlconfig.get("store_file", Path("vectorstore.pkl")),
        modeltype=ModelType.OPENAI,
    )
    logger.info(f"using store settings {storesettings}")

    vectorstore = VectorStore(settings=storesettings)
    vectorstore.create_store(dataloader.get_datastore())
    vectorstore.save()

if __name__ == "__main__":
    main()

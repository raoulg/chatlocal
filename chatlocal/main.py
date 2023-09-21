

from dataloader import read_toml
from loguru import logger

from chatlocal import DataLoader, FileType, ModelType, VectorStore, VectorStoreSettings
from chatlocal.settings import UserConfig


def build_vectorstore() -> None:
    cfg = read_toml()
    userconfig = UserConfig(**cfg)
    userconfig.folder = userconfig.folder.expanduser().resolve()

    logger.info(f"using folder {userconfig.folder} to build vectorstore.")

    filetypes = [FileType(ft) for ft in userconfig.filetypes]
    logger.info(f"Looking for filetypes {[ft.value for ft in filetypes]}.")

    dataloader = DataLoader(filetypes=filetypes)
    dataloader.load_files(userconfig.folder)
    logger.success("Finished loading files")

    storesettings = VectorStoreSettings(
        chunk_size=userconfig.chunk_size,
        separator="\n",
        store_file=userconfig.store_file,
        modeltype=ModelType.OPENAI,
    )

    vectorstore = VectorStore(settings=storesettings)
    if vectorstore.store_path.exists():
        logger.info(f"Found existing vectorstore at {vectorstore.store_path}")
        logger.info("Extending vectorstore with all files found in folder.")
        vectorstore.extend_store(dataloader.get_datastore())
    else:
        logger.info(
            f"No existing vectorstore found at {vectorstore.store_path}"
            "Creating new vectorstore.",
            )

        vectorstore.create_store(dataloader.get_datastore())
    vectorstore.save()
    logger.success(f"Saved vectorstore to {vectorstore.store_path}")

if __name__ == "__main__":
    build_vectorstore()
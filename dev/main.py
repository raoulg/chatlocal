from chatlocal import DataLoader, VectorStore, VectorStoreSettings, ModelType
from pathlib import Path
import click


@click.command()
@click.option("-f", "--folder", help="Path to folder with documents to parse")
def main(folder: str):
    dataloader = DataLoader()
    docpath = Path("~/code/curriculum").expanduser()
    dataloader.load_files(docpath)

    # storesettings = VectorStoreSettings(
    # chunk_size=1500, separator="\n", store_file=Path("vectorstore.pkl"), modeltype=ModelType.OPENAI
    # )

    # vectorstore = VectorStore(settings = storesettings)
    # vectorstore.build(dataloader.get_datastore())
    # vectorstore.save()

if __name__ == "__main__":
    main()
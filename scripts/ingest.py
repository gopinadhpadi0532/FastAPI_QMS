import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ingestion():
    """
    Executes the document ingestion pipeline using LangChain:
    1. Loads documents from a directory.
    2. Splits documents into manageable chunks.
    3. Generates embeddings for each chunk.
    4. Stores the chunks and their embeddings in a persistent ChromaDB.
    """
    try:
        settings = get_settings()
        logging.info("Starting document ingestion pipeline with LangChain...")

        # 1. Load Documents
        # Using DirectoryLoader with UnstructuredFileLoader to handle PDF, DOCX, etc.
        loader = DirectoryLoader(
            settings.DATA_PATH,
            glob="**/*",  # Load all files in all subdirectories
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        if not documents:
            logging.warning("No documents found. Aborting ingestion.")
            return
        logging.info(f"Successfully loaded {len(documents)} documents.")

        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(chunks)} chunks.")

        # 3. Create Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBED_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY
        )

        # 4. Store in ChromaDB
        logging.info("Creating and persisting ChromaDB vector store. This may take a while...")
        # Chroma.from_documents handles embedding and storing in one step.
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=settings.DB_PERSIST_PATH
        )
        logging.info(f"Ingestion complete. Vector store persisted at: {settings.DB_PERSIST_PATH}")

    except Exception as e:
        logging.error(f"An error occurred during ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    run_ingestion()
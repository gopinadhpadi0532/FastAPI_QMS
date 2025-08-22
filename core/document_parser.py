import logging
from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    A parser to load documents from a directory and extract their content.
    Attaches filename metadata to each document.
    """

    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Loads all .pdf and .docx documents from the specified directory.

        Args:
            directory_path: The path to the directory containing QMS documents.

        Returns:
            A list of Document objects, each with content and metadata.
        """
        path = Path(directory_path)
        if not path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []

        logger.info(f"Loading documents from: {directory_path}")
        
        # Using SimpleDirectoryReader which intelligently uses unstructured
        # to parse different file types.
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".pdf", ".docx"],
            recursive=True
        )

        documents = reader.load_data(show_progress=True)
        
        # The file_path is automatically added to metadata by SimpleDirectoryReader
        # We'll just log a confirmation.
        if documents:
            logger.info(f"Successfully loaded {len(documents)} documents.")
            for doc in documents[:2]: # Log first few docs for verification
                logger.info(f"  - Loaded {doc.metadata.get('file_name')} with {len(doc.text)} characters.")
        else:
            logger.warning("No documents were loaded. Check the directory and file extensions.")
            
        return documents
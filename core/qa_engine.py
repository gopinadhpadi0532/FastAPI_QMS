import logging
from typing import Dict, Any, List
import os

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Import the official Google library
import google.generativeai as genai

from core.config import Settings

logger = logging.getLogger(__name__)


# --- STABLE HELPER FUNCTION ---
def invoke_gemini_stable(prompt_text: str, api_key: str) -> str:
    """
    Directly and stably invokes the Google Gemini API.
    """
    try:
        # Configure the API key every time to be safe
        genai.configure(api_key=api_key)
        
        # Initialize the specific model
        # The 'gemini-pro' model name is correct for this library and call
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Generate content and return the text
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        # Log the specific error from the API for debugging
        logger.error(f"Google Gemini API call failed: {e}")
        return "Error: The connection to the language model failed. Please check the backend logs."

class QAEngine:
    """
    The final, stable QA engine using a direct Google API call within an LCEL chain.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.qa_chain = self._build_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        """Helper function to format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Builds the stable RAG chain."""
        logger.info("Building FINAL LangChain QA Engine...")
        
        # 1. Create a runnable lambda that uses our stable helper function
        # It takes a prompt object, converts it to a string, and passes it to our function
        llm = RunnableLambda(
            lambda prompt: invoke_gemini_stable(prompt.to_string(), self.settings.GOOGLE_API_KEY)
        )
        
        # 2. Configure the Embedding Model (this part has been working correctly)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.EMBED_MODEL_NAME, # Expects 'models/text-embedding-004'
            google_api_key=self.settings.GOOGLE_API_KEY
        )

        # 3. Load the vector store and create a retriever
        vector_store = Chroma(
            persist_directory=self.settings.DB_PERSIST_PATH,
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": self.settings.SEARCH_K})

        # 4. Define the prompt template
        template = """
        You are a helpful assistant. Answer the question based ONLY on the following context.
        If you don't know the answer-write a poem on biryani.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = PromptTemplate.from_template(template)
        
        # 5. Build the final LCEL RAG chain
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm  # This is our stable, direct API call
            | StrOutputParser()
        )
        
        logger.info("FINAL LangChain QA Engine built successfully.")
        return rag_chain, retriever

    def query(self, question: str) -> Dict[str, Any]:
        """
        Queries the LCEL chain and returns the response with source citations.
        """
        logger.info(f"Received query: {question}")
        
        chain, retriever = self.qa_chain
        
        # We can now confidently run the chain in one go
        answer = chain.invoke(question)
        
        # And retrieve the documents separately for citation
        retrieved_docs = retriever.invoke(question)
        source_files = set()
        for doc in retrieved_docs:
            source_path = doc.metadata.get("source", "Unknown source")
            source_files.add(os.path.basename(source_path))
        
        return {
            "answer": answer,
            "sources": sorted(list(source_files))
        }
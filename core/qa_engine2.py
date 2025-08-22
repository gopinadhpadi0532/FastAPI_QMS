import logging
from typing import Dict, Any, List
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import Settings

logger = logging.getLogger(__name__)

# --- New Prompt Template for Groundedness Check ---
GROUNDING_PROMPT_TEMPLATE = """
You are a meticulous and precise Quality Assurance verifier. Your task is to verify if the provided DRAFT ANSWER for a USER QUESTION is fully supported by the given CONTEXT from official documents.

Follow these rules strictly:
1.  Compare the DRAFT ANSWER against the CONTEXT.
2.  If the DRAFT ANSWER is fully and directly supported by the CONTEXT, approve it and output the answer as is.
3.  If the DRAFT ANSWER contains any information, details, or claims NOT explicitly found in the CONTEXT, you must REJECT it.
4.  If you reject the draft, you must generate a NEW, corrected answer that is based ONLY on the provided CONTEXT.
5.  If the CONTEXT does not contain enough information to answer the question, you must state: "Based on the provided documents, I cannot answer this question."

USER QUESTION:
{question}

CONTEXT:
{context}

DRAFT ANSWER:
{draft_answer}

Based on your verification, what is the final, verified answer?
FINAL VERIFIED ANSWER:
"""

GROUNDING_PROMPT = PromptTemplate.from_template(GROUNDING_PROMPT_TEMPLATE)

class QAEngine:
    """
    The core RAG engine using LangChain, now with a grounding check.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retrieval_qa_chain = self._build_retrieval_qa_chain()
        self.grounding_chain = self._build_grounding_chain()

    def _build_retrieval_qa_chain(self):
        """Builds the initial RAG chain to generate a draft answer."""
        logger.info("Building initial RetrievalQA chain...")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.EMBED_MODEL_NAME,
            google_api_key=self.settings.GOOGLE_API_KEY
        )
        vector_store = Chroma(
            persist_directory=self.settings.DB_PERSIST_PATH,
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": self.settings.SEARCH_K})
        
        # We use a slightly faster/cheaper model for the initial draft
        draft_llm = ChatGoogleGenerativeAI(
            model=self.settings.MODEL_NAME, # e.g., gemini-1.5-flash
            google_api_key=self.settings.GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=draft_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return chain

    def _build_grounding_chain(self):
        """Builds the second chain responsible for verifying the answer."""
        logger.info("Building grounding verification chain...")
        
        # Use a powerful, precise model for verification
        # NOTE: You could use a more powerful model like "gemini-1.5-pro-latest" here if available
        verifier_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", 
            google_api_key=self.settings.GOOGLE_API_KEY,
            temperature=0.0 # Set temperature to 0 for factual, non-creative checks
        )
        
        return GROUNDING_PROMPT | verifier_llm | StrOutputParser()

    def query(self, question: str) -> Dict[str, Any]:
        """Processes a question with a two-step generate-then-verify process."""
        logger.info(f"Received query for grounding check: {question}")
        
        # Step 1: Generate a draft answer and get source documents
        draft_response = self.retrieval_qa_chain.invoke({"query": question})
        draft_answer = draft_response.get("result", "")
        source_documents = draft_response.get("source_documents", [])
        
        if not source_documents:
            return {
                "answer": "I could not find any relevant documents to answer this question.",
                "sources": []
            }

        # Format the context from source documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])

        # Step 2: Run the grounding check
        logger.info("Performing grounding verification on the draft answer...")
        final_answer = self.grounding_chain.invoke({
            "question": question,
            "context": context_text,
            "draft_answer": draft_answer
        })
        
        # Extract unique source filenames
        source_files = set()
        for doc in source_documents:
            source_path = Path(doc.metadata.get("source", ""))
            source_files.add(source_path.name)
        
        return {
            "answer": final_answer,
            "sources": sorted(list(source_files))
        }
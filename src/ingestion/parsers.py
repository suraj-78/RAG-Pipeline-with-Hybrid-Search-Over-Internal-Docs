import os
from typing import Any
from pypdf import PdfReader
from bs4 import BeautifulSoup
from src.ingestion.schemas import Document, DocumentMetadata  # Assuming your schemas match this placement

class DocumentParserRouter:
    """Enterprise document routing engine with embedded strict unicode sanitization."""

    @staticmethod
    def sanitize_unicode_string(raw_text: str) -> str:
        """
        Removes unpaired surrogates, null bytes, and corrupted characters 
        that cause Pydantic v2 to throw string_unicode validation errors.
        """
        if not raw_text:
            return ""
        
        # Step 1: Strip out unpaired surrogates (\ud800 to \udfff) which choke Pydantic's Rust validator
        clean_chars = [
            char for char in raw_text 
            if not ('\ud800' <= char <= '\udfff')
        ]
        text_without_surrogates = "".join(clean_chars)
        
        # Step 2: Force encode/decode pass to drop any non-utf8 compliant byte sequences
        utf8_bytes = text_without_surrogates.encode("utf-8", errors="ignore")
        clean_string = utf8_bytes.decode("utf-8", errors="ignore")
        
        # Step 3: Remove hidden null bytes (\x00) which break vector database indexing operations
        return clean_string.replace("\x00", "")

    def _parse_pdf(self, file_path: str) -> str:
        """Extracts text loops from standard PDF layers safely."""
        try:
            reader = PdfReader(file_path)
            extracted_pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)
            return " \n ".join(extracted_pages)
        except Exception as e:
            raise ValueError(f"Failed decoding PDF binary layers: {str(e)}")

    def _parse_html(self, file_path: str) -> str:
        """Strips markup nodes from web structures."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator=" \n ")
        except Exception as e:
            raise ValueError(f"Failed extracting text markup blocks: {str(e)}")

    def _parse_txt(self, file_path: str) -> str:
        """Reads flat text files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed loading raw string buffers: {str(e)}")

    def process_file(self, file_path: str) -> Document:
        """Central orchestration node mapping file extensions to clean Pydantic documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target track asset missing from disk: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if not ext:
            ext = "txt"

        # Routing sequence execution
        if ext == "pdf":
            raw_text = self._parse_pdf(file_path)
        elif ext in ["html", "htm"]:
            raw_text = self._parse_html(file_path)
        elif ext in ["txt", "md"]:
            raw_text = self._parse_txt(file_path)
        else:
            print(f"[WARN] Unknown extension '.{ext}'. Falling back to raw string stream layout.")
            raw_text = self._parse_txt(file_path)

        # FIXED: Core Sanitization Filter call injected right before the Pydantic construction window
        sanitized_content = self.sanitize_unicode_string(raw_text)

        # Build and return the strict 100% unicode compliant Pydantic Schema model
        return Document(
            page_content=sanitized_content,
            metadata=DocumentMetadata(
                source_path=file_path,
                file_type=ext,
                parent_document_id=os.path.basename(file_path)
            )
        )
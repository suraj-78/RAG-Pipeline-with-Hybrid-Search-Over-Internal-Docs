import os
from pypdf import PdfReader
from bs4 import BeautifulSoup
from src.ingestion.schemas import Document, DocumentMetadata

class DocumentParserRouter:
    """The automated processing engine that sanitizes raw data into structured schemas."""
    
    @staticmethod
    def parse_txt_or_md(file_path: str) -> str:
        """Reads flat text and markdown files directly."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    @staticmethod
    def parse_html(file_path: str) -> str:
        """Strips structural code elements out of raw HTML files securely."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove scripting and styling blocks completely
            for element in soup(["script", "style"]):
                element.decompose()
            return soup.get_text(separator="\n").strip()

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extracts sequential string lines out of binary PDF pages."""
        reader = PdfReader(file_path)
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n\n".join(extracted_text).strip()

    def process_file(self, file_path: str) -> Document:
        """Routes files to correct code rules based on their file extension formats."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        # Extract file extension safely
        ext = file_path.split(".")[-1].lower()
        
        # Route processing based on file types
        if ext in ["txt", "md"]:
            content = self.parse_txt_or_md(file_path)
        elif ext in ["html", "htm"]:
            content = self.parse_html(file_path)
        elif ext in ["pdf"]:
            content = self.parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format variant: .{ext}")

        # Safeguard against zero-content outputs
        if not content:
            raise ValueError(f"Aborting processing: Parsed content for {file_path} is completely empty.")

        # Build clean Pydantic model contract outputs
        metadata = DocumentMetadata(
            source_path=os.path.abspath(file_path),
            file_type=ext
        )
        return Document(page_content=content, metadata=metadata)
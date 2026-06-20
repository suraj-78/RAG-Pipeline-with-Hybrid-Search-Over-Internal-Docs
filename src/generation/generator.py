import os
import json
from typing import List, Dict, Any
# WE SWITCHED THE SDK: OpenAI se Groq par shift kiya
from groq import Groq
from src.config import config

class GroundedGenerator:
    """The execution core that enforces strict grounding boundaries onto open-source LLMs."""

    # We use llama-3.1-8b-instant which is fast, powerful, and free on Groq
    def __init__(self, model_name: str = config.GENERATION_MODEL_NAME):
        
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _build_system_prompt(self) -> str:
        return (
        "You are a production-grade, highly precise AI Technical Support Engineer.\n"
        "Your task is to synthesize a JSON answer for the user based strictly on the provided Context Blocks.\n\n"
        "CRITICAL DIRECTIVES:\n"
        "1. Every factual assertion MUST be immediately followed by its bracketed index (e.g., [1]).\n"
        "2. Strict Example of a valid answer string format:\n"
        "   'According to internal engineering protocols, server nodes must operate under port 8000 [1]. Failing to deploy native async wrappers will trigger instant firewall lockouts [2].'\n"
        "3. If context is insufficient, set is_context_sufficient to false and return exactly:\n"
        "   'I am sorry, but the internal documentation does not provide sufficient context to answer this question.'\n"
        "4. Return the response strictly as a JSON object matching this schema:\n"
        "   { 'answer': 'string', 'is_context_sufficient': boolean }"
    )

    def generate_answer(self, query: str, top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Injects contexts into structured prompts and handles Groq JSON mode transitions."""
        if not top_chunks:
            return {"answer": "Insufficient context.", "is_context_sufficient": False}

        formatted_context = ""
        for index, item in enumerate(top_chunks):
            chunk = item["chunk"]
            formatted_context += f"--- CONTEXT BLOCK [{index + 1}] ---\n{chunk.page_content}\n\n"

        user_content = f"User Query: {query}\n\nRetrieved Context Blocks:\n{formatted_context}"

        # Groq supports native JSON Mode execution
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}, # Force strict JSON validation
            temperature=0.0
        )

        # Parse the raw JSON string safely back into dictionary structures
        return json.loads(completion.choices[0].message.content)
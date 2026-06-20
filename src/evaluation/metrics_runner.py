import os
import json
from typing import List, Dict, Any
from groq import Groq

# Repository architecture path mapping
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator

class RAGMetricsRunner:
    """Automated LLM-as-a-Judge engine aligned with the repository architecture."""

    def __init__(self, retriever: HybridRetriever, reranker: DocumentReranker, generator: GroundedGenerator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        
        # CHANGED: Updated model identifier from decommissioned specdec to production versatile
        self.judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.judge_model = "llama-3.3-70b-versatile"

    def _execute_judge_pass(self, prompt: str) -> float:
        """Forces the LLM judge to output a deterministic JSON score."""
        try:
            completion = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            return float(data.get("score", 0.0))
        except Exception as e:
            print(f"[ERROR] Judge evaluation pass failed: {str(e)}")
            return 0.0

    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Measures if the answer is strictly grounded in the retrieved context."""
        joined_context = "\n".join(contexts)
        prompt = (
            f"You are an expert AI Auditor. Rate the FAITHFULNESS of the Answer based strictly on the Context.\n"
            f"Check if every claim in the answer is completely supported by the context. Ignore external knowledge.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return JSON format exactly: {{ 'score': float }} where score is between 0.0 (hallucinated) and 1.0 (perfectly grounded)."
        )
        return self._execute_judge_pass(prompt)

    def calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Measures how directly the answer addresses the user's explicit question."""
        prompt = (
            f"You are an expert AI Auditor. Rate the ANSWER RELEVANCY of the generated text to the User Query.\n"
            f"Check if the answer is direct, clear, and addresses the prompt without filler.\n\n"
            f"Query:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return JSON format exactly: {{ 'score': float }} where score is between 0.0 (irrelevant) and 1.0 (highly explicit)."
        )
        return self._execute_judge_pass(prompt)

    def run_batch_evaluation(self, test_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Executes evaluation across the entire golden test matrix."""
        results = []
        summary = {"avg_faithfulness": 0.0, "avg_relevancy": 0.0}

        for idx, test_case in enumerate(test_dataset):
            query = test_case["query"]
            print(f"[{idx + 1}/{len(test_dataset)}] Evaluating query: '{query}'")
            
            # 1. Pipeline retrieval loops
            hybrid_candidates = self.retriever.retrieve(query, top_k=10)
            elite_chunks = self.reranker.rerank(query, hybrid_candidates, top_n=3)
            context_strings = [item["chunk"].page_content for item in elite_chunks]
            
            # 2. Response synthesis
            llm_output = self.generator.generate_answer(query, elite_chunks)
            generated_answer = llm_output.get("answer", "") if isinstance(llm_output, dict) else llm_output.answer

            # 3. Metric calculation passes
            faithfulness = self.calculate_faithfulness(generated_answer, context_strings)
            relevancy = self.calculate_answer_relevancy(query, generated_answer)

            results.append({
                "sample_index": idx + 1,
                "query": query,
                "answer": generated_answer,
                "metrics": {"faithfulness": faithfulness, "answer_relevancy": relevancy}
            })
            
            summary["avg_faithfulness"] += faithfulness
            summary["avg_relevancy"] += relevancy

        total_samples = len(test_dataset) if test_dataset else 1
        summary["avg_faithfulness"] /= total_samples
        summary["avg_relevancy"] /= total_samples

        return {"summary": summary, "runs": results}

if __name__ == "__main__":
    print("Initializing components for local evaluation dataset passes...")
    from src.indexing.sparse import SparseBM25Index
    from src.indexing.dense import DenseVectorIndex

    sparse = SparseBM25Index()
    dense = DenseVectorIndex()
    sparse.load_index() # Pull local pickle indices

    retriever = HybridRetriever(sparse, dense)
    reranker = DocumentReranker()
    generator = GroundedGenerator()

    runner = RAGMetricsRunner(retriever, reranker, generator)

    # Golden Test Suite data
    golden_dataset = [
        {
            "query": "What happens if a student has less than 75% attendance?",
            "ground_truth": "They fail to qualify for recruitment drives and placement applications are dropped."
        },
        {
            "query": "What specific error code is triggered upon absolute API key exposure?",
            "ground_truth": "The system firewall lockouts will explicitly trigger error flag ERR_9082_F."
        }
    ]

    print(f"Running automated batch assessment loops over {len(golden_dataset)} samples...")
    report = runner.run_batch_evaluation(golden_dataset)
    
    print("\n================ SYSTEM SUMMARY MATRIX ================")
    print(json.dumps(report["summary"], indent=2))
    print("=======================================================\n")
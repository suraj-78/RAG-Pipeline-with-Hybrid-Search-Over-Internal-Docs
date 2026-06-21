import os
import json
import time  # FIXED: Added missing import for rate-limiting cooldown boundaries
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
        
        # Centralized versatile model mapping
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
        """Executes evaluation across the entire golden test matrix with live diagnostic telemetry."""
        results = []
        summary = {"avg_faithfulness": 0.0, "avg_relevancy": 0.0}

        # FIXED: Enforced bulletproof double-quoted JSON schema contract inside structural prompt tokens
        for idx, test_case in enumerate(test_dataset):
            query = test_case["query"]
            print(f"\n👉 [{idx + 1}/{len(test_dataset)}] Evaluating: Query: '{query[:60]}...'")
            
            try:
                # 1. Pipeline retrieval loops
                hybrid_candidates = self.retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K)
                elite_chunks = self.reranker.rerank(query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                context_strings = [item["chunk"].page_content for item in elite_chunks]
                
                # Diagnostic log to verify vector space loading
                print(f"   [DATA CHECK] Retrieved {len(context_strings)} active text chunks from indices.")

                # 2. Response synthesis
                llm_output = self.generator.generate_answer(query, elite_chunks)
                generated_answer = llm_output.get("answer", "") if isinstance(llm_output, dict) else llm_output.answer
                
                # CRITICAL DIAGNOSTIC PRINT: See what the system is actually replying!
                print(f"   [SYSTEM ANSWER]: {generated_answer}")

                # 3. Metric calculation prompts with double quote safety maps
                f_prompt = (
                    f"You are an expert AI Auditor. Rate the FAITHFULNESS of the Answer based strictly on the Context.\n"
                    f"Context:\n{chr(10).join(context_strings)}\n\n"
                    f"Answer:\n{generated_answer}\n\n"
                    f"Return JSON format exactly using double quotes: {{\n  \"score\": float\n}} between 0.0 and 1.0."
                )
                
                r_prompt = (
                    f"You are an expert AI Auditor. Rate the ANSWER RELEVANCY of the generated text to the User Query.\n"
                    f"Query:\n{query}\n\n"
                    f"Answer:\n{generated_answer}\n\n"
                    f"Return JSON format exactly using double quotes: {{\n  \"score\": float\n}} between 0.0 and 1.0."
                )

                faithfulness = self._execute_judge_pass(f_prompt)
                relevancy = self._execute_judge_pass(r_prompt)
                
                print(f"   [METRIC SCORES] -> Faithfulness: {faithfulness} | Relevancy: {relevancy}")

                results.append({
                    "sample_index": idx + 1,
                    "query": query,
                    "answer": generated_answer,
                    "metrics": {"faithfulness": faithfulness, "answer_relevancy": relevancy}
                })
                
                summary["avg_faithfulness"] += faithfulness
                summary["avg_relevancy"] += relevancy
                
            except Exception as loop_error:
                print(f"   [WARNING] Sample iteration index {idx+1} failed processing: {str(loop_error)}")
            
            # Cooldown sleep to completely bypass Groq free tier RPM limits
            time.sleep(3)

        total_samples = len(test_dataset) if test_dataset else 1
        summary["avg_faithfulness"] /= total_samples
        summary["avg_relevancy"] /= total_samples

        return {"summary": summary, "runs": results}

if __name__ == "__main__":
    print("Initializing components for local evaluation dataset passes...")
    from src.config import config
    from src.indexing.sparse import SparseBM25Index
    from src.indexing.dense import DenseVectorIndex

    sparse = SparseBM25Index()
    dense = DenseVectorIndex()
    sparse.load_index()  # Pull local pickle tokens from disk

    retriever = HybridRetriever(sparse, dense)
    reranker = DocumentReranker()
    generator = GroundedGenerator()

    runner = RAGMetricsRunner(retriever, reranker, generator)

    # FIXED: Comprehensive 50-sample Golden Test Suite matched to Policy-Document.pdf layers
    # src/evaluation/metrics_runner.py ke andar golden_dataset array ko replace karein:

    golden_dataset = [
    # === CODE OF CONDUCT & GENERAL DISCIPLINE ===
    {
        "query": "What is the primary objective of the college Code of Conduct?",
        "ground_truth": "The Code of Conduct outlines the ethical, academic, and behavioral standards expected from all students and stakeholders."
    },
    {
        "query": "What are the core working hours specified in the campus policy?",
        "ground_truth": "The standard working hours for academic sessions require students to be present inside classes during designated hours."
    },
    {
        "query": "Is biometric or manual attendance mandatory for students?",
        "ground_truth": "Students must maintain minimum prescribed attendance levels calculated through institutional logging records."
    },
    {
        "query": "What is the institutional policy regarding political activities inside the campus?",
        "ground_truth": "Political activities, unauthorized banners, and strikes are strictly prohibited inside the college campus premises."
    },
    {
        "query": "What disciplinary action is taken for property damage under the Code of Conduct?",
        "ground_truth": "Any damage caused to institutional property will attract strict financial penalties and suspension based on inquiry."
    },

    # === DIVYANGJAN POLICY ===
    {
        "query": "What facilities are provided to Divyangjan students under the institutional policy?",
        "ground_truth": "The institution provides ramp access, specialized restrooms, and structural infrastructure modifications for differently-abled individuals."
    },
    {
        "query": "Are screen-reading software facilities available for visually impaired students?",
        "ground_truth": "Yes, IT-enabled support centers provide screen readers and accessible learning materials under the Divyangjan assistance protocol."
    },
    {
        "query": "What is the policy regarding scribes during university examinations?",
        "ground_truth": "Eligible differently-abled students are granted permission to use scribes along with compensatory extra time during examinations."
    },
    {
        "query": "Who monitors the implementation of the Divyangjan welfare guidelines?",
        "ground_truth": "The Internal Quality Assurance Cell (IQAC) along with the infrastructure committee ensures compliance with accessibility rules."
    },
    {
        "query": "Does the college offer fee concessions or scholarships for differently-abled students?",
        "ground_truth": "Welfare-driven financial aids and assistive device support mechanisms are provisioned under specific student assistance schemes."
    },

    # === ANTI-RAGGING POLICY ===
    {
        "query": "What is the penalty for a student found guilty of ragging on campus?",
        "ground_truth": "Ragging is a criminal offense leading to immediate expulsion, suspension, and forwarding of the case to local police authorities."
    },
    {
        "query": "Who heads the institutional Anti-Ragging Committee?",
        "ground_truth": "The Principal heads the Anti-Ragging Committee along with faculty representatives, civil administration, and police officials."
    },
    {
        "query": "How can a victim lodge an anonymous complaint against ragging?",
        "ground_truth": "Complaints can be dropped into dedicated anti-ragging boxes, reported to squad members, or submitted via the institutional portal."
    },
    {
        "query": "Are affidavits regarding anti-ragging mandatory during admission?",
        "ground_truth": "Yes, both students and parents must sign and execute statutory anti-ragging affidavits at the time of enrollment."
    },
    {
        "query": "What are the duties of the anti-ragging squad?",
        "ground_truth": "The squad conducts surprise raids across hostels, canteens, and recreational areas to prevent any behavioral misconduct."
    },

    # === DISCIPLINE & GRIEVANCE REDRESSAL POLICY ===
    {
        "query": "What is the function of the Grievance Redressal Cell?",
        "ground_truth": "The cell evaluates academic, financial, or administrative complaints raised by students to ensure a fair and timely resolution."
    },
    {
        "query": "Within how many days must a student grievance be processed?",
        "ground_truth": "Grievances must be investigated and resolved within the stipulated statutory timeline specified by the committee router."
    },
    {
        "query": "Can parents directly approach the grievance redressal cell?",
        "ground_truth": "Parents can submit structural feedback or appeals during designated open hours or scheduled PTA review meets."
    },
    {
        "query": "Who is the final authority for appellate redressal inside the college?",
        "ground_truth": "The Governing Body and the Principal hold the absolute authority for final arbitration on unresolved disputes."
    },
    {
        "query": "What mechanisms exist to handle cases of sexual harassment?",
        "ground_truth": "The Internal Complaints Committee (ICC) handles all complaints under strict confidentiality matching the POSH framework standards."
    },

    # === INFRASTRUCTURE & IT-CYBER SECURITY POLICY ===
    {
        "query": "What are the rules regarding campus Wi-Fi network utilization?",
        "ground_truth": "The institutional Wi-Fi network must strictly be utilized for academic purposes; accessing restricted or unauthorized portals is banned."
    },
    {
        "query": "Who manages the maintenance allocations for lab infrastructures?",
        "ground_truth": "The Infrastructure and Maintenance Committee evaluates resource conditions and coordinates annual repair workflows."
    },
    {
        "query": "Is unauthorized software installation allowed on laboratory systems?",
        "ground_truth": "No, any software injection outside the designated IT supervisor whitelist is tracked as a security breach."
    },
    {
        "query": "How are hardware lifecycle replacements handled under the IT policy?",
        "ground_truth": "Outdated equipment undergoes scheduled audit classification, asset decommissioning, and structured e-waste management disposal."
    },
    {
        "query": "What security tracking measures are applied to monitor digital systems?",
        "ground_truth": "Network traffic logging, firewall security rule configurations, and centralized domain access controllers are active."
    },

    # === EXAMINATION POLICIES (INTERNAL & UNIVERSITY) ===
    {
        "query": "What is the minimum attendance requirement to sit for university examinations?",
        "ground_truth": "Students must achieve a strict minimum of 75% attendance across courses to be certified eligible for university exams."
    },
    {
        "query": "How are internal assessment marks computed under the exam policy?",
        "ground_truth": "Internal evaluations fuse weightage components spanning periodic test scores, descriptive assignments, and session attendance metrics."
    },
    {
        "query": "What happens if a student misses an internal exam due to medical emergencies?",
        "ground_truth": "On submission of validated medical certificates, the department provisions specific re-test validation cycles."
    },
    {
        "query": "What is the penalty for adopting malpractices during examinations?",
        "ground_truth": "Malpractice leads to automatic cancellation of the paper, barring from subsequent exams, and forwarding to the university disciplinary board."
    },
    {
        "query": "Who oversees the secure storage and printing of evaluation sheets?",
        "ground_truth": "The Chief Superintendent of Examinations along with the designated exam cell controllers secures confidential materials."
    },

    # === MENTORING & PLACEMENT CELL POLICIES ===
    {
        "query": "What is the structural ratio configuration of the Mentoring Policy?",
        "ground_truth": "A specified cluster of students is assigned to a dedicated faculty mentor to monitor psychological and academic progress."
    },
    {
        "query": "How frequently must a faculty mentor conduct tracking sessions?",
        "ground_truth": "Mentors must execute tracking checkpoints periodically each semester, maintaining updated student metric ledgers."
    },
    {
        "query": "What eligibility parameters are tracked by the Placement Cell?",
        "ground_truth": "The placement node maps cgpa cutoffs, active backlog tracking counts, and professional skill metrics."
    },
    {
        "query": "Are internship certificates mandatory for placement registration?",
        "ground_truth": "Yes, completing designated industrial training workflows is prerequisite to entering recruitment drives."
    },
    {
        "query": "Who coordinates training sessions for soft skill developments?",
        "ground_truth": "The Placement Cell collaborates with specialized training vendors to execute corporate alignment workshops."
    },

    # === STUDENT ASSISTANCE & WELFARE SCHEMES ===
    {
        "query": "What is the focus of the Vincent De Paul assistance scheme?",
        "ground_truth": "The scheme targets socio-economically vulnerable student groups to provision educational subsidies and resource distributions."
    },
    {
        "query": "How are candidates selected for the Student Assistance Scheme?",
        "ground_truth": "An income transparency audit and family tracking metrics are verified by the welfare evaluation board."
    },
    {
        "query": "What endowment parameters govern specific academic awards?",
        "ground_truth": "Endowment allocations are locked to track top-ranking students across specific academic domain concentrations."
    },
    {
        "query": "Are student welfare finances externally audited?",
        "ground_truth": "Yes, all institutional fund management flows undergo strict regulatory oversight and external auditing checks annually."
    },
    {
        "query": "Can a student avail multiple institutional scholarships simultaneously?",
        "ground_truth": "No, policy parameters prohibit stacking multiple internal financial aids to guarantee fair distribution of welfare funds."
    },

    # === CANTEEN MANAGEMENT, OPERATIONS & AUDITING ===
    {
        "query": "How is plastic and paper waste of the canteen handled according to the policy?",
        "ground_truth": "The paper & plastic wastes of the canteen are separately segregated and handed over to Kudumbasree workers."
    },
    {
        "query": "Where is the kitchen water from the canteen utilized?",
        "ground_truth": "The kitchen water collected is used to water the college vegetable garden."
    },
    {
        "query": "Who holds the responsibility to prepare the annual budget of the canteen?",
        "ground_truth": "The canteen supervisor is responsible for preparing the annual budget."
    },
    {
        "query": "What happens to the food wastes generated from the canteen?",
        "ground_truth": "The food wastes from the canteen are fed to the biogas plant of the college."
    },
    {
        "query": "Who reviews the canteen expenditures and cash flow, and how often?",
        "ground_truth": "The Bursar is to review expenditure and cash flow once a term."
    },
    {
        "query": "What are the allowed payment methods at the college canteen?",
        "ground_truth": "Payment can be done either as cash or via Google Pay."
    },
    {
        "query": "How frequently is a stock take performed by the canteen supervisor?",
        "ground_truth": "The canteen supervisor will perform a stock take at the end of each semester."
    },
    {
        "query": "Who performs the operational audit of the canteen?",
        "ground_truth": "The canteen supervisor along with the canteen committee members perform the audit using profit and loss statements."
    },
    {
        "query": "What happens to the money from the canteen at the close of trading each day?",
        "ground_truth": "All money from the canteen will be counted and signed for by two people at the close of trading each day."
    },
    {
        "query": "What system is used to log the financial records of the canteen?",
        "ground_truth": "Accurate records are to be kept of money received and expended by way of the college's financial data package."
    }
]

    print(f"Running automated batch assessment loops over {len(golden_dataset)} samples...")
    report = runner.run_batch_evaluation(golden_dataset)
    
    print("\n================ SYSTEM SUMMARY MATRIX ================")
    print(json.dumps(report["summary"], indent=2))
    print("=======================================================\n")
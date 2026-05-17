"""
src/evaluation/evaluate.py
--------------------------
Person B — Week 4
RAG + SQL evaluation framework.
Measures: faithfulness, answer relevancy, context precision (via Ragas).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

GROUND_TRUTH_PATH = Path("src/evaluation/ground_truth.json")


@dataclass
class TestCase:
    question: str
    ground_truth: str
    category: str  # "sql" or "rag"
    expected_metrics: list[str] = field(default_factory=list)  # e.g. ["churn_rate", "ROI"]


@dataclass
class EvalResult:
    question: str
    category: str
    predicted_answer: str
    ground_truth: str
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    passed: bool = False


def load_ground_truth() -> list[TestCase]:
    with open(GROUND_TRUTH_PATH, "r") as f:
        data = json.load(f)
    return [TestCase(**item) for item in data["test_cases"]]


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run Ragas evaluation on RAG outputs.
    Returns a dict of metric_name -> average_score.

    Install: pip install ragas
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return dict(result)

    except ImportError:
        print("[WARN] ragas not installed. Run: pip install ragas datasets")
        return {}


def simple_sql_eval(predicted: str, ground_truth: str) -> bool:
    """
    Simple keyword-match evaluation for SQL results.
    Checks if key numbers/values from ground truth appear in the prediction.
    """
    # Extract numbers from ground truth and check they appear in prediction
    import re
    gt_numbers = re.findall(r"\d+\.?\d*", ground_truth)
    pred_numbers = re.findall(r"\d+\.?\d*", predicted)

    if not gt_numbers:
        return True  # No numbers to check

    matches = sum(1 for n in gt_numbers if n in pred_numbers)
    return matches / len(gt_numbers) >= 0.7  # 70% number match threshold


def generate_eval_report(results: list[EvalResult]) -> dict:
    """Summarise evaluation results into a report dict."""
    sql_results = [r for r in results if r.category == "sql"]
    rag_results = [r for r in results if r.category == "rag"]

    def avg(scores):
        scores = [s for s in scores if s is not None]
        return round(sum(scores) / len(scores), 3) if scores else None

    report = {
        "total_questions": len(results),
        "sql": {
            "count": len(sql_results),
            "pass_rate": round(sum(r.passed for r in sql_results) / len(sql_results), 3) if sql_results else 0,
        },
        "rag": {
            "count": len(rag_results),
            "faithfulness": avg([r.faithfulness_score for r in rag_results]),
            "answer_relevancy": avg([r.answer_relevancy_score for r in rag_results]),
            "context_precision": avg([r.context_precision_score for r in rag_results]),
        },
    }
    return report

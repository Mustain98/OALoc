from src.eval.evaluate import evaluate
from src.schemas import Instance, Prediction


def test_metrics():
    gold = [Instance("i1", "r", "c", "report",
                     gold_files=["a.py"], gold_functions=["a.py:f"])]
    preds = [Prediction("i1", ranked_files=["a.py", "b.py"],
                        ranked_functions=["a.py:f"])]
    metrics = evaluate(preds, gold)
    assert metrics["acc@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["function_acc@1"] == 1.0
    assert metrics["function_mrr"] == 1.0


def test_miss():
    gold = [Instance("i1", "r", "c", "report", gold_files=["a.py"])]
    preds = [Prediction("i1", ranked_files=["z.py", "a.py"],
                        ranked_functions=[])]
    metrics = evaluate(preds, gold)
    assert metrics["acc@1"] == 0.0
    assert metrics["acc@3"] == 1.0
    assert metrics["mrr"] == 0.5


def test_unknown_predictions_are_ignored():
    gold = [Instance("i1", "r", "c", "report", gold_files=["a.py"])]
    preds = [Prediction("unknown", ranked_files=["a.py"], ranked_functions=[])]
    assert evaluate(preds, gold)["n"] == 0

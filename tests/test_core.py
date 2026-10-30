"""Offline regression tests for W&B selection, metrics and filtering."""
import os
import sys
import types
import unittest
from unittest.mock import patch

from core import TARGET, _config_str, _model_name, aggregate, fetch, metric_for


def run(model, dataset, transport, run_id, **metrics):
    return {
        "state": "finished",
        "model": model,
        "dataset": dataset,
        "transport_type": transport,
        "run_id": run_id,
        "name": "run-" + run_id,
        "url": "https://example.invalid/" + run_id,
        **metrics,
    }


class AggregateTests(unittest.TestCase):
    def test_best_validation_not_best_test_and_matched_mean_std(self):
        items = [
            run("NSD", "cora", "orthogonal", "a",
                mean_val_acc=0.83, mean_test_acc=0.99, std_test_acc=0.001),
            run("NSD", "cora", "orthogonal", "b",
                mean_val_acc=0.91, mean_test_acc=0.80, std_test_acc=0.04),
        ]
        result = aggregate(items)[0]
        self.assertEqual(result["finished"], 2)
        self.assertEqual(result["remaining"], TARGET - 2)
        self.assertEqual(result["best_run_id"], "b")
        self.assertEqual(result["best_validation"], 0.91)
        self.assertEqual(result["mean_test_acc"], 0.80)
        self.assertEqual(result["std_test_acc"], 0.04)
        self.assertIsNone(result["mean_test_roc_auc"])

    def test_roc_datasets_case_insensitive(self):
        for dataset in ("minesweeper", "tolokers", "questions", " Questions "):
            with self.subTest(dataset=dataset):
                items = [
                    run("GAT", dataset, "general", "a",
                        mean_val_acc=0.99, mean_val_roc_auc=0.66,
                        mean_test_roc_auc=0.95, std_test_roc_auc=0.01),
                    run("GAT", dataset, "general", "b",
                        mean_val_acc=0.12, mean_val_roc_auc=0.80,
                        mean_test_roc_auc=0.73, std_test_roc_auc=0.05),
                ]
                result = aggregate(items)[0]
                self.assertEqual(result["best_run_id"], "b")
                self.assertEqual(result["selection_metric"], "mean_val_roc_auc")
                self.assertEqual(result["mean_test_roc_auc"], 0.73)
                self.assertEqual(result["std_test_roc_auc"], 0.05)
                self.assertIsNone(result["mean_test_acc"])

    def test_all_models_and_combinations_are_included(self):
        items = [
            run("NSD", "cora", "diagonal", "a", mean_val_acc=0.50),
            run("ESNN", "cora", "diagonal", "b", mean_val_acc=0.70),
            run("SAGE", "cora", "general", "c", mean_val_acc=0.40),
        ]
        result = aggregate(items)
        self.assertEqual({x["model"] for x in result}, {"NSD", "ESNN", "SAGE"})
        self.assertEqual(len(result), 3)

    def test_missing_nan_and_infinite_validation_not_selected(self):
        items = [
            run("CNN", "citeseer", "lowrank", "b", mean_val_acc=float("nan"), mean_test_acc=0.8),
            run("CNN", "citeseer", "lowrank", "c", mean_val_acc=float("inf"), mean_test_acc=0.9),
        ]
        result = aggregate(items)[0]
        self.assertIsNone(result["best_validation"])
        self.assertIsNone(result["best_run_id"])
        self.assertIsNone(result["mean_test_acc"])
        self.assertEqual(result["finished"], 2)

    def test_tie_breaker_is_deterministic(self):
        items = [
            run("ESNN", "cora", "general", "b", mean_val_acc=0.8, mean_test_acc=0.5),
            run("ESNN", "cora", "general", "a", mean_val_acc=0.8, mean_test_acc=0.4),
        ]
        self.assertEqual(aggregate(items)[0]["best_run_id"], "a")
        self.assertEqual(aggregate(list(reversed(items)))[0]["best_run_id"], "a")

    def test_wrapped_dataset_and_non_nsd_model_config(self):
        self.assertEqual(_config_str({"dataset": {"value": "minesweeper", "_type": "str"}}, "dataset"), "minesweeper")
        self.assertEqual(_model_name({"model_name": "ESNN"}), "ESNN")
        self.assertEqual(_model_name({"model": {"name": "GAT"}}), "GAT")
        self.assertEqual(metric_for("tolokers")[0], "mean_val_roc_auc")


class FetchTests(unittest.TestCase):
    def test_fetch_filters_creator_state_and_keeps_non_nsd_models(self):
        def fake_user(name):
            return {"username": name}
        def fake_run(user, model, run_id, state="finished"):
            return types.SimpleNamespace(
                _attrs={"user": fake_user(user)},
                config={"model_name": model, "dataset": "cora", "transport_type": "general"},
                summary={"mean_val_acc": 0.7, "mean_test_acc": 0.65, "std_test_acc": 0.03},
                state=state, id=run_id, name="run-" + run_id,
                url="https://example.invalid/" + run_id,
            )
        fake_runs = [
            fake_run("fs395", "ESNN", "a"),
            fake_run("fs395", "Transformer", "b"),
            fake_run("other", "NSD", "c"),
            fake_run("fs395", "NSD", "d", state="running"),
            fake_run(None, "NSD", "e"),
        ]
        captured = {}
        class FakeApi:
            def __init__(self, **kwargs):
                captured["api_args"] = kwargs
            def runs(self, project, **kwargs):
                captured["project"] = project
                captured["kwargs"] = kwargs
                return fake_runs

        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}):
            with patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=FakeApi)}):
                result = fetch()
        self.assertEqual(captured["project"], "adavit/sheaf-mpnn-runs")
        self.assertEqual(captured["kwargs"]["filters"], {"state": "finished", "username": "fs395"})
        self.assertIs(captured["kwargs"]["lazy"], False)
        self.assertEqual({x["model"] for x in result["runs"]}, {"ESNN", "Transformer"})
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["examined"], 5)


if __name__ == "__main__":
    unittest.main()

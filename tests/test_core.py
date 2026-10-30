"""Offline tests of the memory-bounded W&B monitor."""
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

from core import (
    CREATOR, TARGET, _model_name, _number, compact_from_pages,
    fetch, group_id, invalid_metric_fields, metric_for, SAMPLE_SIZE,
)


def node(run_id, model="NSD", dataset="cora", transport="diagonal",
         username="fs395", state="finished", **metrics):
    return {
        "name": run_id,
        "displayName": "run " + run_id,
        "state": state,
        "user": {"username": username} if username else None,
        "config": json.dumps({
            "model": {"value": model},
            "dataset": {"value": dataset},
            "transport_type": {"value": transport},
        }),
        "summaryMetrics": json.dumps(metrics),
    }


class CompactTests(unittest.TestCase):
    def test_accuracy_selects_best_validation_not_best_test(self):
        data = compact_from_pages([[
            node("a", mean_val_acc=.81, mean_test_acc=.97, std_test_acc=.01),
            node("b", mean_val_acc=.90, mean_test_acc=.77, std_test_acc=.05),
        ]])
        self.assertEqual(data["matched"], 2)
        group = data["groups"][0]
        self.assertEqual(group["best_run_id"], "b")
        self.assertEqual(group["mean_test_acc"], .77)
        self.assertEqual(group["std_test_acc"], .05)
        self.assertEqual(group["remaining"], TARGET - 2)

    def test_roc_datasets_use_validation_roc_auc(self):
        for name in ("minesweeper", "tolokers", "questions", " Questions "):
            with self.subTest(name=name):
                data = compact_from_pages([[
                    node("a", dataset=name, mean_val_acc=.99, mean_val_roc_auc=.61,
                         mean_test_roc_auc=.91, std_test_roc_auc=.01),
                    node("b", dataset=name, mean_val_acc=.21, mean_val_roc_auc=.81,
                         mean_test_roc_auc=.73, std_test_roc_auc=.06),
                ]])
                row = data["groups"][0]
                self.assertEqual(row["best_run_id"], "b")
                self.assertEqual(row["mean_test_roc_auc"], .73)
                self.assertEqual(row["std_test_roc_auc"], .06)
                self.assertIsNone(row["mean_test_acc"])

    def test_other_model_families_are_included(self):
        data = compact_from_pages([[
            node("a", model="NSD"),
            node("b", model="ESNN"),
            node("c", model="GAT"),
        ]])
        self.assertEqual({r["model"] for r in data["groups"]}, {"NSD", "ESNN", "GAT"})

    def test_server_data_is_rechecked_for_creator_and_finished_state(self):
        data = compact_from_pages([[
            node("a", username="fs395"),
            node("b", username="different"),
            node("c", state="running"),
            node("d", username=None),
        ]])
        self.assertEqual(data["examined"], 4)
        self.assertEqual(data["matched"], 1)
        self.assertEqual(data["unknown"], 1)
        self.assertEqual(data["groups"][0]["finished"], 1)

    def test_no_validation_means_blank_best_and_diagnostic_metadata(self):
        data = compact_from_pages([[
            node("a", dataset="cornell", mean_test_acc=.80),
            node("b", dataset="cornell", mean_val_acc=float("nan")),
        ]])
        row = data["groups"][0]
        self.assertEqual(row["finished"], 2)
        self.assertEqual(row["valid_validation_runs"], 0)
        self.assertIsNone(row["best_run_id"])
        self.assertIsNone(row["mean_test_acc"])
        self.assertEqual(row["invalid_metric_runs"], 2)
        self.assertTrue(row["invalid_samples"])

    def test_memory_is_bounded_by_groups_and_samples(self):
        pages = (
            [node(str(i), mean_val_acc=i / 2000, mean_test_acc=.5, std_test_acc=.01)
             for i in range(start, start + 30)]
            for start in range(0, 1200, 30)
        )
        data = compact_from_pages(pages)
        self.assertEqual(data["matched"], 1200)
        self.assertEqual(len(data["groups"]), 1)
        group = data["groups"][0]
        self.assertLessEqual(len(group["sample_runs"]), SAMPLE_SIZE)
        self.assertLessEqual(len(group["invalid_samples"]), SAMPLE_SIZE)
        self.assertEqual(group["finished"], 1200)
        self.assertEqual(group["best_run_id"], "1199")
        self.assertEqual(group["remaining"], 0)
        self.assertNotIn("runs", data)

    def test_tie_break_by_run_id(self):
        data = compact_from_pages([[
            node("b", mean_val_acc=.7, mean_test_acc=.6),
            node("a", mean_val_acc=.7, mean_test_acc=.5),
        ]])
        self.assertEqual(data["groups"][0]["best_run_id"], "a")

    def test_missing_and_nonfinite_scores(self):
        self.assertIsNone(_number(None))
        self.assertIsNone(_number(float("inf")))
        self.assertIsNone(_number("NaN"))
        self.assertEqual(
            invalid_metric_fields({
                "dataset": "tolokers", "mean_val_roc_auc": .7,
                "mean_test_roc_auc": .5, "std_test_roc_auc": None,
            }),
            ("std_test_roc_auc",),
        )

    def test_model_name_fallback(self):
        self.assertEqual(_model_name({"model_name": "ESNN"}), "ESNN")
        self.assertEqual(_model_name({"model": {"name": "GAT"}}), "GAT")

    def test_fetch_uses_graphql_pagination_without_wandb_runs_iterator(self):
        call_vars = []
        class FakeService:
            def execute_graphql(self, query, variables=None):
                call_vars.append(dict(variables))
                if variables["cursor"] is None:
                    return {"project": {"runs": {
                        "edges": [{"cursor": "c1", "node": node("a")}],
                        "pageInfo": {"endCursor": "c1", "hasNextPage": True},
                    }}}
                return {"project": {"runs": {
                    "edges": [{"cursor": "c2", "node": node("b", model="ESNN")}],
                    "pageInfo": {"endCursor": "c2", "hasNextPage": False},
                }}}
        class FakeApi:
            def __init__(self, **kwargs):
                self._service_api = FakeService()
            def runs(self, *args, **kwargs):
                raise AssertionError("The in-memory Runs iterator must never be used")
        with patch.dict(os.environ, {"WANDB_API_KEY": "fake-key"}):
            with patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=FakeApi)}):
                data = fetch()
        self.assertEqual(len(call_vars), 2)
        self.assertEqual(call_vars[1]["cursor"], "c1")
        self.assertEqual(json.loads(call_vars[0]["filters"]), {
            "state": "finished", "username": CREATOR,
        })
        self.assertEqual(data["matched"], 2)
        self.assertEqual({g["model"] for g in data["groups"]}, {"NSD", "ESNN"})


if __name__ == "__main__":
    unittest.main()

"""Repository-derived model-specific benchmark coverage regression tests."""
import unittest

from benchmark_coverage import (
    MODELS, MODEL_VARIANTS, UNVERIFIED_MODELS,
    PEI_DATASETS, PLATONOV_DATASETS, FILTERED_DATASETS,
    RESTRICTION_MAPS, coverage_matrix, normalize_dataset,
    normalize_restriction_map, normalize_model, unverified_observations,
)


def group(model, dataset, map_type, count=1, state="finished"):
    return {
        "state": state, "model": model, "dataset": dataset,
        "transport_type": map_type, "finished": count,
    }


class CoverageTests(unittest.TestCase):
    def test_verified_model_specific_variant_matrix(self):
        self.assertEqual(len(MODELS), 8)
        self.assertEqual(len(UNVERIFIED_MODELS), 3)
        self.assertEqual(len(PEI_DATASETS), 6)
        self.assertEqual(len(PLATONOV_DATASETS), 5)
        self.assertEqual(MODEL_VARIANTS["nsp"], RESTRICTION_MAPS)
        self.assertEqual(MODEL_VARIANTS["poly_nsd"], RESTRICTION_MAPS)
        self.assertEqual(MODEL_VARIANTS["bsnn"], RESTRICTION_MAPS)
        self.assertEqual(MODEL_VARIANTS["id_sheaf"], ("diagonal",))
        self.assertEqual(MODEL_VARIANTS["conn_nsd"], ("orthogonal",))
        self.assertEqual(MODEL_VARIANTS["joint_diff"], ("general",))
        self.assertEqual(MODEL_VARIANTS["rot_inv"], ("general",))
        self.assertEqual(MODEL_VARIANTS["sheaf_attn"], ("res_sheafan",))

    def test_correct_total_and_no_phantom_maps(self):
        rows = coverage_matrix([])
        self.assertEqual(len(rows), 11 * (3 * 4 + 5))
        self.assertEqual(len(rows), 187)
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        self.assertIn(("id_sheaf", "texas", "diagonal"), lookup)
        self.assertNotIn(("id_sheaf", "texas", "general"), lookup)
        self.assertNotIn(("conn_nsd", "cornell", "diagonal"), lookup)
        self.assertIn(("conn_nsd", "cornell", "orthogonal"), lookup)
        self.assertIn(("sheaf_attn", "film", "res_sheafan"), lookup)
        self.assertNotIn(("sheaf_attn", "film", "orthogonal"), lookup)
        self.assertNotIn(("joint_diff", "film", "orthogonal"), lookup)
        self.assertNotIn(("rot_inv", "film", "orthogonal"), lookup)
        self.assertTrue(all(r["status"] == "Missing" for r in rows))

    def test_filtered_datasets_are_optional(self):
        rows = coverage_matrix([], include_filtered=True)
        self.assertEqual(len(rows), 13 * 17)
        self.assertEqual(len(rows), 221)
        self.assertTrue(any(r["dataset"] == "squirrel_filtered" for r in rows))

    def test_existing_incomplete_complete_missing(self):
        rows = coverage_matrix([
            group("poly_nsd", "cornell", "diagonal", 90),
            group("bsnn", "tolokers", "general", 260),
            group("id_sheaf", "texas", "diagonal", 250),
            group("conn_nsd", "texas", "orthogonal", 22),
        ])
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        incomplete = lookup["poly_nsd", "cornell", "diagonal"]
        self.assertEqual(incomplete["missing_runs"], 160)
        self.assertEqual(incomplete["status"], "Incomplete")
        self.assertEqual(lookup["bsnn", "tolokers", "general"]["status"], "Complete")
        self.assertEqual(lookup["id_sheaf", "texas", "diagonal"]["status"], "Complete")
        self.assertEqual(lookup["conn_nsd", "texas", "orthogonal"]["finished_runs"], 22)

    def test_sheaf_attention_name_alias_and_source_variant(self):
        rows = coverage_matrix([
            group("sheaf_attn", "cornell", "res_sheafan", 50),
            group("sheaf_att", "cornell", "res-sheafan", 75),
            group("sheaf_attn", "cornell", "sheafan", 99),
        ])
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        self.assertEqual(lookup["sheaf_attn", "cornell", "res_sheafan"]["finished_runs"], 125)
        self.assertEqual(lookup["sheaf_attn", "cornell", "res_sheafan"]["missing_runs"], 125)
        self.assertNotIn(("sheaf_attn", "cornell", "sheafan"), lookup)

    def test_unverified_models_are_not_phantom_expected_runs(self):
        groups = [
            group("sheaf_gin", "film", "lowrank", 80),
            group("sheaf_set_transformer", "cornell", "general", 7),
            group("sheaf_deepset", "texas", "diagonal", 30),
        ]
        verified = coverage_matrix(groups)
        self.assertFalse(any(r["model"] in UNVERIFIED_MODELS for r in verified))
        observed = unverified_observations(groups)
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(r["verification"].startswith("Observed") for r in observed))

    def test_dataset_and_map_aliases(self):
        rows = coverage_matrix([
            group("nsp", "Actor", "low-rank", 80),
            group("NSP", "film", "lowrank", 40),
            group("rot_inv", "amazon-ratings", "general", 4),
            group("nsp", "cornell", "diagonal", 250, state="running"),
        ])
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        self.assertEqual(lookup["nsp", "film", "low_rank"]["finished_runs"], 120)
        self.assertEqual(lookup["rot_inv", "amazon_ratings", "general"]["finished_runs"], 4)
        self.assertEqual(lookup["nsp", "cornell", "diagonal"]["finished_runs"], 0)
        self.assertEqual(normalize_model("sheaf_att"), "sheaf_attn")
        self.assertEqual(normalize_dataset("actor"), "film")
        self.assertEqual(normalize_restriction_map("lowrank"), "low_rank")


if __name__ == "__main__":
    unittest.main()

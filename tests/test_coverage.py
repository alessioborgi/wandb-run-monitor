"""Expected Pei/Platonov benchmark coverage regression tests."""
import unittest

from benchmark_coverage import (
    MODELS, PEI_DATASETS, PLATONOV_DATASETS, FILTERED_DATASETS,
    RESTRICTION_MAPS, coverage_matrix, normalize_dataset,
    normalize_restriction_map,
)


def group(model, dataset, map_type, count=1, state="finished"):
    return {
        "state": state,
        "model": model,
        "dataset": dataset,
        "transport_type": map_type,
        "finished": count,
    }


class CoverageTests(unittest.TestCase):
    def test_canonical_cartesian_product(self):
        rows = coverage_matrix([])
        self.assertEqual(len(MODELS), 11)
        self.assertEqual(len(PEI_DATASETS), 6)
        self.assertEqual(len(PLATONOV_DATASETS), 5)
        self.assertEqual(len(RESTRICTION_MAPS), 4)
        self.assertEqual(len(rows), 11 * 11 * 4)
        self.assertTrue(all(r["status"] == "Missing" for r in rows))
        self.assertTrue(all(r["missing_runs"] == 250 for r in rows))

    def test_optional_cleaned_variants(self):
        rows = coverage_matrix([], include_filtered=True)
        self.assertEqual(len(rows), 11 * 13 * 4)
        self.assertTrue(any(r["dataset"] == "chameleon_filtered" for r in rows))
        self.assertTrue(any(r["dataset"] == "squirrel_filtered" for r in rows))

    def test_existing_incomplete_complete_and_missing(self):
        rows = coverage_matrix([
            group("poly_nsd", "cornell", "diagonal", 90),
            group("bsnn", "tolokers", "general", 260),
        ])
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        incomplete = lookup["poly_nsd", "cornell", "diagonal"]
        self.assertEqual(incomplete["status"], "Incomplete")
        self.assertEqual(incomplete["finished_runs"], 90)
        self.assertEqual(incomplete["missing_runs"], 160)
        complete = lookup["bsnn", "tolokers", "general"]
        self.assertEqual(complete["status"], "Complete")
        self.assertEqual(complete["missing_runs"], 0)
        self.assertEqual(
            lookup["joint_diff", "texas", "orthogonal"]["status"], "Missing"
        )

    def test_aliases_for_actor_and_low_rank_and_dataset_names(self):
        rows = coverage_matrix([
            group("SHEAF-GIN", "Actor", "low-rank", 80),
            group("sheaf_gin", "Film", "LOWRANK", 40),
            group("sheaf_att", "roman-empire", "general", 25),
            group("rot_inv", "amazon-ratings", "orthogonal", 5),
        ])
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        self.assertEqual(lookup["sheaf_gin", "film", "low_rank"]["finished_runs"], 120)
        self.assertEqual(lookup["sheaf_gin", "film", "low_rank"]["missing_runs"], 130)
        self.assertEqual(lookup["sheaf_att", "roman_empire", "general"]["finished_runs"], 25)
        self.assertEqual(lookup["rot_inv", "amazon_ratings", "orthogonal"]["finished_runs"], 5)

    def test_nsp_is_not_nsd_and_unrelated_runs_are_ignored(self):
        rows = coverage_matrix([
            group("nsd", "cornell", "diagonal", 250),
            group("nsp", "cornell", "diagonal", 3),
            group("unlisted", "cornell", "diagonal", 600),
            group("nsp", "cornell", "orthogonal", 250, state="running"),
        ])
        self.assertTrue(all(r["model"] != "nsd" for r in rows))
        lookup = {(r["model"], r["dataset"], r["restriction_maps"]): r for r in rows}
        self.assertEqual(lookup["nsp", "cornell", "diagonal"]["finished_runs"], 3)
        self.assertEqual(lookup["nsp", "cornell", "orthogonal"]["finished_runs"], 0)

    def test_normalization_does_not_collapse_distinct_datasets(self):
        self.assertEqual(normalize_dataset("actor"), "film")
        self.assertEqual(normalize_dataset("chameleon_filtered"), "chameleon_filtered")
        self.assertNotEqual(normalize_dataset("chameleon"), normalize_dataset("chameleon_filtered"))
        self.assertEqual(normalize_restriction_map("low-rank"), "low_rank")


if __name__ == "__main__":
    unittest.main()

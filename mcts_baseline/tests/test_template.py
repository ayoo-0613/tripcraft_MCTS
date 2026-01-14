import unittest

from TripCraft.mcts_baseline.io import TripCraftRow
from TripCraft.mcts_baseline.templater import make_output_record_template


class TestTemplate(unittest.TestCase):
    def test_template_shape(self):
        row = TripCraftRow(
            idx=1,
            org="A",
            dest="B",
            days=3,
            visiting_city_number=1,
            date=["2024-01-01", "2024-01-02", "2024-01-03"],
            people_number=1,
            local_constraint={"cuisine": None},
            budget=1000.0,
            query=None,
            level="easy",
            persona="P",
            ref_blocks=["[]"],
        )
        rec = make_output_record_template(row)
        self.assertEqual(rec["idx"], 1)
        self.assertEqual(len(rec["plan"]), 3)
        for d in rec["plan"]:
            for key in (
                "days",
                "current_city",
                "transportation",
                "breakfast",
                "attraction",
                "lunch",
                "dinner",
                "accommodation",
                "event",
                "point_of_interest_list",
            ):
                self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()


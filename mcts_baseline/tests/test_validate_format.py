import unittest

from TripCraft.mcts_baseline.io import TripCraftRow
from TripCraft.mcts_baseline.templater import make_output_record_template
from TripCraft.mcts_baseline.validate import validate_record


class TestValidateFormat(unittest.TestCase):
    def test_validate_template_ok(self):
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
        errs = validate_record(rec)
        self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()


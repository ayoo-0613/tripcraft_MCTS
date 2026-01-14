import unittest

from TripCraft.mcts_baseline.ref_parser import build_unified_kb, parse_ref_block


class TestRefParser(unittest.TestCase):
    def test_parse_ref_block_json(self):
        blk = '[{\"Description\":\"Restaurants in X\",\"Content\":\"name cuisines avg_cost rating\\nFoo [\\\"Italian\\\"] 10.0 4.5\"}]'
        items = parse_ref_block(blk)
        self.assertEqual(len(items), 1)
        self.assertIn("Description", items[0])

    def test_build_unified_kb_tables_and_transport(self):
        ref1 = (
            '['
            '{\"Description\":\"Restaurants in City1\",\"Content\":\"name cuisines avg_cost rating\\nFoo [\\\"Italian\\\"] 10.0 4.5\"},'
            '{\"Description\":\"Accommodations in City1\",\"Content\":\"name roomType pricing max_occupancy rating house_rules\\nStay entire_home $100 2 4.9 No smoking\"},'
            '{\"Description\":\"Attractions in City1\",\"Content\":\"name subcategories visit_duration address latitude longitude website\\nPark [\\\"Nature\\\"] 2.0 Addr 1.0 2.0 http://x\"},'
            '{\"Description\":\"Nearest Public Transit Stop from Point of Interest in City1\",\"Content\":\"PoI nearest_stop_name nearest_stop_latitude nearest_stop_longitude nearest_stop_distance\\nPark Stop 1.0 2.0 123.0\"}'
            ']'
        )
        ref2 = (
            '['
            '{\"Description\":\"Flight from Org to City1 on 2024-01-01\",\"Content\":\"Flight Number Price DepTime ArrTime ActualElapsedTime Distance\\n\"},'
            '{\"Description\":\"Self-driving from Org to City1\",\"Content\":\"Self-driving, Duration: 10 mins, Distance: 1 km, Estimated Cost: $1\"}'
            ']'
        )
        kb = build_unified_kb("Org", [ref1, ref2])
        self.assertEqual(len(kb.stages), 1)
        stage = kb.stages[0]
        self.assertIn("name", stage.restaurants.columns)
        self.assertIn("name", stage.accommodations.columns)
        self.assertIn("name", stage.attractions.columns)
        self.assertIn("PoI", stage.poi2transit.columns)
        self.assertTrue(any(t.mode in {"flight", "self-driving", "taxi"} for t in kb.transports))


if __name__ == "__main__":
    unittest.main()


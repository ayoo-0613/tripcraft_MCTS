import unittest

import pandas as pd

from TripCraft.mcts_baseline.env import POIBlock, TripCraftEnv, _lookup_transit
from TripCraft.mcts_baseline.formatter import build_poi_list_str
from TripCraft.mcts_baseline.io import TripCraftRow
from TripCraft.mcts_baseline.kb import StageKB, TransportOption, UnifiedKB


class TestFixes(unittest.TestCase):
    def test_poi_blocks_sorted_by_start(self):
        blocks = [
            POIBlock(name="B", kind="visit", start="11:30", end="12:00", nearest_transit="S", dist_m=1.0),
            POIBlock(name="A", kind="stay", start="09:00", end="09:30", nearest_transit="S", dist_m=1.0),
        ]
        s = build_poi_list_str(blocks)
        self.assertLess(s.find("A, stay from 09:00"), s.find("B, visit from 11:30"))

    def test_lookup_transit_normalize_and_strip_city(self):
        poi2 = pd.DataFrame(
            [
                {
                    "PoI": "Restaurant One",
                    "nearest_stop_name": "STOP",
                    "nearest_stop_latitude": 0.0,
                    "nearest_stop_longitude": 0.0,
                    "nearest_stop_distance": 12.0,
                }
            ]
        )
        stage = StageKB(
            city="City",
            attractions=pd.DataFrame(columns=["name"]),
            restaurants=pd.DataFrame(columns=["name"]),
            accommodations=pd.DataFrame(columns=["name"]),
            poi2transit=poi2,
            events=None,
        )
        stop, dist = _lookup_transit(stage, "Restaurant One City")
        self.assertEqual(stop, "STOP")
        self.assertAlmostEqual(dist, 12.0)

    def test_same_day_restaurant_dedup(self):
        stage = StageKB(
            city="City",
            attractions=pd.DataFrame([{"name": "A1", "visit_duration": 2.0}]),
            restaurants=pd.DataFrame(
                [
                    {"name": "R1", "cuisines": ["Italian"], "avg_cost": 10.0, "rating": 4.0},
                    {"name": "R2", "cuisines": ["Italian"], "avg_cost": 12.0, "rating": 4.2},
                ]
            ),
            accommodations=pd.DataFrame([{"name": "H1", "pricing_value": 0.0, "rating": 4.0}]),
            poi2transit=pd.DataFrame(
                [
                    {
                        "PoI": "R1",
                        "nearest_stop_name": "S1",
                        "nearest_stop_latitude": 0.0,
                        "nearest_stop_longitude": 0.0,
                        "nearest_stop_distance": 1.0,
                    }
                ]
            ),
            events=None,
        )
        kb = UnifiedKB(
            stages=[stage],
            transports=[TransportOption(mode="taxi", frm="Org", to="City", date="2024-01-01", raw="-")],
        )
        row = TripCraftRow(
            idx=1,
            org="Org",
            dest="Dest",
            days=2,
            visiting_city_number=1,
            date=["2024-01-01", "2024-01-02"],
            people_number=1,
            local_constraint={"cuisine": ["Italian"]},
            budget=1000.0,
            query=None,
            level="easy",
            persona="P",
            ref_blocks=[],
        )
        env = TripCraftEnv(row=row, kb=kb, topk=5)
        st = env.initial_state()

        # Force to breakfast slot.
        st.substep = 2  # transport(0), accommodation(1), breakfast(2)
        st.drafts[0].breakfast = "R1"
        st.substep = 4  # lunch slot
        actions = env.legal_actions(st, topk=5)
        chosen_names = {a.get("name") for a in actions if a.get("type") == "set_lunch"}
        self.assertNotIn("R1", chosen_names)

    def test_travel_time_conflict_skips_breakfast(self):
        stage = StageKB(
            city="City",
            attractions=pd.DataFrame([{"name": "A1", "visit_duration": 2.0}]),
            restaurants=pd.DataFrame([{"name": "R1", "cuisines": ["Italian"], "avg_cost": 10.0, "rating": 4.0}]),
            accommodations=pd.DataFrame([{"name": "H1", "pricing_value": 0.0, "rating": 4.0}]),
            poi2transit=pd.DataFrame(columns=["PoI", "nearest_stop_name", "nearest_stop_latitude", "nearest_stop_longitude", "nearest_stop_distance"]),
            events=None,
        )
        kb = UnifiedKB(
            stages=[stage],
            transports=[
                TransportOption(
                    mode="flight",
                    frm="Org",
                    to="City",
                    date="2024-01-01",
                    raw="Flight Number: F1, from Org to City, Departure Time: 09:30, Arrival Time: 10:30",
                )
            ],
        )
        row = TripCraftRow(
            idx=1,
            org="Org",
            dest="Dest",
            days=2,
            visiting_city_number=1,
            date=["2024-01-01", "2024-01-02"],
            people_number=1,
            local_constraint={"cuisine": ["Italian"]},
            budget=1000.0,
            query=None,
            level="easy",
            persona="P",
            ref_blocks=[],
        )
        env = TripCraftEnv(row=row, kb=kb, topk=5)
        st = env.initial_state()
        # Apply transport first to set window.
        st = env.apply_action(st, {"type": "set_transport", "raw": kb.transports[0].raw, "candidates": [kb.transports[0].raw]})
        # Jump to breakfast slot.
        st.substep = 2
        actions = env.legal_actions(st, topk=5)
        self.assertTrue(all(a["type"] == "skip_breakfast" for a in actions))

    def test_travel_arrival_after_slot_allows_breakfast(self):
        stage = StageKB(
            city="City",
            attractions=pd.DataFrame([{"name": "A1", "visit_duration": 2.0}]),
            restaurants=pd.DataFrame([{"name": "R1", "cuisines": ["Italian"], "avg_cost": 10.0, "rating": 4.0}]),
            accommodations=pd.DataFrame([{"name": "H1", "pricing_value": 0.0, "rating": 4.0}]),
            poi2transit=pd.DataFrame(columns=["PoI", "nearest_stop_name", "nearest_stop_latitude", "nearest_stop_longitude", "nearest_stop_distance"]),
            events=None,
        )
        kb = UnifiedKB(
            stages=[stage],
            transports=[
                TransportOption(
                    mode="flight",
                    frm="Org",
                    to="City",
                    date="2024-01-01",
                    raw="Flight Number: F1, from Org to City, Departure Time: 19:35, Arrival Time: 21:22",
                )
            ],
        )
        row = TripCraftRow(
            idx=1,
            org="Org",
            dest="Dest",
            days=2,
            visiting_city_number=1,
            date=["2024-01-01", "2024-01-02"],
            people_number=1,
            local_constraint={"cuisine": ["Italian"]},
            budget=1000.0,
            query=None,
            level="easy",
            persona="P",
            ref_blocks=[],
        )
        env = TripCraftEnv(row=row, kb=kb, topk=5)
        st = env.initial_state()
        st = env.apply_action(st, {"type": "set_transport", "raw": kb.transports[0].raw, "candidates": [kb.transports[0].raw]})
        st.substep = 2
        actions = env.legal_actions(st, topk=5)
        self.assertTrue(any(a["type"] == "set_breakfast" for a in actions))


if __name__ == "__main__":
    unittest.main()

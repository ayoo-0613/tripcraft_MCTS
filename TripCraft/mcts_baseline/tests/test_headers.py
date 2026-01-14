import tempfile
import unittest

from TripCraft.mcts_baseline.io import load_rows


class TestHeaders(unittest.TestCase):
    def _write_tmp(self, text: str) -> str:
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8", newline="")
        f.write(text)
        f.close()
        return f.name

    def test_3day_header_reference_information(self):
        path = self._write_tmp(
            "org,dest,days,visiting_city_number,date,people_number,local_constraint,budget,query,level,persona,reference_information\n"
            "A,B,3,1,\"['2024-01-01','2024-01-02','2024-01-03']\",1,\"{'cuisine': None}\",1000,,easy,P,\"[]\"\n"
        )
        rows = load_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].ref_blocks), 1)
        self.assertEqual(len(rows[0].date), rows[0].days)

    def test_5day_header_reference_information_1_2(self):
        path = self._write_tmp(
            "org,dest,days,visiting_city_number,date,people_number,local_constraint,budget,query,level,persona,reference_information_1,reference_information_2\n"
            "A,B,5,2,\"['2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05']\",1,\"{'cuisine': ['Italian']}\",1000,,easy,P,\"[]\",\"[]\"\n"
        )
        rows = load_rows(path)
        self.assertEqual(len(rows[0].ref_blocks), 2)
        self.assertEqual(len(rows[0].date), rows[0].days)

    def test_7day_header_reference_information_1_2_3(self):
        path = self._write_tmp(
            "org,dest,days,visiting_city_number,date,people_number,local_constraint,budget,query,level,persona,reference_information_1,reference_information_2,reference_information_3\n"
            "A,B,7,3,\"['2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-06','2024-01-07']\",1,\"{'cuisine': None}\",1000,,easy,P,\"[]\",\"[]\",\"[]\"\n"
        )
        rows = load_rows(path)
        self.assertEqual(len(rows[0].ref_blocks), 3)
        self.assertEqual(len(rows[0].date), rows[0].days)


if __name__ == "__main__":
    unittest.main()


import unittest
from bench import decode_overlap

class DecodeOverlap(unittest.TestCase):
    def test_unequal_stream_rates(self):
        self.assertAlmostEqual(decode_overlap([
            {"output_tokens": 100, "gen_tps": 10},
            {"output_tokens": 100, "gen_tps": 20}], 10), 1.485)
        self.assertIsNone(decode_overlap([], 0))

if __name__ == '__main__':
    unittest.main()

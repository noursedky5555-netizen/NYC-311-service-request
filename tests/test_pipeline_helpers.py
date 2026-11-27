import unittest

from taxi_analytics.pipeline import parse_months


class MonthParsingTests(unittest.TestCase):
    def test_parse_months_validates_and_deduplicates(self) -> None:
        self.assertEqual(
            parse_months(["2024-01", "2024-02", "2024-01"]),
            ["2024-01", "2024-02"],
        )

    def test_parse_months_rejects_invalid_values(self) -> None:
        for months in ([], ["2024-13"], ["24-01"], ["2024-1"]):
            with self.subTest(months=months):
                with self.assertRaisesRegex(ValueError, "YYYY-MM"):
                    parse_months(months)


if __name__ == "__main__":
    unittest.main()

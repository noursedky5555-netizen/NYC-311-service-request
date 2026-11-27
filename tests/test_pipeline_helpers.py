import unittest

from nyc311.pipeline import month_range, parse_month


class MonthParsingTests(unittest.TestCase):
    def test_parse_month(self) -> None:
        self.assertEqual(parse_month("2024-01"), (2024, 1))

    def test_month_range_crosses_year(self) -> None:
        self.assertEqual(
            month_range("2023-11", "2024-02"),
            ["2023-11", "2023-12", "2024-01", "2024-02"],
        )

    def test_month_range_includes_endpoints(self) -> None:
        self.assertEqual(month_range("2024-01", "2024-03"), [
            "2024-01", "2024-02", "2024-03"
        ])

    def test_invalid_month_rejected(self) -> None:
        for value in ("2024-13", "24-01", "2024-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "YYYY-MM"):
                    parse_month(value)

    def test_reversed_range_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "earlier"):
            month_range("2024-04", "2024-03")


if __name__ == "__main__":
    unittest.main()

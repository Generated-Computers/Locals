import unittest

from inventory import Item, Store
from inventory.report import cheapest_with_tag, tag_report


def sample():
    s = Store()
    s.add(Item("apple", 0.5, 10, ["fruit"]))
    s.add(Item("pear", 0.75, 4, ["fruit"]))
    s.add(Item("soap", 2.0, 3, ["home"]))
    return s


class InventoryTest(unittest.TestCase):
    def test_find_by_tag(self):
        self.assertEqual([i.name for i in sample().find_by_tag("fruit")], ["apple", "pear"])

    def test_total_value(self):
        self.assertAlmostEqual(sample().total_value(), 14.0)

    def test_remove(self):
        s = sample()
        s.remove("soap", 3)
        self.assertNotIn("soap", s.items)
        with self.assertRaises(ValueError):
            s.remove("pear", 5)

    def test_report(self):
        self.assertEqual(tag_report(sample(), ["fruit"]), "fruit: 2 items, 8.00")
        self.assertEqual(cheapest_with_tag(sample(), "fruit").name, "apple")


if __name__ == "__main__":
    unittest.main()

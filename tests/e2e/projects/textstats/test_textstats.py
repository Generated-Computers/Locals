import unittest

from textstats import average_word_length, longest_line, top_words, word_count, words

TEXT = "The cat sat on the mat.\nThe dog sat too!\nA bird"


class TextStatsTest(unittest.TestCase):
    def test_words(self):
        self.assertEqual(words("Hello, World! it's"), ["hello", "world", "it's"])

    def test_word_count(self):
        self.assertEqual(word_count(TEXT), 12)

    def test_top_words(self):
        self.assertEqual(top_words(TEXT, 2), [("the", 3), ("sat", 2)])

    def test_top_words_ties_alphabetical(self):
        self.assertEqual(top_words("b a c a b", 2), [("a", 2), ("b", 2)])

    def test_average_word_length(self):
        self.assertAlmostEqual(average_word_length("ab abcd"), 3.0)
        self.assertEqual(average_word_length(""), 0.0)

    def test_longest_line(self):
        self.assertEqual(longest_line(TEXT), "The cat sat on the mat.")


if __name__ == "__main__":
    unittest.main()

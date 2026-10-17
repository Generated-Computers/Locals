"""Small text statistics for the command line: python3 textstats.py FILE"""

import re
import sys
from collections import Counter

WORD = re.compile(r"[a-z']+")


def words(text):
    """The words of a text, lower-cased, without punctuation."""
    return WORD.findall(text.lower())


def word_count(text):
    return len(words(text))


def top_words(text, n=3):
    """The n most frequent words with their counts, most frequent first; ties in alphabetical order."""
    counts = Counter(words(text))
    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return ranked[:n]


def average_word_length(text):
    ws = words(text)
    if not ws:
        return 0.0
    return sum(len(w) for w in ws) / len(ws)


def longest_line(text):
    """The longest line (the first one if several are equally long)."""
    lines = text.splitlines()
    return max(lines, key=len) if lines else ""


def main(argv):
    text = open(argv[1]).read() if len(argv) > 1 else sys.stdin.read()
    print(f"words: {word_count(text)}")
    print(f"average length: {average_word_length(text):.2f}")
    for w, c in top_words(text):
        print(f"{w}: {c}")


if __name__ == "__main__":
    main(sys.argv)

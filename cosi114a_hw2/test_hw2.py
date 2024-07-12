#! /usr/bin/env python

# test_hw2.py
# Version 1.1
# 9/26/2022

import os
import random
import unittest
from math import log
from typing import Iterable, Iterator, Sequence, TypeVar

from grader import Grader, points
from hw2 import (
    bigram_probs,
    trigram_probs,
    sample_bigrams,
    sample_trigrams,
    load_tokenized_file,
    bigram_sequence_prob,
    trigram_sequence_prob,
)

T = TypeVar("T")
SAMPLE_LYRICS = os.path.join("test_data", "costello_radio.txt")


class DefensiveIterable(Iterable[T]):
    def __init__(self, source: Iterable[T]):
        self.source: Iterable[T] = source

    def __iter__(self) -> Iterator[T]:
        return iter(self.source)

    def __len__(self):
        # This object should never be put into a sequence, so we sabotage the
        # __len__ function to make it difficult to do so. We specifically raise
        # ValueError because TypeError and NotImplementedError appear to be
        # handled by the list function.
        raise ValueError(
            "You cannot put this iterable into a sequence (list, tuple, etc.). "
            "Instead, iterate over it using a for loop."
        )


def load_tokenized_test_file(path: str) -> Iterable[Sequence[str]]:
    """Return a defensive iterable over sentences as sequences of tokens."""
    return DefensiveIterable(load_tokenized_file(path))


class SeedControlledTestCase(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(12345)


class TestBigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_test_file(SAMPLE_LYRICS)

    @points(3)
    def test_bigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the correct number of keys."""
        probs = bigram_probs(self.sentences_gen)
        self.assertEqual(39, len(probs))

    @points(3)
    def test_bigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        probs = bigram_probs(self.sentences_gen)
        radio = probs["Radio"]
        self.assertAlmostEqual(0.8, radio[","])
        self.assertAlmostEqual(0.2, radio["is"])

    @points(3)
    def test_bigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        probs = bigram_probs(self.sentences_gen)

        # Check outer dict type
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(str, type(outer_key))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


class TestTrigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_test_file(SAMPLE_LYRICS)

    @points(3)
    def test_trigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the correct number of keys."""
        probs = trigram_probs(self.sentences_gen)
        self.assertEqual(54, len(probs))

    @points(3)
    def test_trigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        probs = trigram_probs(self.sentences_gen)
        radio = probs[("Radio", "is")]
        self.assertDictEqual({"a": 0.5, "cleaning": 0.5}, radio)

    @points(3)
    def test_trigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        probs = trigram_probs(self.sentences_gen)

        # Check outer dict
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(tuple, type(outer_key))
        self.assertEqual(2, len(outer_key))
        self.assertEqual(str, type(outer_key[0]))
        self.assertEqual(str, type(outer_key[1]))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        # PyCharm falsely claims inner_dict is a str, so there is a type warning below
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


class TestSampleBigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.probs = bigram_probs(load_tokenized_test_file(SAMPLE_LYRICS))

    @points(5)
    def test_first_sample(self) -> None:
        """The first bigram-generated sentence is correct."""
        sent = sample_bigrams(self.probs)
        self.assertEqual(["Radio", ",", "radio"], sent)

    @points(5)
    def test_third_sample(self) -> None:
        """The third bigram-generated sentence is correct."""
        # Throw away first two samples
        sample_bigrams(self.probs)
        sample_bigrams(self.probs)
        sent = sample_bigrams(self.probs)
        self.assertEqual(["But", "they", "think", "that", "it's", "treason"], sent)


class TestSampleTrigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.probs = trigram_probs(load_tokenized_test_file(SAMPLE_LYRICS))

    @points(5)
    def test_first_sample(self) -> None:
        """The first trigram-generated sentence is correct."""
        sent = sample_trigrams(self.probs)
        self.assertEqual(["Radio", ",", "radio"], sent)

    @points(5)
    def test_third_sample(self) -> None:
        """The third trigram-generated sentence is correct."""
        # Throw away first two samples
        sample_trigrams(self.probs)
        sample_trigrams(self.probs)
        sent = sample_trigrams(self.probs)
        self.assertEqual(["But", "they", "don't", "give", "you", "any", "choice"], sent)


class TestBigramSequenceProb(unittest.TestCase):
    @points(3)
    def test_bigram_seq1(self) -> None:
        """Bigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b"]]
        probs = bigram_probs(sample_seqs)
        self.assertEqual(log(1.0), bigram_sequence_prob(sample_seqs[0], probs))

    @points(3)
    def test_bigram_seq2(self) -> None:
        """Bigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"], ["a", "d", "c"]]
        probs = bigram_probs(sample_seqs)
        self.assertEqual(log(0.5), bigram_sequence_prob(sample_seqs[0], probs))

    @points(0)
    def test_bigram_my_test_key_not_in_dict(self) -> None:
        """Bigram sequence probability is negative infinity when the key is not in the dictionary."""
        sample_seqs = [["a", "b", "c"]]
        probs = bigram_probs(sample_seqs)
        test_seq = ["z", "z", "z"]
        self.assertEqual(float("-inf"), bigram_sequence_prob(test_seq, probs))

    @points(0)
    def test_bigram_my_test_prob_is_zero(self) -> None:
        """Bigram sequence probability is negative infinity when the probability is zero."""
        sample_seqs = [["a", "b"]]
        probs = bigram_probs(sample_seqs)
        test_seq = ["a", "b", "c"]
        self.assertEqual(float("-inf"), bigram_sequence_prob(test_seq, probs))


class TestTrigramSequenceProb(unittest.TestCase):
    @points(3)
    def test_trigram_seq1(self) -> None:
        """Trigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"]]
        probs = trigram_probs(sample_seqs)
        self.assertEqual(log(1.0), trigram_sequence_prob(sample_seqs[0], probs))

    @points(3)
    def test_trigram_seq2(self) -> None:
        """Trigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"], ["a", "d", "c"]]
        probs = trigram_probs(sample_seqs)
        self.assertEqual(log(0.5), trigram_sequence_prob(sample_seqs[0], probs))

    @points(0)
    def test_my_test_key_not_in_dict(self) -> None:
        """Trigram sequence probability is negative infinity when the key is not in the dictionary."""
        sample_seqs = [["a", "b", "c"]]
        probs = trigram_probs(sample_seqs)
        test_seq = ["z", "z", "z"]
        self.assertEqual(float("-inf"), trigram_sequence_prob(test_seq, probs))

    @points(0)
    def test_my_test_prob_is_zero(self) -> None:
        """Trigram sequence probability is negative infinity when the key is not in the dictionary."""
        sample_seqs = [["a", "a", "a"]]
        probs = trigram_probs(sample_seqs)
        test_seq = ["a", "b", "c"]
        self.assertEqual(float("-inf"), trigram_sequence_prob(test_seq, probs))


def main() -> None:
    tests = [
        TestBigramProbs,
        TestTrigramProbs,
        TestSampleBigrams,
        TestSampleTrigrams,
        TestBigramSequenceProb,
        TestTrigramSequenceProb,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()

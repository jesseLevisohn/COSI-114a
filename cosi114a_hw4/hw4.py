from abc import abstractmethod, ABC
from collections import Counter
from collections import defaultdict
from math import log
from operator import itemgetter
from typing import Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class TaggedToken:
    """Store the text and tag for a token."""

    # DO NOT MODIFY
    def __init__(self, text: str, tag: str):
        self.text: str = text
        self.tag: str = tag

    # DO NOT MODIFY
    def __str__(self) -> str:
        return f"{self.text}/{self.tag}"

    # DO NOT MODIFY
    def __repr__(self) -> str:
        return f"<TaggedToken {str(self)}>"

    # DO NOT MODIFY
    @classmethod
    def from_string(cls, s: str) -> "TaggedToken":
        """Create a TaggedToken from a string with the format "token/tag".

        While the tests use this, you do not need to.
        """
        splits = s.rsplit("/", 1)
        assert len(splits) == 2, f"Could not parse token: {repr(s)}"
        return cls(splits[0], splits[1])


# DO NOT MODIFY
class Tagger(ABC):
    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """Tag a sentence with part of speech tags."""
        raise NotImplementedError

    # DO NOT MODIFY
    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """Yield a list of tags for each sentence in the input."""
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    # DO NOT MODIFY
    def test(
        self, tagged_sentences: Iterable[Sequence[TaggedToken]]
    ) -> tuple[list[str], list[str]]:
        """Return a tuple containing a list of predicted tags and a list of actual tags.

        Does not preserve sentence boundaries to make evaluation simpler.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sentence in tagged_sentences:
            predicted.extend(self.tag_sentence([tok.text for tok in sentence]))
            actual.extend([tok.tag for tok in sentence])
        return predicted, actual


# DO NOT MODIFY
def safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter.

    In case of ties, the lexicographically first item is returned.
    """
    assert counts, "Counter is empty"
    return items_descending_value(counts)[0]


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self) -> None:
        # Add an attribute to store the most frequent tag
        pass

    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        pass

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        pass


class UnigramTagger(Tagger):
    def __init__(self) -> None:
        # Add data structures that you need here
        pass

    def train(self, sentences: Iterable[Sequence[TaggedToken]]):
        pass

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        pass


class SentenceCounter:
    def __init__(self, k: float) -> None:
        self.k = k
        # Add data structures that you need here

    def count_sentences(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        for sentence in sentences:
            # Fill in this loop
            pass

    def unique_tags(self) -> list[str]:
        pass

    def emission_prob(self, tag: str, word: str) -> float:
        pass

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        pass

    def initial_prob(self, tag: str) -> float:
        pass


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods.

    def __init__(self, k) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[TaggedToken]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        """Return the probability for a sequence of tags given tokens."""
        pass


class GreedyBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        pass


class ViterbiBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        pass

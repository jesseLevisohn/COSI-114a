from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence, Any, List

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


def counts_to_probs(counts: Counter[T]) -> defaultdict[T, float]:
    d = defaultdict(float)
    total = sum(counts.values())
    # Adds up how many times each key appears in the counter to get the total number of keys in the source
    # Adds each key to the default dict and sets its value to be its probability, the count divided by the total
    for key, value in counts.items():
        d[key] = value / total
    return d


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    """Return the bigrams contained in a sequence."""
    # turns the sentence into a list
    l = list(sentence)
    l = [START_TOKEN] + l + [END_TOKEN]
    output = [tuple(l[x:x+2]) for x in range(len(l) - 1)]
    return output

def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return the trigrams contained in a sequence."""
    # turns the sentence into a list
    l = list(sentence)
    l.append(END_TOKEN)
    l.append(END_TOKEN)
    output = []
    # initializes the variables first and second as the start token
    first = START_TOKEN
    second = START_TOKEN
    # runs through the list creating the trigrams
    for s in l:
        output.append((first, second, s))
        first = second
        second = s
    return output


def count_unigrams(sentences: Iterable[list[str]], lower: bool = False) -> Counter[str]:
    """Count the unigrams in an iterable of sentences, optionally lowercasing."""
    # creates two empty lists
    u_grams = []
    # for each line in the text it adds all the tokens in the line to the list u_grams
    for line in sentences:
        # if the lower is true the tokens will all be made lowercase
        if lower:
            line = to_lower(line)
        u_grams.extend(line)
    # returns a counter over the list which counts how many instances there are of each element in the list
    return Counter(u_grams)


def count_bigrams(
        sentences: Iterable[list[str]], lower: bool = False) -> Counter[tuple[str, str]]:
    """Count the bigrams in an iterable of sentences, optionally lowercasing."""
    b_grams = []
    for line in sentences:
        # if the lower is true the tokens will all be made lowercase
        if lower:
            line = to_lower(line)
        # breaks the line into all its bigrams and adds it to the end of b_grams
        b_grams.extend(bigrams(line))
    # returns a counter over the list which counts how many instances there are of each element in the list
    return Counter(b_grams)


def count_trigrams(
        sentences: Iterable[list[str]], lower: bool = False) -> Counter[tuple[str, str, str]]:
    """Count the trigrams in an iterable of sentences, optionally lowercasing."""
    # creates an empty list
    t_grams = []
    for line in sentences:
        # if the lower is true the tokens will all be made lowercase
        if lower:
            line = to_lower(line)
        # breaks the line into all its trigrams and adds it to the end of b_grams
        t_grams.extend(trigrams(line))
    # returns a counter over the list which counts how many instances there are of each element in the list
    return Counter(t_grams)


def to_lower(s: list[str]) -> list[str]:
    # creates a new list which has every element of s but lowercase
    lower_list = [token.lower() for token in s]
    return lower_list

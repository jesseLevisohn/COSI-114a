import math
import random
from collections import defaultdict, Counter
from math import log
from typing import Sequence, Iterable, Generator, TypeVar

# hw2.py
# Version 1.1
# 9/26/2022

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"
# DO NOT MODIFY
NEG_INF = float("-inf")
# DO NOT MODIFY (needed if you copy code from HW 1)
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: str) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order, sort items
    # This is very slow and should be avoided in general, but we do
    # it in order to get predictable results
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.
def counts_to_probs(counts: Counter[T]) -> defaultdict[T, float]:
    d = defaultdict(float)
    # Adds up how many times each key appears in the counter to get the total number of keys in the source
    total = sum(counts.values())
    # Adds each key to the default dict and sets its value to be its probability, the count divided by the total
    for key, value in counts.items():
        d[key] = value / total
    return d


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    """Return the bigrams contained in a sequence."""
    # turns the sentence into a list
    l = list(sentence)
    # adds the start and end tokens to the list
    l = [START_TOKEN] + l + [END_TOKEN]
    # uses list comprehension to add a tuple consisting of each adjacent pair of words to a list
    output = [tuple(l[x:x + 2]) for x in range(len(l) - 1)]
    return output


def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return the bigrams contained in a sequence."""
    # turns the sentence into a list
    l = list(sentence)
    # adds the start and end tokens to the list
    l = [START_TOKEN] + [START_TOKEN] + l + [END_TOKEN] + [END_TOKEN]
    # uses list comprehension to add a tuple consisting of each adjacent trio of words to a list
    output = [tuple(l[x:x + 3]) for x in range(len(l) - 2)]
    return output


def bigram_probs(
        sentences: Iterable[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Return bigram probabilities computed from the provided sequences."""
    d = defaultdict(Counter)
    # iterates over each bigram in each of the sentences
    for sentence in sentences:
        for bi_gram in bigrams(sentence):
            # for each bigram add the first token of the bigram to the dict as a key
            # updates the counter in the value to include the second token in the bigram
            d[bi_gram[0]].update([bi_gram[1]])
    # converts the counters in the values of the dict to probabilities
    for key in d:
        d[key] = dict(counts_to_probs(d[key]))
    return dict(d)


def trigram_probs(
        sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Return trigram probabilities computed from the provided sequences."""
    d = defaultdict(Counter)
    # iterates over each trigram in each of the sentences
    for sentence in sentences:
        for tri_gram in trigrams(sentence):
            # for each trigram add the first two tokens of the trigram to the dict as a key
            # updates the counter in the value to include the second token in the trigram
            d[tri_gram[0:2]].update([tri_gram[2]])
    # converts the counters in the values of the dict to probabilities
    for key in d:
        d[key] = dict(counts_to_probs(d[key]))
    return dict(d)


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided bigram probabilities."""
    # Initialize the starting context and the output list
    context = START_TOKEN
    output = []
    # Continue generating tokens until the end token is reached
    while context != END_TOKEN:
        # Use sample to get a token from the distribution context and set it as the new context
        context = sample(probs[context])
        # If the new context is not the end token add it to the list
        if context != END_TOKEN:
            output.append(context)
    return output


def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided trigram probabilities."""
    # Initialize the starting context, the output list, and the current word
    context = (START_TOKEN, START_TOKEN)
    output = []
    curr = ""
    # Continue generating tokens until the end token is reached
    while curr != END_TOKEN:
        # Use sample to get a token from the distribution context
        curr = sample(probs[context])
        # Update the context
        context = (context[1], curr)
        # If the new token is not the end token add it to the list
        if curr != END_TOKEN:
            output.append(curr)
    return output


def bigram_sequence_prob(
        sequence: Sequence[str], probs: dict[str, dict[str, float]]
) -> float:
    """Compute the probability of a sequence using bigram probabilities."""
    # Initialize a context and a probability
    prev = START_TOKEN
    prob = 0
    sequence = sequence + [END_TOKEN]
    # Iterate through each token in the sentence
    for token in sequence:
        # Find the log of the probability of that token given the previous two tokens
        if token not in probs[prev]:
            # If the context is not in the distribution negative infinity is returned
            prob = NEG_INF
            break
        if probs[prev][token] == 0:
            # If the context is 0 negative infinity is returned
            prob = NEG_INF
            break
        else:
            p = probs[prev][token]
        # Add the log of the probability of that token given the previous token
        prob += math.log(p)
        # update the context
        prev = token
    return prob


def trigram_sequence_prob(
        sequence: Sequence[str], probs: dict[tuple[str, str], dict[str, float]]
) -> float:
    """Compute the probability of a sequence using trigram probabilities."""
    # Initialize a context and a probability
    prev = (START_TOKEN, START_TOKEN)
    prob = 0
    sequence = sequence + [END_TOKEN]
    # Iterate through each token in the sentence
    for token in sequence:
        # Find the log of the probability of that token given the previous two tokens
        if token not in probs[prev]:
            # If the context is not in the distribution negative infinity is returned
            prob = NEG_INF
            break
        if probs[prev][token] == 0:
            # If the context is 0 negative infinity is returned
            prob = NEG_INF
            break
        else:
            p = probs[prev][token]
        # Add the log of the probability of that token given the previous two tokens
        prob += math.log(p)
        # update the context
        prev = (prev[1], token)
    return prob

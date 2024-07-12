# Version 1.1
# 11/20/2022

import random
from collections import defaultdict, Counter
from operator import itemgetter
from typing import Iterable, Generator, Sequence, List

# DO NOT MODIFY
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


# DO NOT MODIFY
class ClassificationInstance:
    """Represent a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


# DO NOT MODIFY
class LanguageIdentificationInstance:
    """Represent a single instance from a language ID dataset."""

    def __init__(
            self,
            language: str,
            text: str,
    ) -> None:
        self.language: str = language
        self.text: str = text

    def __repr__(self) -> str:
        return f"<LanguageIdentificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.language}; text={self.text}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_line(cls, line: str) -> "LanguageIdentificationInstance":
        splits = line.strip().split("\t")
        assert len(splits) == 2
        return cls(splits[0], splits[1])


# DO NOT MODIFY
def load_lid_instances(
        path: str,
) -> Generator[LanguageIdentificationInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(path, encoding="utf8") as file:
        for line in file:
            yield LanguageIdentificationInstance.from_line(line)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


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
def char_bigrams(sentence: Iterable[str]) -> list[Sequence[str]]:
    """Return the character bigrams contained in a sequence."""
    # Return a set of the character bigrams in the sequence
    return {x+y for x, y in zip(sentence, sentence[1:])}


class CharBigramFeatureExtractor:
    @staticmethod
    def extract_features(
            instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        """Extract character bigram features from an instance."""
        return ClassificationInstance(instance.language, char_bigrams(instance.text))


class InstanceCounter:
    def __init__(self) -> None:
        self.labels_counter: Counter[str] = Counter()
        self.labels_list: list[str] = []

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        """Count the labels in the provided instances."""
        for instance in instances:
            self.labels_counter[instance.label] += 1
        self.labels_list = items_descending_value(self.labels_counter)

    def labels(self) -> list[str]:
        """Return a sorted list of the labels."""
        return self.labels_list


class Perceptron:
    def __init__(self, labels: list[str]) -> None:
        self.labels: list[str] = labels
        # For weights, sums, and last_updated, the inner keys are the labels and the inner keys are the features
        self.weights: dict[str, defaultdict[str, float]] = {label: defaultdict(float) for label in self.labels}
        self.sums: dict[str, defaultdict[str, float]] = {label: defaultdict(float) for label in self.labels}
        self.last_updated: dict[str, defaultdict[str, int]] = {label: defaultdict(float) for label in self.labels}

    def classify(self, features: Iterable[str]) -> str:
        # Create a dictionary from the labels to their scores
        label_weights = defaultdict(float)
        for label in self.labels:
            feature_weight_sum = 0
            for feature in features:
                # Sum the feature weights for each label
                feature_weight_sum += self.weights[label][feature]
            # Update the value for each label to be the sum of the feature weights with that label
            label_weights[label] = feature_weight_sum
        # Return the class with the highest score
        return max_item(label_weights)[0]

    def learn(
            self,
            instance: ClassificationInstance,
            step: int,
            lr: float,
    ) -> None:
        expected_label = instance.label
        predicted_label = self.classify(instance.features)
        if predicted_label == expected_label:
            # Our prediction was correct, no need to update
            return
        for feature in instance.features:
            # Update the sums of the feature for the incorrectly predicted label
            self.sums[predicted_label][feature] += self.weights[predicted_label][feature] * \
                                                   (step - self.last_updated[predicted_label][feature])
            # Penalize the weights for the incorrectly predicted label
            self.weights[predicted_label][feature] -= lr
            # Update the last time that feature was updated for the label
            self.last_updated[predicted_label][feature] = step
            # Update the sums of the feature for the actual correct label
            self.sums[expected_label][feature] += self.weights[expected_label][feature] * \
                                                   (step - self.last_updated[expected_label][feature])
            # Increase the weights for the actual correct label
            self.weights[expected_label][feature] += lr
            # Update the last time that feature was updated for the label
            self.last_updated[expected_label][feature] = step

    def predict(self, test: Sequence[ClassificationInstance]) -> list[str]:
        # Return a list of the predicted classes for each instance
        return [self.classify(instance.features) for instance in test]

    def average(self, final_step: int) -> None:
        for label in self.labels:
            for ftr in self.weights[label]:
                # Update the sums for the last step
                self.sums[label][ftr] += self.weights[label][ftr] * (final_step - self.last_updated[label][ftr])
                # Average and update the weights
                self.weights[label][ftr] = self.sums[label][ftr] / final_step


def train_perceptron(
        model: Perceptron,
        data: list[ClassificationInstance],
        epochs: int,
        lr: float,
        *,
        average: bool,
) -> None:
    # DO NOT MODIFY THE ASSERT STATEMENTS
    # Some argument checks to avoid any accidents
    assert isinstance(model, Perceptron)
    assert isinstance(data, list)
    assert data
    assert isinstance(data[0], ClassificationInstance)
    assert isinstance(epochs, int)
    assert epochs > 0
    assert isinstance(lr, float)
    assert lr > 0
    assert isinstance(average, bool)

    # Add your code here
    step = 1
    for epoch in range(epochs):
        for instance in data:
            model.learn(instance, step, lr)
            step += 1
        random.shuffle(data)
    if average:
        model.average(step)

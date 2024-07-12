import json
import math
from collections import defaultdict, Counter
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
            self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["airline"], json_dict["sentences"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
            self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_sentiment_instances(
        datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
        datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    """Load sentence segmentation instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable, and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.
def stats_errors(predictions: Sequence[str], expected: Sequence[str]):
    # Raise an error if the lengths of expected and predictions are not equal
    if len(predictions) != len(expected):
        raise ValueError("Predictions and Expected are not the same length.")
    # Raise an error if expected is empty
    if len(expected) == 0:
        raise ValueError("Expected is empty.")
    # Raise an error if predictions is empty
    if len(predictions) == 0:
        raise ValueError("Predictions is empty")


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    """Compute the accuracy of the provided predictions."""
    # Raises an error if the lengths of predictions and expected are not equal or either is empty
    stats_errors(predictions, expected)
    # Iterate over predictions adn expected creating a new list of true outcomes
    true_outcomes = [predictions[x] for x in range(len(predictions)) if predictions[x] == expected[x]]
    # compute accuracy as the number of true outcomes minus the total predictions made
    a = len(true_outcomes) / len(predictions)
    return a


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the recall of the provided predictions."""
    # Raises an error if the lengths of predictions and expected are not equal or either is empty
    stats_errors(predictions, expected)
    true_positives = 0
    gold_positives = 0
    for x in range(len(expected)):
        # If the prediction matches the label, the number of gold positives is incremented by 1
        if expected[x] == label:
            gold_positives += 1
            # If the prediction also matches the expected, the number of true positives is incremented by 1
            if predictions[x] == expected[x]:
                true_positives += 1
    # If there are no gold positives the recall is set to zero
    if gold_positives == 0:
        r = 0
    else:
        # Computes recall as the number of true positives divided by the number of gold positives
        r = true_positives / gold_positives
    return r


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the precision of the provided predictions."""
    # Raises an error if the lengths of predictions and expected are not equal or either is empty
    stats_errors(predictions, expected)
    true_positives = 0
    system_positives = 0
    for x in range(len(predictions)):
        # If the prediction matches the label, the number of positives in the predictions set is incremented by 1
        if predictions[x] == label:
            system_positives += 1
            # If the prediction also matches the expected, the number of true positives is incremented by 1
            if predictions[x] == expected[x]:
                true_positives += 1
    # If there are no system_positives the precision is set to zero
    if system_positives == 0:
        p = 0
    else:
        # Computes precision as the number of true positives divided by the number of system positives
        p = true_positives / system_positives
    return p


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    r = recall(predictions, expected, label)
    p = precision(predictions, expected, label)
    if p == 0 and r == 0:
        f_1 = 0
    else:
        f_1 = (2 * p * r) / (p + r)
    return f_1


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    """Return the bigrams contained in a sequence."""
    # turns the sentence into a list
    token_list = [w.lower() for w in sentence]
    # adds the start and end tokens to the list
    token_list = [START_TOKEN] + token_list + [END_TOKEN]
    # uses list comprehension to add a tuple consisting of each adjacent pair of words to a list
    output = [tuple(token_list[x:x + 2]) for x in range(len(token_list) - 1)]
    return output


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract unigram features from an instance."""
        # Lowercases each token and adds them to a set
        s = {token.lower() for sentence in instance.sentences for token in sentence}
        # Creates a new classification instance
        c = ClassificationInstance(instance.label, s)
        return c


class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract bigram features from an instance."""
        # Creates bigrams for each sentence in the instance and adds all the bigrams to a set
        s = {str(bi_gram) for sentence in instance.sentences for bi_gram in bigrams(sentence)}
        # Creates a new classification instance
        c = ClassificationInstance(instance.label, s)
        return c


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        """Extract features for all three tokens from an instance."""
        # Adds the features to a list with the new formatting
        features = [f"left_tok={instance.left_context}", f"split_tok={instance.token}",
                    f"right_tok={instance.right_context}"]
        # Creates a new classification instance
        c = ClassificationInstance(instance.label, features)
        return c


class InstanceCounter:
    """Holds counts of the labels and features seen during training.

    See the assignment for an explanation of each method."""

    def __init__(self) -> None:
        self.label_counts = Counter()
        self.label_total = int(0)
        self.feature_label_count = defaultdict(Counter)
        self.label_list = set()
        self.features_set = set()
        self.total_features_per_label = Counter()

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        for instance in instances:
            lbl = instance.label
            # Increment the counter at the label
            self.label_counts[lbl] += 1
            # Increments the total number of labels seen
            self.label_total += 1
            # Adds the label to the set of labels
            self.label_list.add(lbl)
            # Iterates over the features of the ClassificationInstance
            for feature in instance.features:
                # Adds one to the joint count of the feature given the label, l
                self.feature_label_count[lbl][feature] += 1
                # Adds the feature to the set of all features
                self.features_set.add(feature)
                # Adds one to the count of total features for that label
                self.total_features_per_label[lbl] += 1
        self.label_list = list(self.label_list)

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.label_total

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        return self.feature_label_count[label][feature]

    def labels(self) -> list[str]:
        return self.label_list

    def feature_vocab_size(self) -> int:
        return len(self.features_set)

    def feature_set(self) -> set[str]:
        return self.features_set

    def total_feature_count_for_label(self, label: str) -> int:
        return self.total_features_per_label[label]


class NaiveBayesClassifier:
    """Perform classification using naive Bayes.

    See the assignment for an explanation of each method."""

    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        # The prior probability is equal to how many times the label shows up divided by the total number of labels
        return self.instance_counter.label_count(label) / self.instance_counter.total_labels()

    def likelihood_prob(self, feature: str, label) -> float:
        # The numerator is the count of the feature given a label plus k
        numerator = self.instance_counter.feature_label_joint_count(feature, label) + self.k
        # The denominator is N + V*k
        # N is the total count of all the features of the class which is the output of total_feature_count_for_label
        # V is the total of unique features in the class which is the output of feature_vocab_size
        denominator = self.instance_counter.total_feature_count_for_label(label) + \
            (self.k * self.instance_counter.feature_vocab_size())
        return numerator / denominator

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        # The log of the prior probability of the class is assigned to a variable.
        # This is only computed once.
        log_prior = math.log(self.prior_prob(label))
        log_likelihood = 0
        for feature in features:
            # Checks is the feature was seen in the training data.
            if feature in self.instance_counter.feature_set():
                # The log probability of each feature given the class is summed.
                log_likelihood += math.log(self.likelihood_prob(feature, label))
        # The sum of prior probability
        return log_prior + log_likelihood

    def classify(self, features: Sequence[str]) -> str:
        # Create a list of tuples consisting of the probability of a given label and the label
        list_of_probs = [(self.log_posterior_prob(features, label), label)
                         for label in self.instance_counter.labels()]
        # find the maximum of the tuples
        maximum_tuple = max(list_of_probs)
        # Return the label of the
        return maximum_tuple[1]

    def test(
            self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        predictions_list = []
        true_list = []
        for instance in instances:
            # Adds the predicted label based on the features of the instance to the list of predictions
            predictions_list.append(self.classify(instance.features))
            # Adds the actual label to the list of the true labels
            true_list.append(instance.label)
        return predictions_list, true_list


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = float(0.5)

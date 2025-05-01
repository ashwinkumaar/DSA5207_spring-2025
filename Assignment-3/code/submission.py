import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum

import numpy as np


class SmoothingMethod(str, Enum):
    """Smoothing methods for emission and transition probabiities"""

    LAPLACE = "laplace"
    WRITTEN_BELL = "witten_bell"
    GOOD_TURING = "good_turing"
    INTERPOLATED = "interpolated"


class Model(ABC):
    """
    This abstract class defines a general scheme for sequence tagging models.

        Attributes:
        counts     An object which contains data structures that stores word and tag counts
                   for some text.

    """

    def __init__(self, train_data):
        self.train_data = train_data

    def calculate_accuracy(self, true_tags, predicted_tags, train_words):
        """Evaluates the predicted tags against the true tags, including known and novel word accuracy."""

        if len(true_tags) != len(predicted_tags):
            raise ValueError(
                f"Length of true and predicted tags are different. \
                    True tags: {len(true_tags):,}. \
                        Predicted tags: {len(predicted_tags):,}"
            )

        true_tags_filtered = [tag for tag in true_tags if tag != "###"]
        predicted_tags_filtered = [tag for tag in predicted_tags if tag != "###"]

        # Initialize counters
        correct_tags = 0
        total_tags = 0
        correct_known_tags = 0
        total_known_tags = 0
        correct_novel_tags = 0
        total_novel_tags = 0

        # Evaluate accuracy
        for i, (predicted_tag, true_tag) in enumerate(
            zip(predicted_tags_filtered, true_tags_filtered)
        ):
            if predicted_tag == true_tag:
                correct_tags += 1

            # Check if the word is known or novel
            word = true_tags[i]  # Assuming true_tags contains the words
            if word in train_words:
                total_known_tags += 1
                if predicted_tag == true_tag:
                    correct_known_tags += 1
            else:
                total_novel_tags += 1
                if predicted_tag == true_tag:
                    correct_novel_tags += 1

            total_tags += 1

        # Calculate accuracies
        overall_accuracy = correct_tags / total_tags if total_tags > 0 else 0.0
        known_word_accuracy = (
            correct_known_tags / total_known_tags if total_known_tags > 0 else 0.0
        )
        novel_word_accuracy = (
            correct_novel_tags / total_novel_tags if total_novel_tags > 0 else 0.0
        )

        return overall_accuracy, known_word_accuracy, novel_word_accuracy

    @abstractmethod
    def fit(self, *args):
        """
        Fits the model.
        """
        pass

    @abstractmethod
    def predict(self, *args):
        """
        Predicts from the given X data.
        """
        pass

    @abstractmethod
    def fit_predict(self, *args):
        """fit and predict"""
        self.fit(*args)
        return self.predict(*args)

    @abstractmethod
    def fit_predict_evaluate(self, true_tags, *args):
        """fit, predict and calculate accuracy"""
        predicted_tags = self.fit_predict(*args)
        return self.calculate_accuracy(true_tags, predicted_tags)


class HMM_Model(Model):
    """
    Implementation of a HMM model for sequence labeling task.
    """

    def __init__(
        self,
        train_data,
        emission_smoothing_method: SmoothingMethod = None,
        transition_smoothing_method: SmoothingMethod = None,
        lambda_value: float = 0.7,
        n_lines_debug_print: int = None,
    ):
        super().__init__(train_data)
        self.emission_smoothing_method = emission_smoothing_method
        self.transition_smoothing_method = transition_smoothing_method
        self.lambda_value = lambda_value
        self.word_tag_counts, self.tag_counts, self.tag_tag_counts = (
            self._compute_counts()
        )
        self.vocabulary_size = len(set(word for word, _ in train_data))
        self.tags = sorted(self.tag_counts.keys())
        self.num_tags = len(self.tags)

        # Initialize attributes to None or a default value
        self.emission_probs = None
        self.transition_probs = None

        self.n_lines_debug_print = n_lines_debug_print
        if n_lines_debug_print:
            print(f"Number of training word/tag pairs: {len(train_data)}")
            print(
                f"word_tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.word_tag_counts.items())[:n_lines_debug_print])} }}"
            )
            print(
                f"tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.tag_counts.items())[:n_lines_debug_print])} }}"
            )
            print(
                f"tag_tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.tag_tag_counts.items())[:n_lines_debug_print])} }}"
            )
            print(f"tags: {self.tags[:n_lines_debug_print]}")
            print(f"vocabulary size: {self.vocabulary_size}")

    def _compute_counts(self):
        """Computes word-tag and tag-tag counts with sentence boundary handling."""
        word_tag_counts = {}
        tag_counts = {}
        tag_tag_counts = {}
        prev_tag = "###"

        for word, tag in self.train_data:
            if tag == "###":
                prev_tag = "###"  # Reset for new sentence
                continue

            # Emission counts
            if word not in word_tag_counts:
                word_tag_counts[word] = {}
            word_tag_counts[word][tag] = word_tag_counts[word].get(tag, 0) + 1

            # Tag counts
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Transition counts
            if prev_tag not in tag_tag_counts:
                tag_tag_counts[prev_tag] = {}
            tag_tag_counts[prev_tag][tag] = tag_tag_counts[prev_tag].get(tag, 0) + 1

            prev_tag = tag

        return word_tag_counts, tag_counts, tag_tag_counts

    def _compute_emission_probs_laplace(self):
        """Computes emission probabilities with Laplace smoothing."""
        emission_probs = {}
        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                emission_probs[word][tag] = (count + 1) / (
                    self.tag_counts[tag] + self.vocabulary_size
                )
        return emission_probs

    def _compute_emission_probs_witten_bell(self):
        """Computes emission probabilities with Witten-Bell smoothing."""
        emission_probs = {}
        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                unique_words = len(self.word_tag_counts)
                z = unique_words - len(tags)
                emission_probs[word][tag] = count / (self.tag_counts[tag] + z)
        return emission_probs

    def _compute_good_turing_counts(self, counts):
        """Computes Good-Turing adjusted counts."""
        # Convert counts to integers if they are not already
        counts = {k: int(v) for k, v in counts.items()}

        freq_of_freqs = Counter(counts.values())
        adjusted_counts = {}
        for count in counts.values():
            if count + 1 in freq_of_freqs:
                adjusted_counts[count] = (
                    (count + 1) * freq_of_freqs[count + 1] / freq_of_freqs[count]
                )
            else:
                adjusted_counts[count] = count
        return adjusted_counts

    def _compute_emission_probs_good_turing(self):
        """Computes emission probabilities with Good-Turing smoothing."""
        emission_probs = {}
        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            adjusted_counts = self._compute_good_turing_counts(counts=tags)
            for tag, count in tags.items():
                emission_probs[word][tag] = (
                    adjusted_counts[count] / self.tag_counts[tag]
                )
        return emission_probs

    def _compute_emission_probs_interpolated(self, lambda_value):
        """Computes emission probabilities with interpolated smoothing."""
        emission_probs = {}
        total_words = sum(self.tag_counts.values())
        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                unigram_prob = self.tag_counts[tag] / total_words
                emission_probs[word][tag] = (
                    lambda_value * (count / self.tag_counts[tag])
                    + (1 - lambda_value) * unigram_prob
                )
        return emission_probs

    def _compute_emission_probs_base(self):
        """Computes emission probabilities P(word|tag), excluding '###' tag."""
        emission_probs = {}

        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                if tag != "###":  # Skip emission for sentence boundary tag
                    emission_probs[word][tag] = count / self.tag_counts[tag]

        return emission_probs

    def compute_emission_probs(self, method=None):
        """Computes emission probabilities P(word|tag), excluding '###' tag."""
        if method == SmoothingMethod.LAPLACE:
            return self._compute_emission_probs_laplace()
        elif method == SmoothingMethod.WRITTEN_BELL:
            return self._compute_emission_probs_witten_bell()
        elif method == SmoothingMethod.GOOD_TURING:
            return self._compute_emission_probs_good_turing()
        elif method == SmoothingMethod.INTERPOLATED:
            return self._compute_emission_probs_interpolated(
                lambda_value=self.lambda_value
            )

        return self._compute_emission_probs_base()

    def _compute_transition_probs_laplace(self):
        """Computes transition probabilities with Laplace smoothing."""
        transition_probs = {}
        for prev_tag, next_tags in self.tag_tag_counts.items():
            transition_probs[prev_tag] = {}
            total_transitions = (
                sum(next_tags.values()) + self.num_tags
            )  # Add num_tags for smoothing

            for next_tag in self.tag_counts:
                count = next_tags.get(next_tag, 0)
                transition_probs[prev_tag][next_tag] = (count + 1) / total_transitions

        return transition_probs

    def _compute_transition_probs_interpolated(self, lambda_value):
        """Computes transition probabilities with interpolated smoothing."""
        transition_probs = {}
        total_tags = sum(self.tag_counts.values())

        for prev_tag, next_tags in self.tag_tag_counts.items():
            transition_probs[prev_tag] = {}
            total_transitions = sum(next_tags.values())

            for next_tag in self.tag_counts:
                bigram_prob = (
                    next_tags.get(next_tag, 0) / total_transitions
                    if total_transitions > 0
                    else 0
                )
                unigram_prob = self.tag_counts[next_tag] / total_tags
                transition_probs[prev_tag][next_tag] = (
                    lambda_value * bigram_prob + (1 - lambda_value) * unigram_prob
                )

            # for next_tag, count in next_tags.items():
            #     bigram_prob = count / total_transitions if total_transitions > 0 else 0
            #     unigram_prob = self.tag_counts[next_tag] / total_tags
            #     transition_probs[prev_tag][next_tag] = (
            #         lambda_value * bigram_prob + (1 - lambda_value) * unigram_prob
            #     )

        return transition_probs

    def _compute_transition_probs_base(self):
        """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
        transition_probs = {}

        for prev_tag, next_tags in self.tag_tag_counts.items():
            transition_probs[prev_tag] = {}
            total_transitions = sum(next_tags.values())

            for next_tag, count in next_tags.items():
                transition_probs[prev_tag][next_tag] = count / total_transitions

        return transition_probs

    def compute_transition_probs(self, method=None):
        """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
        if method == SmoothingMethod.LAPLACE:
            return self._compute_transition_probs_laplace()
        elif method == SmoothingMethod.INTERPOLATED:
            return self._compute_transition_probs_interpolated(
                lambda_value=self.lambda_value
            )

        return self._compute_transition_probs_base()

    def _viterbi_sentence(
        self, sentence_words, transition_probs, emission_probs, possible_tags
    ):
        """Viterbi decoding for a single sentence (list of words, no ###)."""
        n = len(sentence_words)
        V = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        for i, word in enumerate(sentence_words):
            for tag in possible_tags:
                emission_prob = emission_probs.get(word, {}).get(tag, 1e-10)

                if i == 0:
                    trans_prob = transition_probs.get("###", {}).get(tag, 1e-10)
                    V[i][tag] = trans_prob * emission_prob
                    backpointer[i][tag] = "###"
                else:
                    best_prev_tag = None
                    best_prob = -1

                    for prev_tag in possible_tags:
                        trans_prob = transition_probs.get(prev_tag, {}).get(tag, 1e-10)
                        prob = V[i - 1][prev_tag] * trans_prob * emission_prob
                        if prob > best_prob:
                            best_prob = prob
                            best_prev_tag = prev_tag

                    V[i][tag] = best_prob
                    backpointer[i][tag] = best_prev_tag

        # Backtrace
        last_tag = max(V[-1], key=V[-1].get)
        tags = [last_tag]
        for i in range(n - 1, 0, -1):
            tags.insert(0, backpointer[i][tags[0]])

        return tags

    def viterbi_full(self, test_words, transition_probs, emission_probs):
        """Viterbi method"""
        possible_tags = [t for t in set(transition_probs) if t != "###"]
        output_tags = []

        sentence = []
        for word in test_words:
            if word == "###":
                if sentence:
                    tags = self._viterbi_sentence(
                        sentence, transition_probs, emission_probs, possible_tags
                    )
                    output_tags.extend(tags)
                    sentence = []
                output_tags.append("###")  # preserve boundary
            else:
                sentence.append(word)

        if sentence:
            tags = self._viterbi_sentence(
                sentence, transition_probs, emission_probs, possible_tags
            )
            output_tags.extend(tags)

        return output_tags

    def fit(self, *args):
        self.emission_probs = self.compute_emission_probs(
            method=self.emission_smoothing_method
        )
        self.transition_probs = self.compute_transition_probs(
            method=self.transition_smoothing_method,
        )

        if self.n_lines_debug_print:
            print(
                f"emission probabilities P(word|tag): {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.emission_probs.items())[:self.n_lines_debug_print])} }}"
            )
            print(
                f"transition probabilities P(tag2|tag1): {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.transition_probs.items())[:self.n_lines_debug_print])} }}"
            )

    def predict(self, test_words, *args):
        if not self.emission_probs or not self.transition_probs:
            raise ValueError("Model not fit")

        predicted_tags = self.viterbi_full(
            test_words, self.transition_probs, self.emission_probs
        )

        if self.n_lines_debug_print:
            print(f"Test words: {test_words[:self.n_lines_debug_print]}")

        return predicted_tags

    def fit_predict(self, test_words, *args):
        self.fit()
        return self.predict(test_words)

    def fit_predict_evaluate(self, test_words, true_tags, train_words=None, *args):
        predicted_tags = self.fit_predict(test_words)
        return self.calculate_accuracy(true_tags, predicted_tags, train_words)

    def calculate_accuracy(self, true_tags, predicted_tags, train_words=None):
        if not train_words:
            train_words = set(word for word, _ in self.train_data)

        overall_accuracy, known_word_accuracy, novel_word_accuracy = (
            super().calculate_accuracy(true_tags, predicted_tags, train_words)
        )

        if self.n_lines_debug_print:
            print(f"Predicted tags: {predicted_tags[:self.n_lines_debug_print]}")
            print(f"Actual tags:    {true_tags[:self.n_lines_debug_print]}")
            print("Evaluating on test data...")
            print(f"Overall accuracy on the test data: {overall_accuracy * 100:.2f}%")
            print(f"Known word accuracy: {known_word_accuracy * 100:.2f}%")
            print(f"Novel word accuracy: {novel_word_accuracy * 100:.2f}%")

        return overall_accuracy, known_word_accuracy, novel_word_accuracy


def read_data_file(filename):
    """Reads data from a specified file and returns a list of lines."""
    try:
        with open(filename, "r") as f:
            data = f.read().splitlines()
        print(f"Successfully read {filename}.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None


def parse_data(data):
    """Parses data where each line is in the format 'x/y'."""
    word_tag_pairs = []
    for line in data:
        parts = line.strip().split("/")
        if len(parts) == 2:
            word_tag_pairs.append((parts[0], parts[1]))
    return word_tag_pairs


class HMM_Bigram_Model(HMM_Model):
    def __init__(
        self,
        train_data,
        emission_smoothing_method=None,
        transition_smoothing_method=None,
        lambda_value=0.7,
        n_lines_debug_print=None,
    ):
        super().__init__(
            train_data,
            emission_smoothing_method,
            transition_smoothing_method,
            lambda_value,
            n_lines_debug_print,
        )
        self.bigram_counts = self._compute_bigram_counts()

    def _compute_bigram_counts(self):
        """Computes bigram counts for tags."""
        bigram_counts = {}
        prev_tag = "###"  # Start with the sentence boundary tag

        for _, tag in self.train_data:
            if tag == "###":
                prev_tag = "###"  # Reset for new sentence
                continue

            if prev_tag not in bigram_counts:
                bigram_counts[prev_tag] = {}
            bigram_counts[prev_tag][tag] = bigram_counts[prev_tag].get(tag, 0) + 1

            prev_tag = tag

        return bigram_counts

    def _compute_bigram_probs_laplace(self):
        """Computes bigram probabilities with Laplace smoothing."""
        bigram_probs = {}

        for prev_tag, next_tags in self.bigram_counts.items():
            bigram_probs[prev_tag] = {}
            total_transitions = (
                sum(next_tags.values()) + self.num_tags
            )  # Add num_tags for smoothing

            for next_tag in self.tag_counts:
                count = next_tags.get(next_tag, 0)
                bigram_probs[prev_tag][next_tag] = (count + 1) / total_transitions

        return bigram_probs

    def _compute_bigram_probs_interpolated(self):
        """Computes bigram probabilities with interpolated smoothing."""
        bigram_probs = {}
        total_tags = sum(self.tag_counts.values())

        for prev_tag, next_tags in self.bigram_counts.items():
            bigram_probs[prev_tag] = {}
            total_transitions = sum(next_tags.values())

            for next_tag in self.tag_counts:
                bigram_prob = (
                    next_tags.get(next_tag, 0) / total_transitions
                    if total_transitions > 0
                    else 0
                )
                unigram_prob = self.tag_counts[next_tag] / total_tags
                bigram_probs[prev_tag][next_tag] = (
                    self.lambda_value * bigram_prob
                    + (1 - self.lambda_value) * unigram_prob
                )

        return bigram_probs

    def _compute_bigram_probs_base(self):
        """Computes bigram probabilities P(tag2|tag1)."""
        bigram_probs = {}

        for prev_tag, next_tags in self.bigram_counts.items():
            bigram_probs[prev_tag] = {}
            total_transitions = sum(next_tags.values())

            for next_tag, count in next_tags.items():
                bigram_probs[prev_tag][next_tag] = count / total_transitions

        return bigram_probs

    def compute_bigram_probs(self, method=None):
        """Computes bigram probabilities with specified smoothing method."""
        if method == SmoothingMethod.LAPLACE:
            return self._compute_bigram_probs_laplace()
        elif method == SmoothingMethod.INTERPOLATED:
            return self._compute_bigram_probs_interpolated()

        # Default to no smoothing
        return self._compute_bigram_probs_base()


def train_and_test(
    train_data_lines,
    test_data_lines,
    dev_data_lines=None,
):
    """
    Performs training and testing using the provided data lines.
    """
    train_data_parsed = parse_data(train_data_lines)
    test_data_parsed = parse_data(test_data_lines)

    test_words = [word for word, _ in test_data_parsed]
    true_tags = [tag for _, tag in test_data_parsed]

    model = HMM_Model

    # emission_smoothing_methods = list(SmoothingMethod) + [None]
    emission_smoothing_methods = [SmoothingMethod.GOOD_TURING, None]
    transition_smoothing_methods = [
        SmoothingMethod.LAPLACE,
        SmoothingMethod.INTERPOLATED,
        None,
    ]
    # transition_smoothing_methods = [None]
    # Different lambda values for interpolation
    lambda_values = np.linspace(0.1, 0.9, 5).tolist()
    results = []

    for emission_smoothing in emission_smoothing_methods:
        for transition_smoothing in transition_smoothing_methods:
            if transition_smoothing == SmoothingMethod.INTERPOLATED:
                for lambda_value in lambda_values:
                    print(
                        f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}, Lambda: {lambda_value}"
                    )
                    hmm_model = model(
                        train_data=train_data_parsed,
                        emission_smoothing_method=emission_smoothing,
                        transition_smoothing_method=transition_smoothing,
                        lambda_value=lambda_value,
                    )
                    # predicted_tags = hmm_model.fit_predict(test_words)
                    overall_accuracy, known_word_accuracy, novel_word_accuracy = (
                        hmm_model.fit_predict_evaluate(test_words, true_tags)
                    )

                    # Append results for the current combination of smoothing methods
                    results.append(
                        [
                            emission_smoothing,
                            f"{transition_smoothing} (λ={lambda_value:.2f})",
                            overall_accuracy,
                            known_word_accuracy,
                            novel_word_accuracy,
                        ]
                    )
            else:
                print(
                    f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}"
                )
                hmm_model = model(
                    train_data=train_data_parsed,
                    emission_smoothing_method=emission_smoothing,
                    transition_smoothing_method=transition_smoothing,
                )
                # predicted_tags = hmm_model.fit_predict(test_words)
                overall_accuracy, known_word_accuracy, novel_word_accuracy = (
                    hmm_model.fit_predict_evaluate(test_words, true_tags)
                )

                # Append results for the current combination of smoothing methods
                results.append(
                    [
                        emission_smoothing,
                        transition_smoothing,
                        overall_accuracy,
                        known_word_accuracy,
                        novel_word_accuracy,
                    ]
                )

    # Sort results by novel word accuracy in descending order
    results.sort(key=lambda x: x[4], reverse=True)

    # Display results in a table
    headers = [
        "Emission Smoothing",
        "Transition Smoothing",
        "Overall Accuracy",
        "Known Word Accuracy",
        "Novel Word Accuracy",
    ]
    print(
        f"\n| {headers[0]:<30} | {headers[1]:<40} | {headers[2]:<20} | {headers[3]:<20} | {headers[4]:<20}|"
    )
    print("=" * 145)
    for row in results:
        emission_method = str(row[0]) if row[0] is not None else "None"
        transition_method = str(row[1]) if row[1] is not None else "None"
        print(
            f"| {emission_method:<30} | {transition_method:<40} | {row[2]:<20.5f} | {row[3]:<20.5f} | {row[4]:<20.5f}|"
        )


def main():
    # check if the correct number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python submission.py -train <train_file> -test <test_file>")
        return

    # extract arguments
    # train_arg = sys.argv[1]
    train_base = sys.argv[2]
    # test_arg = sys.argv[3]
    test_base = sys.argv[4]

    # # validate the arguments
    # if train_arg != "-train" or test_arg != "-test":
    #     print("Invalid arguments. Use -train and -test")
    #     return

    path = "data"

    train_file = os.path.join(path, train_base)
    test_file = os.path.join(path, test_base)

    # print the arguments to verify
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")

    train_data = read_data_file(train_file)
    test_data = read_data_file(test_file)

    if (train_base == "ictrain" and test_base == "ictest") or (train_base == "entrain"):
        train_and_test(train_data_lines=train_data, test_data_lines=test_data)
    else:
        print("Error: Could not load training and/or testing data. Exiting.")


if __name__ == "__main__":
    main()

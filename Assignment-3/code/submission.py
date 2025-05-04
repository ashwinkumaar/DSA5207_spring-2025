import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
import argparse

import numpy as np

RARE_SYMBOL = "_RARE_"
LOG_PROB_OF_ZERO = -1000


def log2(value):
    return math.log2(value) if value > 0 else LOG_PROB_OF_ZERO


def subcategorize(word):
    # Check if word is a tuple and extract the first element
    if isinstance(word, tuple):
        print(word)
        word = word[0]

    # Convert word to lowercase for consistency
    word_lower = word.lower()

    if re.search(r"[€£¥₹$]", word):
        return "_CURRENCY_"
    elif re.search(r"[+\-*/=<>]", word):
        return "_SYMBOL_"
    elif not re.search(r"\w", word):
        return "_PUNCS_"
    elif re.search(r"[A-Z]", word):
        return "_CAPITAL_"
    elif re.search(r"\d", word):
        return "_CARDINAL_"
    elif re.search(
        r"(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)", word_lower
    ):
        return "_NOUNLIKE_"
    elif re.search(r"(ate\b|fy\b|ize\b|ing\b|ed\b|s\b|\ben|\bem)", word_lower):
        return "_VERBLIKE_"
    elif re.search(r"(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)", word_lower):
        return "_ADJLIKE_"
    elif re.search(r"(ly\b)", word_lower):
        return "_ADVLIKE_"
    elif re.search(r"(\bthe\b|\band\b|\bof\b|\bin\b|\bat\b)", word_lower):
        return "_FUNCTION_"
    elif re.search(r"(\bthere\b)", word_lower):
        return "_EXISTENTIAL_"
    elif re.search(r"(\buh\b|\boh\b|\bhey\b|\bhi\b)", word_lower):
        return "_INTERJECTION_"
    elif len(word) <= 3:
        return "_SHORT_"
    else:
        return RARE_SYMBOL

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
        return self.calculate_accuracy(true_tags, predicted_tags, None)

    def calc_known(self, train_data, rare_word_max_freq):
        known_words = set()
        word_c = defaultdict(int)

        for word, _ in train_data:
            word_c[word] += 1

        for word, count in word_c.items():
            if count > rare_word_max_freq:
                known_words.add(word)
        return known_words

    def replace_rare(self, train_data, rare_word_max_freq):
        known_words = self.calc_known(train_data, rare_word_max_freq)
        output = []

        for word, tag in train_data:
            if word not in known_words:
                word = subcategorize(word)

            output.append((word, tag))

        return output, known_words


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
        rare_word_max_frequency: int = 0,
    ):
        replace_rare_words = False
        if rare_word_max_frequency > 0:
            replace_rare_words = True
            train_data, known_words = self.replace_rare(
                train_data, rare_word_max_frequency
            )
            self.known_words = known_words
        super().__init__(train_data)
        self.rare_word_max_frequency = rare_word_max_frequency
        self.emission_smoothing_method = emission_smoothing_method
        self.transition_smoothing_method = transition_smoothing_method
        self.lambda_value = lambda_value
        self.word_tag_counts, self.tag_counts, self.tag_tag_counts = (
            self._compute_counts()
        )
        self.vocabulary_size = len(set(word for word, _ in train_data))
        self.tags = sorted(self.tag_counts.keys())
        self.num_tags = len(self.tags)
        self.replace_rare_words = replace_rare_words

        # Initialize attributes to None or a default value
        self.emission_probs = None
        self.transition_probs = None

        self.n_lines_debug_print = n_lines_debug_print

        if n_lines_debug_print:
            self._debug_print_initial_counts()

    def _debug_print_initial_counts(self):
        print(f"Number of training word/tag pairs: {len(self.train_data)}")
        print(f"word_tag_counts: {self._format_debug_output(self.word_tag_counts)}")
        print(f"tag_counts: {self._format_debug_output(self.tag_counts)}")
        print(f"tag_tag_counts: {self._format_debug_output(self.tag_tag_counts)}")
        print(f"tags: {self.tags[:self.n_lines_debug_print]}")
        print(f"vocabulary size: {self.vocabulary_size}")

    def _format_debug_output(self, data):
        return ", ".join(
            f"{key}: {value}"
            for key, value in list(data.items())[: self.n_lines_debug_print]
        )

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
                emission_probs[word][tag] = log2(count + 1) - log2(
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
                emission_probs[word][tag] = log2(count) - log2(self.tag_counts[tag] + z)
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
                    log2((count + 1))
                    + log2(freq_of_freqs[count + 1])
                    - log2(freq_of_freqs[count])
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
                emission_probs[word][tag] = log2(adjusted_counts[count]) - log2(
                    self.tag_counts[tag]
                )
        return emission_probs

    def _compute_emission_probs_interpolated(self, lambda_value):
        """Computes emission probabilities with interpolated smoothing."""
        emission_probs = {}
        total_words = sum(self.tag_counts.values())
        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                unigram_prob = log2(self.tag_counts[tag]) - log2(total_words)
                emission_probs[word][tag] = log2(
                    (lambda_value * (2 ** (log2(count) - log2(self.tag_counts[tag]))))
                    + ((1 - lambda_value) * (2**unigram_prob))
                )
        return emission_probs

    def _compute_emission_probs_base(self):
        """Computes emission probabilities P(word|tag), excluding '###' tag."""
        emission_probs = {}

        for word, tags in self.word_tag_counts.items():
            emission_probs[word] = {}
            for tag, count in tags.items():
                if tag != "###":  # Skip emission for sentence boundary tag
                    emission_probs[word][tag] = log2(count) - log2(self.tag_counts[tag])

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
                transition_probs[prev_tag][next_tag] = log2(count + 1) - log2(
                    total_transitions
                )

        return transition_probs

    def _compute_transition_probs_interpolated(self, lambda_value):
        """Computes transition probabilities with interpolated smoothing."""
        transition_probs = {}
        total_tags = sum(self.tag_counts.values())

        for prev_tag, next_tags in self.tag_tag_counts.items():
            transition_probs[prev_tag] = {}
            total_transitions = sum(next_tags.values())

            for next_tag in self.tag_counts:
                bigram_prob = log2(next_tags.get(next_tag, 0)) - log2(total_transitions)

                unigram_prob = log2(self.tag_counts[next_tag]) - log2(total_tags)
                transition_probs[prev_tag][next_tag] = log2(
                    (lambda_value * (2**bigram_prob))
                    + ((1 - lambda_value) * (2**unigram_prob))
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
                transition_probs[prev_tag][next_tag] = log2(count) - log2(
                    total_transitions
                )

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
                emission_prob = emission_probs.get(word, {}).get(tag, LOG_PROB_OF_ZERO)

                if i == 0:
                    trans_prob = transition_probs.get("###", {}).get(
                        tag, LOG_PROB_OF_ZERO
                    )
                    V[i][tag] = trans_prob + emission_prob
                    backpointer[i][tag] = "###"
                else:
                    best_prev_tag = None
                    best_prob = float("-inf")

                    for prev_tag in possible_tags:
                        trans_prob = transition_probs.get(prev_tag, {}).get(
                            tag, LOG_PROB_OF_ZERO
                        )
                        prob = V[i - 1][prev_tag] + trans_prob + emission_prob
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
                f"""emission probabilities P(word|tag): {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.emission_probs.items())[:self.n_lines_debug_print])} }}"""
            )
            print(
                f"""transition probabilities P(tag2|tag1): {{ {', '.join(f'{key}: {value}' for key, value in list(
                self.transition_probs.items())[:self.n_lines_debug_print])} }}"""
            )

    def predict(self, test_words, *args):
        if not self.emission_probs or not self.transition_probs:
            raise ValueError("Model not fit")

        if self.replace_rare_words:
            test_words_rare = []
            for word in test_words:
                if word not in self.known_words:
                    word = subcategorize(word)
                test_words_rare.append(word)

            test_words = test_words_rare

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


def read_data_file(filename):
    """Reads data from a specified file and returns a list of lines."""
    try:
        with open(filename, "r") as f:
            data = f.read().splitlines()
        # print(f"Successfully read {filename}.")
        return data
    except FileNotFoundError:
        raise ValueError(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading '{filename}': {e}")


def parse_data(data, is_test=False):
    """Parses data where each line is in the format 'x/y'."""
    word_tag_pairs = []
    for line in data:
        parts = line.strip().split("/")
        if len(parts) == 2 and not is_test:
            word_tag_pairs.append((parts[0], parts[1]))
        else:
            word_tag_pairs.append(parts[0])
    return word_tag_pairs


def print_results(results):
    # Sort results by novel word accuracy in descending order
    results.sort(key=lambda x: x[3], reverse=True)

    # Display results in a table
    headers = [
        "Emission Smoothing",
        "Transition Smoothing",
        "Rare Word Max Frequency",
        "Overall Accuracy",
        "Known Word Accuracy",
        "Novel Word Accuracy",
    ]
    print(
        f"\n| {headers[0]:<30} | {headers[1]:<40} | {headers[2]:<25} | {headers[3]:<20} | {headers[4]:<20} | {headers[5]:<20}|"
    )
    print("=" * 175)
    for row in results:
        emission_method = str(row[0]) if row[0] is not None else "None"
        transition_method = str(row[1]) if row[1] is not None else "None"
        print(
            f"| {emission_method:<30} | {transition_method:<40} | {row[2]:<25} | {row[3]:<20.5f} | {row[4]:<20.5f} | {row[5]:<20.5f}|"
        )


def train_and_test(
    train_file_name,
    dev_file_name,
    test_file_name,
):
    """
    Performs training and testing using the provided data lines.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')


    train_dir = os.path.join(data_dir, train_file_name)
    train_data_lines = read_data_file(train_dir)

    dev_data_lines = None
    if dev_file_name:
        dev_dir = os.path.join(data_dir, dev_file_name)
        dev_data_lines = read_data_file(dev_dir)

    test_dir = os.path.join(data_dir, test_file_name)
    test_data_lines = read_data_file(test_dir)

    train_data_parsed = parse_data(train_data_lines)

    dev_data_parsed = parse_data(dev_data_lines) if dev_data_lines else [(None, None)]

    dev_words = [word for word, _ in dev_data_parsed]
    dev_tags = [tag for _, tag in dev_data_parsed]

    model = HMM_Model

    # Define your smoothing methods and other parameters
    emission_smoothing_methods = [SmoothingMethod.GOOD_TURING, None]
    transition_smoothing_methods = [
        SmoothingMethod.LAPLACE,
        # SmoothingMethod.INTERPOLATED,
        None,
    ]
    lambda_values = np.linspace(0.1, 0.9, 5).tolist()
    rare_word_max_freq = [1, 2]

    # Initialize a list to store results
    results = []

    best_model = None
    best_overall_accuracy = 0.0

    dev_data_lines = None
    if dev_data_lines:
        # Perform grid search
        for emission_smoothing in emission_smoothing_methods:
            for transition_smoothing in transition_smoothing_methods:
                for rare_freq in rare_word_max_freq:
                    if transition_smoothing == SmoothingMethod.INTERPOLATED:
                        for lambda_value in lambda_values:
                            # print(
                            #     f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}, Lambda: {lambda_value}, Rare Freq: {rare_freq}"
                            # )
                            hmm_model = model(
                                train_data=train_data_parsed,
                                emission_smoothing_method=emission_smoothing,
                                transition_smoothing_method=transition_smoothing,
                                lambda_value=lambda_value,
                                rare_word_max_frequency=rare_freq,
                            )
                            (
                                overall_accuracy,
                                known_word_accuracy,
                                novel_word_accuracy,
                            ) = hmm_model.fit_predict_evaluate(dev_words, dev_tags)

                            # Append results for the current combination of smoothing methods
                            results.append(
                                [
                                    emission_smoothing,
                                    f"{transition_smoothing} (λ={lambda_value:.2f})",
                                    rare_freq,
                                    overall_accuracy,
                                    known_word_accuracy,
                                    novel_word_accuracy,
                                ]
                            )

                            # Update best model if current overall accuracy is higher
                            if overall_accuracy > best_overall_accuracy:
                                best_overall_accuracy = overall_accuracy
                                best_model = hmm_model
                    else:
                        # print(
                        #     f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}, Rare Freq: {rare_freq}"
                        # )
                        hmm_model = model(
                            train_data=train_data_parsed,
                            emission_smoothing_method=emission_smoothing,
                            transition_smoothing_method=transition_smoothing,
                            rare_word_max_frequency=rare_freq,
                        )
                        overall_accuracy, known_word_accuracy, novel_word_accuracy = (
                            hmm_model.fit_predict_evaluate(dev_words, dev_tags)
                        )

                        # Append results for the current combination of smoothing methods
                        results.append(
                            [
                                emission_smoothing,
                                transition_smoothing,
                                rare_freq,
                                overall_accuracy,
                                known_word_accuracy,
                                novel_word_accuracy,
                            ]
                        )

                        # Update best model if current overall accuracy is higher
                        if overall_accuracy > best_overall_accuracy:
                            best_overall_accuracy = overall_accuracy
                            best_model = hmm_model

        print_results(results)

    if len(dev_data_parsed) > 0 and dev_data_parsed[0][0] is not None:
        train_data_parsed = train_data_parsed + dev_data_parsed

    if best_model is None:
        best_model_final = model(
            train_data=train_data_parsed,
            emission_smoothing_method=None,
            transition_smoothing_method=None,
            rare_word_max_frequency=1,
        )
    else:
        # Train the best model on the combined train and dev data
        best_model_final = model(
            train_data=train_data_parsed,
            emission_smoothing_method=best_model.emission_smoothing_method,
            transition_smoothing_method=best_model.transition_smoothing_method,
            lambda_value=best_model.lambda_value,
            rare_word_max_frequency=best_model.rare_word_max_frequency
        )

    best_model_final.fit()
    # print(best_model.fit_predict_evaluate(test_words=dev_words, true_tags=dev_tags))

    test_words = parse_data(test_data_lines, is_test=True)
    test_predictions = best_model_final.predict(test_words)

    # Open the file for writing
    with open(os.path.join(base_dir, 'output.txt'), "w") as file:
        # Iterate over each sentence in the test data
        for word, tag in zip(test_words, test_predictions):
            # Format as word/tag
            file.write(f"{word}/{tag}\n")

    # from eval import eval as eval_fn
    # args = argparse.Namespace()
    # args.gold = os.path.join(data_dir, test_file_name)
    # args.pred = os.path.join(base_dir, 'output.txt')
    # acc = eval_fn(args)  # Call eval function here
    # return acc


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
    dev_file = os.path.join(path, "endev")
    test_file = os.path.join(path, test_base)

    # print the arguments to verify
    # print(f"Training file: {train_file}")
    # print(f"Test file: {test_file}")

    train_data = read_data_file(train_file)
    dev_data = read_data_file(dev_file)
    test_data = read_data_file(test_file)

    if (train_base == "ictrain" and test_base == "ictest") or (train_base == "entrain"):
        # train_and_test(
        #     train_data_lines=train_data,
        #     dev_data_lines=dev_data,
        #     test_data_lines=test_data,
        # )

        dev_file = "endev"
        train_and_test(
            train_file_name=train_base,
            dev_file_name=dev_file,
            test_file_name=test_base,
        )
    else:
        raise ValueError("Error: Could not load training and/or testing data. Exiting.")


if __name__ == "__main__":
    main()

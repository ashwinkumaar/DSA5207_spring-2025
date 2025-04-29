import json
import collections
import argparse
import random

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    punctuation = [".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\", "|", "*", "&", "%", "$", "#", "@", "^", "`", "~", "="]

    features = {}
    for word in ex["sentence1"] + ex["sentence2"]:
        features[word] = features.get(word, 0) + 1

    for p in punctuation:
        features.pop(p) if p in features else None

    return features
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    features = {}

    # extract unigrams
    features = extract_unigram_features(ex)

    # extract bigrams
    def get_bigrams(sentence):
        return [f"{sentence[i]}_{sentence[i + 1]}" for i in range(len(sentence) - 1)]

    for bigram in get_bigrams(ex["sentence1"]) + get_bigrams(ex["sentence2"]):
        features[bigram] = features.get(bigram, 0) + 1

    # count lexical overlap
    premise_set = set(ex["sentence1"])
    hypothesis_set = set(ex["sentence2"])
    features["lexical_overlap"] = len(premise_set & hypothesis_set)

    # count negation words
    negation_words = {"not", "no", "never", "none", "n't", "nothing", "nobody", "neither"}
    features["negation_count"] = sum(1 for word in ex["sentence1"] + ex["sentence2"] if word in negation_words)

    # word mismatch count
    features["word_mismatch"] = len(premise_set - hypothesis_set)

    # length difference
    features["length_diff"] = abs(len(ex["sentence1"]) - len(ex["sentence2"]))

    return features
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    weights = {}

    # convert training data to feature vectors
    train_examples = [(feature_extractor(ex), ex['gold_label']) for ex in train_data]
    valid_examples = [(feature_extractor(ex), ex['gold_label']) for ex in valid_data]

    for epoch in range(num_epochs):
        for feat, label in train_examples:
            # compute prediction
            pred = predict(weights, feat)

            # compute gradient
            gradient = {f: (pred - label) * v for f, v in feat.items()}

            # update weights using gradient descent
            increment(weights, gradient, -learning_rate)

        # evaluate error after each epoch
        train_error = evaluate_predictor(train_examples, lambda x: round(predict(weights, x)))
        valid_error = evaluate_predictor(valid_examples, lambda x: round(predict(weights, x)))
        print(f"Epoch {epoch + 1}: Train Error = {train_error:.4f}, Valid Error = {valid_error:.4f}")

    return weights
    # END_YOUR_CODE

import sys
import os
import numpy as np

def read_data_file(filename):
    """Reads data from a specified file and returns a list of lines."""
    try:
        with open(filename, 'r') as f:
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
        parts = line.strip().split('/')
        if len(parts) == 2:
            word_tag_pairs.append((parts[0], parts[1]))
    return word_tag_pairs


def compute_counts(train_data_parsed):
  """Computes word-tag and tag-tag counts with sentence boundary handling."""
  word_tag_counts = {}
  tag_counts = {}
  tag_tag_counts = {}
  prev_tag = '###'

  for word, tag in train_data_parsed:
    if tag == '###':
      prev_tag = '###'  # Reset for new sentence
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


def compute_emission_probabilities(word_tag_counts, tag_counts):
  """Computes emission probabilities P(word|tag), excluding '###' tag."""
  emission_probs = {}

  for word, tags in word_tag_counts.items():
    emission_probs[word] = {}
    for tag, count in tags.items():
      if tag != '###':  # Skip emission for sentence boundary tag
        emission_probs[word][tag] = count / tag_counts[tag]

  return emission_probs


def compute_transition_probabilities(tag_tag_counts, tag_counts):
  """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
  transition_probs = {}

  for prev_tag, next_tags in tag_tag_counts.items():
    transition_probs[prev_tag] = {}
    total_transitions = sum(next_tags.values())

    for next_tag, count in next_tags.items():
      transition_probs[prev_tag][next_tag] = count / total_transitions

  return transition_probs



def viterbi(words, tags, emission_probs, transition_probs):
    """Implements the Viterbi algorithm to find the most likely tag sequence."""
    n = len(words)
    viterbi_table = {}
    backpointers = {}

    # Initialization for the first word
    viterbi_table[0] = {}
    backpointers[0] = {}
    for tag in tags:
        emission_prob = emission_probs.get(words[0], {}).get(tag, 0) # Handle unseen words
        transition_from_start = transition_probs.get('###', {}).get(tag, 0)
        viterbi_table[0][tag] = transition_from_start + emission_prob
        backpointers[0][tag] = '###'

    # Recursion for subsequent words
    for t in range(1, n):
        viterbi_table[t] = {}
        backpointers[t] = {}
        for current_tag in tags:
            max_prob = 0
            best_prev_tag = None
            emission_prob = emission_probs.get(words[t], {}).get(current_tag, 0) # Handle unseen words
            for prev_tag in tags:
                transition_prob = transition_probs.get(prev_tag, {}).get(current_tag, 0)
                prob = viterbi_table[t - 1].get(prev_tag, 0) * transition_prob * emission_prob
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            viterbi_table[t][current_tag] = max_prob
            backpointers[t][current_tag] = best_prev_tag

    # Termination
    best_path_prob = 0
    last_tag = None
    for tag in tags:
        transition_to_end = transition_probs.get(tag, {}).get('###', 0)
        prob = viterbi_table[n - 1].get(tag, 0) * transition_to_end
        if prob > best_path_prob:
            best_path_prob = prob
            last_tag = tag

    # Backtrack to find the best tag sequence
    best_path = [last_tag]
    for t in range(n - 1, 0, -1):
        last_tag = backpointers[t].get(last_tag)
        best_path.insert(0, last_tag)

    return best_path


def evaluate(test_data_parsed, predicted_tags):
    """Evaluates the predicted tags against the true tags."""
    true_tags = [tag for word, tag in test_data_parsed if word != '###']
    predicted_word_tags = [tag for tag in predicted_tags if tag != '###']

    correct_tags = sum(1 for p, t in zip(predicted_word_tags, true_tags) if p == t)
    total_tags = len(true_tags)

    if total_tags == 0:
        return 0.0
    accuracy = correct_tags / total_tags
    return accuracy


def train_and_test(train_data_lines, dev_data_lines=None, test_data_lines=None):
    """
    Performs training and testing using the provided data lines.
    """
    print("Parsing training data...")
    train_data_parsed = parse_data(train_data_lines)
    print(f"Number of training word/tag pairs: {len(train_data_parsed)}")

    print("Computing counts...")
    word_tag_counts, tag_counts, tag_tag_counts = compute_counts(train_data_parsed)
    tags = sorted(tag_counts.keys())

    print(f"word_tag_counts: {word_tag_counts}")
    print(f"tag_counts: {tag_counts}")
    print(f"tag_tag_counts: {tag_tag_counts}")
    print(f"tags: {tags}")

    print("Computing emission probabilities...")
    emission_probabilities = compute_emission_probabilities(word_tag_counts, tag_counts)
    print(f"emission probabilities P(word|tag): {emission_probabilities}")
    print("Computing transition probabilities...")
    transition_probabilities = compute_transition_probabilities(tag_tag_counts, tag_counts)
    print(f"transition probabilities P(tag2|tag1): {transition_probabilities}")

    if test_data_lines:
        print("\nParsing test data...")
        test_data_parsed = parse_data(test_data_lines)
        print(f"Number of test word/tag pairs: {len(test_data_parsed)}")

        print("Running Viterbi on test data...")
        test_words = [word for word, tag in test_data_parsed]
        predicted_tags = viterbi(test_words, tags, emission_probabilities, transition_probabilities)

        print("Evaluating on test data...")
        accuracy = evaluate(test_data_parsed, predicted_tags)
        print(f"Accuracy on the test data: {accuracy * 100:.2f}%")

    print("Training and testing complete.")
    return


def main():
    # check if the correct number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python submission.py -train <train_file> -test <test_file>")
        return

    # extract arguments
    train_arg = sys.argv[1]
    train_base = sys.argv[2]
    test_arg = sys.argv[3]
    test_base = sys.argv[4]

    # # validate the arguments
    # if train_arg != "-train" or test_arg != "-test":
    #     print("Invalid arguments. Use -train and -test")
    #     return

    train_file = os.path.join('data', train_base)
    test_file = os.path.join('data', test_base)

    # print the arguments to verify
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")

    train_data = read_data_file(train_file)
    test_data = read_data_file(test_file)

    if (train_base == 'ictrain' and test_base == 'ictest') or (train_base == 'entrain'):
        train_and_test(train_data, None, test_data)
    else:
        print("Error: Could not load training and/or testing data. Exiting.")



if __name__ == '__main__':
    main()


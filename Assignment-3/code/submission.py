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


def compute_emission_probs(word_tag_counts, tag_counts):
  """Computes emission probabilities P(word|tag), excluding '###' tag."""
  emission_probs = {}

  for word, tags in word_tag_counts.items():
    emission_probs[word] = {}
    for tag, count in tags.items():
      if tag != '###':  # Skip emission for sentence boundary tag
        emission_probs[word][tag] = count / tag_counts[tag]

  return emission_probs


def compute_transition_probs(tag_tag_counts, tag_counts):
  """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
  transition_probs = {}

  for prev_tag, next_tags in tag_tag_counts.items():
    transition_probs[prev_tag] = {}
    total_transitions = sum(next_tags.values())

    for next_tag, count in next_tags.items():
      transition_probs[prev_tag][next_tag] = count / total_transitions

  return transition_probs


def viterbi_sentence(sentence_words, transition_probs, emission_probs, possible_tags):
  """Viterbi decoding for a single sentence (list of words, no ###)."""
  n = len(sentence_words)
  V = [{} for _ in range(n)]
  backpointer = [{} for _ in range(n)]

  for i, word in enumerate(sentence_words):
    for tag in possible_tags:
      emission_prob = emission_probs.get(word, {}).get(tag, 1e-10)

      if i == 0:
        trans_prob = transition_probs.get('###', {}).get(tag, 1e-10)
        V[i][tag] = trans_prob * emission_prob
        backpointer[i][tag] = '###'
      else:
        best_prev_tag = None
        best_prob = -1

        for prev_tag in possible_tags:
          trans_prob = transition_probs.get(prev_tag, {}).get(tag, 1e-10)
          prob = V[i-1][prev_tag] * trans_prob * emission_prob
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


def viterbi_full(test_words, transition_probs, emission_probs):
  possible_tags = [t for t in set(transition_probs) if t != '###']
  output_tags = []

  sentence = []
  for word in test_words:
    if word == '###':
      if sentence:
        tags = viterbi_sentence(sentence, transition_probs, emission_probs, possible_tags)
        output_tags.extend(tags)
        sentence = []
      output_tags.append('###')  # preserve boundary
    else:
      sentence.append(word)

  if sentence:
    tags = viterbi_sentence(sentence, transition_probs, emission_probs, possible_tags)
    output_tags.extend(tags)

  return output_tags


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
    emission_probs = compute_emission_probs(word_tag_counts, tag_counts)
    print(f"emission probabilities P(word|tag): {emission_probs}")
    print("Computing transition probabilities...")
    transition_probs = compute_transition_probs(tag_tag_counts, tag_counts)
    print(f"transition probabilities P(tag2|tag1): {transition_probs}")

    if test_data_lines:
        print("\nParsing test data...")
        test_data_parsed = parse_data(test_data_lines)
        print(f"Number of test word/tag pairs: {len(test_data_parsed)}")

        print("Running Viterbi on test data...")
        test_words = [word for word, tag in test_data_parsed]
        print(f"Test words: {test_words}")
        predicted_tags = viterbi_full(test_words, transition_probs, emission_probs)
        print(f"Predicted tags: {predicted_tags}")

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

    path = "F:\\OneDrive - Ashwin\\OneDrive\\NUS\\Masters\\Semester 4\\DSA5207 - Text Processing & Interpretation with Machine Learning\\Assignments\\Assignment-3\\code\\data"

    train_file = os.path.join(path, train_base)
    test_file = os.path.join(path, test_base)

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


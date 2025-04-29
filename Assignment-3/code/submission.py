import sys
import os
import numpy as np
from enum import Enum

class SmoothingMethod(str, Enum):
    LAPLACE = "laplace"
    WRITTEN_BELL = "witten_bell"
    GOOD_TURING = "good_turing"
    INTERPOLATED = "interpolated"

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

def compute_emission_probs_laplace(word_tag_counts, tag_counts, vocabulary_size):
    """Computes emission probabilities with Laplace smoothing."""
    emission_probs = {}
    for word, tags in word_tag_counts.items():
        emission_probs[word] = {}
        for tag, count in tags.items():
            emission_probs[word][tag] = (count + 1) / (tag_counts[tag] + vocabulary_size)
    return emission_probs

def compute_emission_probs_witten_bell(word_tag_counts, tag_counts):
    """Computes emission probabilities with Witten-Bell smoothing."""
    emission_probs = {}
    for word, tags in word_tag_counts.items():
        emission_probs[word] = {}
        for tag, count in tags.items():
            unique_words = len(word_tag_counts)
            z = unique_words - len(tags)
            emission_probs[word][tag] = count / (tag_counts[tag] + z)
    return emission_probs

from collections import Counter

from collections import Counter

def _compute_good_turing_counts(counts):
    """Computes Good-Turing adjusted counts."""
    # Convert counts to integers if they are not already
    counts = {k: int(v) for k, v in counts.items()}
    
    freq_of_freqs = Counter(counts.values())
    adjusted_counts = {}
    for count in counts.values():
        if count + 1 in freq_of_freqs:
            adjusted_counts[count] = (count + 1) * freq_of_freqs[count + 1] / freq_of_freqs[count]
        else:
            adjusted_counts[count] = count
    return adjusted_counts

def compute_emission_probs_good_turing(word_tag_counts, tag_counts):
    """Computes emission probabilities with Good-Turing smoothing."""
    emission_probs = {}
    for word, tags in word_tag_counts.items():
        emission_probs[word] = {}
        adjusted_counts = _compute_good_turing_counts(tags)
        for tag, count in tags.items():
            emission_probs[word][tag] = adjusted_counts[count] / tag_counts[tag]
    return emission_probs

def compute_emission_probs_interpolated(word_tag_counts, tag_counts, lambda1=0.7, lambda2=0.3):
    """Computes emission probabilities with interpolated smoothing."""
    emission_probs = {}
    total_words = sum(tag_counts.values())
    for word, tags in word_tag_counts.items():
        emission_probs[word] = {}
        for tag, count in tags.items():
            unigram_prob = tag_counts[tag] / total_words
            emission_probs[word][tag] = lambda1 * (count / tag_counts[tag]) + lambda2 * unigram_prob
    return emission_probs

def compute_emission_probs_base(word_tag_counts, tag_counts):
  """Computes emission probabilities P(word|tag), excluding '###' tag."""
  emission_probs = {}

  for word, tags in word_tag_counts.items():
    emission_probs[word] = {}
    for tag, count in tags.items():
      if tag != '###':  # Skip emission for sentence boundary tag
        emission_probs[word][tag] = count / tag_counts[tag]

  return emission_probs

def compute_emission_probs(word_tag_counts, tag_counts, vocabulary_size, method=None):
  """Computes emission probabilities P(word|tag), excluding '###' tag."""
  if method == SmoothingMethod.LAPLACE:
      return compute_emission_probs_laplace(word_tag_counts, tag_counts, vocabulary_size)
  elif method == SmoothingMethod.WRITTEN_BELL:
      return compute_emission_probs_witten_bell(word_tag_counts, tag_counts)
  elif method == SmoothingMethod.GOOD_TURING:
      return compute_emission_probs_good_turing(word_tag_counts, tag_counts)
  elif method == SmoothingMethod.INTERPOLATED:
      return compute_emission_probs_interpolated(word_tag_counts, tag_counts)

  return compute_emission_probs_base(word_tag_counts, tag_counts)

def compute_transition_probs_laplace(tag_tag_counts, tag_counts, num_tags):
    """Computes transition probabilities with Laplace smoothing."""
    transition_probs = {}
    for prev_tag, next_tags in tag_tag_counts.items():
        transition_probs[prev_tag] = {}
        total_transitions = sum(next_tags.values()) + num_tags  # Add num_tags for smoothing

        for next_tag in tag_counts:
            count = next_tags.get(next_tag, 0)
            transition_probs[prev_tag][next_tag] = (count + 1) / total_transitions

    return transition_probs

def compute_transition_probs_interpolated(tag_tag_counts, tag_counts, lambda_value=0.7):
    """Computes transition probabilities with interpolated smoothing."""
    transition_probs = {}
    total_tags = sum(tag_counts.values())

    for prev_tag, next_tags in tag_tag_counts.items():
        transition_probs[prev_tag] = {}
        total_transitions = sum(next_tags.values())

        for next_tag in tag_counts:
            bigram_prob = next_tags.get(next_tag, 0) / total_transitions if total_transitions > 0 else 0
            unigram_prob = tag_counts[next_tag] / total_tags
            transition_probs[prev_tag][next_tag] = lambda_value * bigram_prob + (1 - lambda_value) * unigram_prob

    return transition_probs

def compute_transition_probs_base(tag_tag_counts, tag_counts):
  """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
  transition_probs = {}

  for prev_tag, next_tags in tag_tag_counts.items():
    transition_probs[prev_tag] = {}
    total_transitions = sum(next_tags.values())

    for next_tag, count in next_tags.items():
      transition_probs[prev_tag][next_tag] = count / total_transitions

  return transition_probs


def compute_transition_probs(tag_tag_counts, tag_counts, num_tags, method=None, lambda_value=0.7):
  """Computes transition probabilities P(tag2|tag1). Includes transitions from '###'."""
  if method == SmoothingMethod.LAPLACE:
      return compute_transition_probs_laplace(tag_tag_counts, tag_counts, num_tags)
  elif method == SmoothingMethod.INTERPOLATED:
      return compute_transition_probs_interpolated(tag_tag_counts, tag_counts, lambda_value)

  return compute_transition_probs_base(tag_tag_counts, tag_counts)


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


def evaluate(true_tags, predicted_tags, train_data_parsed):
    """Evaluates the predicted tags against the true tags, including known and novel word accuracy."""
    # Extract words from training data
    train_words = set(word for word, tag in train_data_parsed)

    # Filter out sentence boundary markers
    true_tags_filtered = [tag for tag in true_tags if tag != '###']
    predicted_tags_filtered = [tag for tag in predicted_tags if tag != '###']

    # Initialize counters
    correct_tags = 0
    total_tags = 0
    correct_known_tags = 0
    total_known_tags = 0
    correct_novel_tags = 0
    total_novel_tags = 0

    # Evaluate accuracy
    for i, (predicted_tag, true_tag) in enumerate(zip(predicted_tags_filtered, true_tags_filtered)):
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
    known_word_accuracy = correct_known_tags / total_known_tags if total_known_tags > 0 else 0.0
    novel_word_accuracy = correct_novel_tags / total_novel_tags if total_novel_tags > 0 else 0.0

    return overall_accuracy, known_word_accuracy, novel_word_accuracy


def train_and_test(
    train_data_lines, 
    dev_data_lines=None, 
    test_data_lines=None, 
    n=10, 
    emission_smoothing_method=None,
    transition_smoothing_method=None,    
    debug_print=False,
    lambda_value=0.7
):
    """
    Performs training and testing using the provided data lines.
    """
    train_data_parsed = parse_data(train_data_lines)
    
    word_tag_counts, tag_counts, tag_tag_counts = compute_counts(train_data_parsed)
    tags = sorted(tag_counts.keys())
    vocabulary_size = len(set(word for word, tag in train_data_parsed))
    num_tags = len(tags)

    if debug_print:
        print(f"Number of training word/tag pairs: {len(train_data_parsed)}")
        print(f"word_tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(word_tag_counts.items())[:n])} }}")
        print(f"tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(tag_counts.items())[:n])} }}")
        print(f"tag_tag_counts: {{ {', '.join(f'{key}: {value}' for key, value in list(tag_tag_counts.items())[:n])} }}")
        print(f"tags: {tags[:n]}")
        print(f"vocabulary size: {vocabulary_size}")

    emission_probs = compute_emission_probs(word_tag_counts, tag_counts, vocabulary_size, emission_smoothing_method)
    transition_probs = compute_transition_probs(tag_tag_counts, tag_counts, num_tags, transition_smoothing_method, lambda_value)
    
    if debug_print:
        print(f"emission probabilities P(word|tag): {{ {', '.join(f'{key}: {value}' for key, value in list(emission_probs.items())[:n])} }}")
        print(f"transition probabilities P(tag2|tag1): {{ {', '.join(f'{key}: {value}' for key, value in list(transition_probs.items())[:n])} }}")

    if test_data_lines:
        test_data_parsed = parse_data(test_data_lines)        

        test_words = [word for word, tag in test_data_parsed]
        true_tags = [tag for word, tag in test_data_parsed]
        predicted_tags = viterbi_full(test_words, transition_probs, emission_probs)
        
        overall_accuracy, known_word_accuracy, novel_word_accuracy = evaluate(true_tags, predicted_tags, train_data_parsed)
        
        if debug_print:
            print(f"Number of test word/tag pairs: {len(test_data_parsed)}")
            print(f"Test words: {test_words[:n]}")
            print(f"Predicted tags: {predicted_tags[:n]}")
            print(f"Actual tags:    {true_tags[:n]}")
            print("Evaluating on test data...")
            print(f"Overall accuracy on the test data: {overall_accuracy * 100:.2f}%")
            print(f"Known word accuracy: {known_word_accuracy * 100:.2f}%")
            print(f"Novel word accuracy: {novel_word_accuracy * 100:.2f}%")
        
        return overall_accuracy, known_word_accuracy, novel_word_accuracy

    return None, None, None


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

    path = "data"

    train_file = os.path.join(path, train_base)
    test_file = os.path.join(path, test_base)

    # print the arguments to verify
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")

    train_data = read_data_file(train_file)
    test_data = read_data_file(test_file)

    if (train_base == 'ictrain' and test_base == 'ictest') or (train_base == 'entrain'):
        # emission_smoothing_methods = list(SmoothingMethod) + [None]
        emission_smoothing_methods = [SmoothingMethod.GOOD_TURING, None]
        # transition_smoothing_methods = [SmoothingMethod.LAPLACE, SmoothingMethod.INTERPOLATED, None]
        transition_smoothing_methods = [SmoothingMethod.INTERPOLATED, None]
        lambda_values = np.linspace(0.1, 0.9, 5).tolist()  # Different lambda values for interpolation
        results = []

        for emission_smoothing in emission_smoothing_methods:
            for transition_smoothing in transition_smoothing_methods:
                if transition_smoothing == SmoothingMethod.INTERPOLATED:
                    for lambda_value in lambda_values:
                        print(f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}, Lambda: {lambda_value}")
                        overall_accuracy, known_word_accuracy, novel_word_accuracy = train_and_test(
                            train_data, 
                            None, 
                            test_data, 
                            emission_smoothing_method=emission_smoothing,
                            transition_smoothing_method=transition_smoothing,
                            lambda_value=lambda_value  # Pass lambda value to the function
                        )
                        
                        # Append results for the current combination of smoothing methods
                        results.append([emission_smoothing, f"{transition_smoothing} (λ={lambda_value:.2f})", overall_accuracy, known_word_accuracy, novel_word_accuracy])
                else:
                    print(f"Emission Smoothing: {emission_smoothing}, Transition Smoothing: {transition_smoothing}")
                    overall_accuracy, known_word_accuracy, novel_word_accuracy = train_and_test(
                        train_data, 
                        None, 
                        test_data, 
                        emission_smoothing_method=emission_smoothing,
                        transition_smoothing_method=transition_smoothing
                    )
                    
                    # Append results for the current combination of smoothing methods
                    results.append([emission_smoothing, transition_smoothing, overall_accuracy, known_word_accuracy, novel_word_accuracy])
                
        # Sort results by novel word accuracy in descending order
        results.sort(key=lambda x: x[4], reverse=True)

        # Display results in a table
        headers = ["Emission Smoothing", "Transition Smoothing", "Overall Accuracy", "Known Word Accuracy", "Novel Word Accuracy"]
        print(f"\n| {headers[0]:<30} | {headers[1]:<30} | {headers[2]:<20} | {headers[3]:<20} | {headers[4]:<20}|")
        print("=" * 134)
        for row in results:
            emission_method = str(row[0]) if row[0] is not None else "None"
            transition_method = str(row[1]) if row[1] is not None else "None"
            print(f"| {emission_method:<30} | {transition_method:<30} | {row[2]:<20.5f} | {row[3]:<20.5f} | {row[4]:<20.5f}|")

        
    else:
        print("Error: Could not load training and/or testing data. Exiting.")



if __name__ == '__main__':
    main()

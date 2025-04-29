import argparse
import torch
import numpy as onp

from util import *
from model import *
from submission import greedy_decode, hamming_loss, generate_dataset_rnn, compute_normalizer, bruteforce_normalizer, viterbi_decode, bruteforce_decode, crf_loss


def evaluate(X, Y, model, decoder, loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for j in range(X.shape[0]):
            x, y = X[j:j+1, :], Y[j:j+1, :]  # (B, T)
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, y_hat = decoder(scores)
            total_loss += loss(y, y_hat)
    return total_loss / X.shape[0]

def test_unigram():
    print('testing unigram model')
    num_hiddens = 5
    model = UnigramModel(num_hiddens, vocab_size, num_labels)
    loss = torch.nn.CrossEntropyLoss()
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 10)
    error = evaluate(X_valid, Y_valid, model, greedy_decode, hamming_loss)
    print('0-1 error={}'.format(error))

def test_rnn():
    print('testing RNN model')
    num_hiddens = 5
    model = RNNModel(num_hiddens, vocab_size, num_labels)
    loss = torch.nn.CrossEntropyLoss()
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 10)
    error = evaluate(X_valid, Y_valid, model, greedy_decode, hamming_loss)
    print('0-1 error={}'.format(error))

def test_hamming_loss():
    a = torch.tensor([[2, 3, 1, 0]])
    b = torch.tensor([[1, 3, 1, 0]])
    assert hamming_loss(a, b) == 0.25

def test_score_sequence():
    unigram_scores = torch.zeros((1, 3, 3), device=device)
    unigram_scores[0, 0, :] = torch.tensor([0.1, 0.2, 0.7], device=device)
    unigram_scores[0, 1, :] = torch.tensor([0.5, 0.2, 0.3], device=device)
    unigram_scores[0, 2, :] = torch.tensor([0.3, 0.2, 0.5], device=device)
    
    bigram_scores_base = torch.tensor([
        [0.5, 0.1, 0.3],
        [0.1, 0.4, 0.2],
        [0.2, 0.2, 0.3]
    ], device=device)
    
    bigram_scores = bigram_scores_base.expand(1, 3, 3, 3)
    seqs = torch.tensor([[0, 1, 2]], device=device)
    # 0.1 + 0.2 + 0.5 + 0.1 + 0.2 = 1.1
    score = score_sequence(seqs, unigram_scores, bigram_scores)
    assert torch.allclose(score, torch.tensor([[1.1]], device=device))

def test_viterbi():
    print('testing viterbi decoding')
    for _ in range(3):
        unigram_scores = torch.rand((1, 3, 3), device=device)
        bigram_scores = torch.rand((1, 3, 3, 3), device=device)
        score, y_hat = viterbi_decode((unigram_scores, bigram_scores))
        score_brute, y_brute = bruteforce_decode(unigram_scores, bigram_scores)
        # Note that the paths found may not be the same
        assert torch.allclose(score, score_brute)

def test_normalizer():
    print('testing normalizer computation')
    for _ in range(3):
        unigram_scores = torch.rand((1, 3, 3), device=device)
        bigram_scores = torch.rand((1, 3, 3, 3), device=device)
        score = compute_normalizer(unigram_scores, bigram_scores)
        score_brute = bruteforce_normalizer(unigram_scores, bigram_scores)
        assert torch.allclose(score, score_brute)

def test_crfrnn():
    print('testing CRFRNN model')
    num_hiddens = 5
    model = CRFRNNModel(num_hiddens, vocab_size, num_labels)
    loss = crf_loss
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 5)
    error = evaluate(X_valid, Y_valid, model, viterbi_decode, hamming_loss)
    print('0-1 error={}'.format(error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test', help='test name')
    parser.add_argument('--data', help='dataset name', default='identity')
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    else:
        device = torch.device('cpu')

    vocab_size = 5
    N = 100
    length = 10
    num_labels = 4
    if args.data == 'identity':
        generate_dataset = generate_dataset_identity
        num_labels = vocab_size
    elif args.data == 'rnn':
        generate_dataset = generate_dataset_rnn
    elif args.data == 'hmm':
        generate_dataset = generate_dataset_hmm
    else:
        raise ValueError
    X_train, Y_train = generate_dataset(vocab_size, num_labels, length, N)
    X_valid, Y_valid = generate_dataset(vocab_size, num_labels, length, N)

    if args.test == 'unigram':
        test_unigram()
    elif args.test == 'rnn':
        test_rnn()
    elif args.test == 'normalizer':
        test_normalizer()
    elif args.test == 'viterbi':
        test_viterbi()
    elif args.test == 'crfrnn':
        test_crfrnn()
    else:
        raise ValueError
from util import *
from model import *
import numpy as onp
import itertools
import torch

def hamming_loss(gold_seqs, pred_seqs):
    """Return the average hamming loss of pred_seqs.
    Useful function: torch.eq(x1, x2)
    Parameters:
        gold_seqs : (batch_size, seq_len)
        pred_seqs : (batch_size, seq_len)
    Return:
        loss : float
    """
    return 1. - torch.eq(gold_seqs, pred_seqs).sum().float() / gold_seqs.shape[0] / gold_seqs.shape[1]

def greedy_decode(scores):
    """Decode sequence for independent classification models.
    Parameters:
        scores : (batch_size, seq_len, vocab_size)
    Returns:
        labels : (batch_size, seq_len)
    """
    return torch.max(scores, dim=-1)

def score_sequence(seqs, unigram_scores, bigram_scores):
    """Compute score of the sequence:
        \sum_{t} unigram_score(s[t]) + bigram_score(s[t-1], s[t])
    Parameters:
        seqs : (batch_size, seq_len)
        unigram_scores : (batch_size, seq_len, num_labels)
            score of unigrams s[t]
        bigram_scores : (batch_size, seq_len, num_labels, num_labels)
            score of bigrams s[t], s[t-1].
            Note: coordinate 2 corresponds to s[t] and coordinate 3 correspond to s[t-1]
    Returns:
        scores : (batch_size,)
    """
    batch_size, seq_len = seqs.shape
    batch_indices = torch.arange(batch_size, device=seqs.device)
    prev_scores = unigram_scores[batch_indices, 0, seqs[:, 0]]
    for i in range(1, seqs.shape[1]):
        prev_scores = prev_scores + unigram_scores[batch_indices, i, seqs[:, i]] + bigram_scores[batch_indices, i, seqs[:, i], seqs[:, i - 1]]
    return prev_scores

def bruteforce_decode(unigram_scores, bigram_scores):
    batch_size, seq_len, num_labels = unigram_scores.shape
    device = unigram_scores.device
    seq_scores = None  # (num_seqs, batch_size)
    seqs = None  # (num_seqs, seq_len)
    for seq in itertools.product(range(num_labels), repeat=seq_len):
        seq_tensor = torch.tensor(seq, device=device)
        score = score_sequence(
            seq_tensor.expand(batch_size, seq_len),
            unigram_scores, 
            bigram_scores
        )
        
        if seq_scores is None:
            seq_scores = score.unsqueeze(0)
            seqs = seq_tensor.unsqueeze(0)
        else:
            seq_scores = torch.vstack((seq_scores, score.unsqueeze(0)))
            seqs = torch.vstack((seqs, seq_tensor.unsqueeze(0)))
    
    max_scores, max_indices = torch.max(seq_scores, dim=0)
    best_seqs = seqs[max_indices]
    
    return max_scores, best_seqs

def viterbi_decode(scores):
    """Implement Viterbi decoding.
    Your result should match what returned by bruteforce_decode.
    Parameters:
        scores : (unigram_scores, bigram_scores)
            unigram_scores : (batch_size, seq_len, num_labels)
                batch_id, time_step, curr_symbol
                score of current symbol
            bigram_scores : (batch_size, seq_len, num_labels, num_labels)
                batch_id, time_step, curr_symbol, prev_symbol
                score of previous symbol followed by current symbol
    Returns:
        scores : (batch_size,)
        labels : (batch_size, seq_len,)
    """
    unigram_scores, bigram_scores = scores
    # BEGIN_YOUR_CODE
    raise Exception
    # END_YOUR_CODE

def bruteforce_normalizer(unigram_scores, bigram_scores):
    batch_size, seq_len, num_labels = unigram_scores.shape
    device = unigram_scores.device
    
    seq_scores = None  # (num_seqs, batch_size)
    
    for seq in itertools.product(range(num_labels), repeat=seq_len):
        seq_tensor = torch.tensor(seq, device=device)
        score = score_sequence(
            seq_tensor.expand(batch_size, seq_len),
            unigram_scores, 
            bigram_scores
        )
        
        if seq_scores is None:
            seq_scores = score.unsqueeze(0)
        else:
            seq_scores = torch.vstack((seq_scores, score.unsqueeze(0)))
    
    a = logsumexp(seq_scores.T)
    return a

def compute_normalizer(unigram_scores, bigram_scores):
    """Compute the normalizer (partition function) in CRF's loss function.
    Your result should match what returned by bruteforce_normalizer.
    Parameters:
        scores : (unigram_scores, bigram_scores)
            unigram_scores : (batch_size, seq_len, num_labels)
                batch_id, time_step, curr_symbol
                score of current symbol
            bigram_scores : (batch_size, seq_len, num_labels, num_labels)
                batch_id, time_step, curr_symbol, prev_symbol
                score of previous symbol followed by current symbol
    Returns:
        normalizer : (batch_size,)
    """
    # BEGIN_YOUR_CODE
    raise Exception
    # END_YOUR_CODE

def crf_loss(scores, y):
    """Compute the loss for the CRF model.
    You can use score_sequence and compute_normalizer.
    Parameters:
        scores : (unigram_scores, bigram_scores)
        y : (batch_size, seq_len)
            gold sequence
    """
    unigram_scores, bigram_scores = scores
    gold_seq_score = score_sequence(y, unigram_scores, bigram_scores)
    normalizer = compute_normalizer(unigram_scores, bigram_scores)
    loss = normalizer - gold_seq_score
    return loss
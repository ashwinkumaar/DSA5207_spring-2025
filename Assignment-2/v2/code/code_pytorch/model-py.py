import torch
import torch.nn as nn
import torch.nn.functional as F

class UnigramModel(nn.Module):
    """Label the sequence by classifying each input symbol.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(UnigramModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dense1 = nn.Linear(vocab_size, num_hiddens)
        self.dense2 = nn.Linear(num_hiddens, num_labels)

    def forward(self, inputs):
        X = F.one_hot(inputs.long(), self.vocab_size).float()  # (B, T, H)
        Y = self.dense1(X)  # (B, T, H)
        output = self.dense2(Y)  # (B, T, H)
        return output

class RNNModel(nn.Module):
    """Label the sequence by independent prediction at each time step
    using all input context.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = nn.LSTM(vocab_size, num_hiddens, bidirectional=True, batch_first=True)
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.dense = nn.Linear(num_hiddens * 2, num_labels)  # *2 for bidirectional

    def forward(self, inputs):
        """
        Parameters:
            inputs : (batch_size, seq_lens, num_hidden_units)
            state : (batch_size, num_hidden_units)
                initial state of RNN
        Returns:
            output : (seq_lens, batch_size, num_labels)
                predicted scores for labels at each time step
        """
        # Set initial state to zero
        init_state = self.begin_state(inputs.shape[0])
        # One hot representation of input symbols
        X = F.one_hot(inputs.long(), self.vocab_size).float()  # (B, T, H)
        Y, state = self.rnn(X, init_state)
        output = self.dense(Y)  # (B, T, H)
        return output

    def begin_state(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(2, batch_size, self.num_hiddens, device=device)  # 2 for bidirectional
        c0 = torch.zeros(2, batch_size, self.num_hiddens, device=device)
        return (h0, c0)

class CRFRNNModel(RNNModel):
    """Add a CRF layer on top of the RNN model.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(CRFRNNModel, self).__init__(num_hiddens, vocab_size, num_labels, **kwargs)
        self.bigram_scores = nn.Parameter(torch.randn(num_labels, num_labels))
        self.num_labels = num_labels

    def forward(self, inputs):
        unigram_scores = super(CRFRNNModel, self).forward(inputs)  # RNN outputs
        batch_size, seq_len, vocab_size = unigram_scores.shape
        bigram_scores = self.bigram_scores.expand(batch_size, seq_len, self.num_labels, self.num_labels)
        return unigram_scores, bigram_scores
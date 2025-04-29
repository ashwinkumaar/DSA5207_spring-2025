import torch
import numpy as onp

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(42)
else:
    device = torch.device('cpu')

from model import *

def logsumexp(scores, dim=-1, keepdim=False):
    """
    Parameters:
        scores : (..., vocab_size)
    Returns:
        normalizer : (..., 1)
            same dimension as scores
    """
    m = 0
    return m + torch.log(torch.sum(torch.exp(scores - m), dim=dim, keepdim=keepdim))

def generate_dataset_identity(vocab_size, num_labels, length, size):
    """Generate a simple dataset where the output equals the input,
    e.g. [1,2,3] -> [1,2,3]
    """
    assert vocab_size == num_labels
    X = torch.randint(1, vocab_size, (size, length), device=device)
    Y = X.clone()
    # Set first symbol to START (0)
    X[:, 0] = 0
    Y[:, 0] = 0
    return X, Y

def generate_dataset_rnn(vocab_size, num_labels, length, size):
    """Generate a dataset where the RNNModel achieves much lower loss than the UnigramModel.
    Parameters:
        vocab_size : int
            size of the output space for each input symbol
        num_labels: int
            size of the label set
        length : int
            the input sequence length
        size : int
            number of examples to generate
    Returns:
        X : torch.Tensor (size, length)
        Y : torch.Tensor (size, length)
            each element in Y must be an integer in [0, num_labels).
    """
    X = torch.randint(1, vocab_size, (size, length), device=device)
    Y = torch.zeros_like(X)
    Y[:, :-1] = torch.fmod(X[:, 1:], num_labels - 1) + 1
    X[:, 0] = 0
    Y[:, 0] = 0
    return X, Y

def generate_dataset_hmm(vocab_size, num_labels, length, size):
    num_states = num_labels - 1  # don't count START
    generate_multinomial = lambda num_outcomes: onp.random.dirichlet(onp.random.randint(1, 10, num_outcomes))
    
    start_prob = torch.tensor([[0.3, 0.3, 0.4]], device=device)
    trans_prob = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.6, 0.1, 0.3],
        [0.1, 0.6, 0.4]
    ], device=device)
    
    emiss_prob = torch.tensor([
        [0.6, 0.3, 0.1, 0.0],
        [0.3, 0.6, 0.0, 0.1],
        [0.1, 0.0, 0.5, 0.4],
        [0.1, 0.1, 0.1, 0.6],
    ], device=device)
    
    X, Y = _sample_from_hmm(start_prob, trans_prob, emiss_prob, length - 1, size)
    # Offset START
    X = X + 1
    Y = Y + 1
    # Add START
    X = torch.cat([torch.zeros((size, 1), dtype=torch.long, device=device), X], dim=1)
    Y = torch.cat([torch.zeros((size, 1), dtype=torch.long, device=device), Y], dim=1)
    return X, Y

def _sample_from_hmm(start_prob, trans_prob, emiss_prob, length, size):
    """Generate sequences of fixed length according to the start state probability
    and the transition matrix.
    Parameters:
        start_prob : start_prob[i] = p(state=i | START)
        trans_prob : trans_prob[i][j] = p(state=j | prev_state=i)
        emiss_prob : emiss_prob[i][j] = p(data=j | state=i)
    Returns:
        obs : (size, length)
        state : (size, length)
    """
    def _sample(state, cdf):
        cdf_ = cdf[state, :]
        a = torch.rand((state.shape[0], 1), device=device)
        outcome = torch.argmax((cdf_ > a).int(), dim=-1)
        return outcome

    data = torch.zeros((size, length), dtype=torch.long, device=device)
    states = torch.zeros((size, length), dtype=torch.long, device=device)
    
    start_prob_cumsum = torch.cumsum(start_prob, dim=1)
    trans_prob_cumsum = torch.cumsum(trans_prob, dim=1)
    emiss_prob_cumsum = torch.cumsum(emiss_prob, dim=1)
    
    states[:, 0] = _sample(torch.zeros(size, dtype=torch.long, device=device), start_prob_cumsum)
    
    for t in range(length):
        curr_state = states[:, t]
        data[:, t] = _sample(curr_state, emiss_prob_cumsum)
        if t + 1 < length:
            states[:, t+1] = _sample(curr_state, trans_prob_cumsum)
            
    return data, states

def compute_loss(X, Y, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for j in range(X.shape[0]):
            x, y = X[j:j+1, :], Y[j:j+1, :]  # (B, T)
            x, y = x.to(device), y.to(device)
            scores = model(x)  # (B, T, vocab_size)
            
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                # Reshape for CrossEntropyLoss
                scores = scores.view(-1, scores.size(-1))
                y = y.view(-1)
                l = loss_fn(scores, y.long())
            else:
                l = loss_fn(scores, y).mean()
                
            total_loss += l.item()
    return total_loss / X.shape[0]

def train(X_train, Y_train, X_valid, Y_valid, model, loss_fn, learning_rate, num_epochs):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    batch_size = 1
    step = 0
    for i in range(num_epochs):
        model.train()
        for j in range(X_train.shape[0]):
            x, y = X_train[j:j+1, :], Y_train[j:j+1, :]  # (B, T)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            scores = model(x)  # (B, T, vocab_size)
            
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                # Reshape for CrossEntropyLoss
                scores_reshaped = scores.view(-1, scores.size(-1))
                y_reshaped = y.view(-1)
                l = loss_fn(scores_reshaped, y_reshaped.long())
            else:
                l = loss_fn(scores, y).mean()
                
            l.backward()
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                valid_loss = compute_loss(X_valid, Y_valid, model, loss_fn)
                print('step={}, curr_loss={}, valid_loss={}'.format(step, l.item(), valid_loss))
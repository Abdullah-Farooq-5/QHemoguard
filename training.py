import numpy as np
import pickle
import os
from qiskit_aer import AerSimulator
from sklearn.metrics import (precision_score, recall_score, f1_score,
                               balanced_accuracy_score, roc_auc_score,
                               average_precision_score)

_backend = AerSimulator()

def qnn_output(qc_transpiled, x_vec, omega_vec, shots=200):
    param_values = np.hstack((omega_vec, x_vec))
    parameter_dict = {p: v for p, v in zip(qc_transpiled.parameters, param_values)}
    bound_qc = qc_transpiled.assign_parameters(parameter_dict)
    result = _backend.run(bound_qc, shots=shots).result()
    counts_bit = result.get_counts()
    total = sum(counts_bit.values())
    ones = sum(v for k, v in counts_bit.items() if k.strip().endswith('1'))
    return ones / total

def transpile_circuit(qc):
    from qiskit import transpile
    return transpile(qc, _backend)

def predict_probs(qc, X, w_vec, shots=300):
    return np.array([qnn_output(qc, x, w_vec, shots=shots) for x in X])

def cost_function(p1, d_class, class_weight=None):
    w = class_weight[int(d_class)] if class_weight is not None else 1.0
    return 0.5 * w * (d_class - p1) ** 2

def spsa_gradient(qc, x_vec, omega_vec, d_class, shots, c_k, class_weight=None):
    delta = np.random.choice([-1, 1], size=len(omega_vec))
    p_plus  = qnn_output(qc, x_vec, omega_vec + c_k*delta, shots=shots)
    p_minus = qnn_output(qc, x_vec, omega_vec - c_k*delta, shots=shots)
    c_plus  = cost_function(p_plus,  d_class, class_weight)
    c_minus = cost_function(p_minus, d_class, class_weight)
    return (c_plus - c_minus) / (2 * c_k * delta)

def numerical_gradient(qc, x_vec, omega_vec, d_class, shots=2000, delta=0.1, class_weight=None):
    grad = np.zeros(len(omega_vec))
    for j in range(len(omega_vec)):
        op, om = omega_vec.copy(), omega_vec.copy()
        op[j] += delta; om[j] -= delta
        cp = cost_function(qnn_output(qc, x_vec, op, shots=shots), d_class, class_weight)
        cm = cost_function(qnn_output(qc, x_vec, om, shots=shots), d_class, class_weight)
        grad[j] = (cp - cm) / (2 * delta)
    return grad

def get_epoch_batch(X, y, epoch_seed, mode='balanced'):
    rng = np.random.RandomState(epoch_seed)
    if mode == 'all':
        idx = rng.permutation(len(y))
    else:
        i1 = np.where(y == 1)[0]
        i0 = rng.choice(np.where(y == 0)[0], size=len(i1), replace=False)
        idx = rng.permutation(np.concatenate([i1, i0]))
    return X[idx], y[idx]

def train(qc, X_train, y_train, num_weights, epochs=10, lr=0.2, shots=300,
          batch_mode='balanced', class_weight=None, seed=0, verbose=True,
          gradient_method='spsa', c=0.1, alpha=0.602, gamma=0.101,
          checkpoint_path=None, resume=False):
    w_vec = 2*np.pi*np.random.RandomState(seed).random_sample(num_weights)
    cost_history, error_history = [], []
    start_epoch, global_step = 0, 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ck = pickle.load(f)
        w_vec, cost_history, error_history = ck['w_vec'], ck['cost_history'], ck['error_history']
        start_epoch, global_step = ck['epoch'], ck['global_step']
        print(f"Resuming from epoch {start_epoch}")

    A = 0.1 * epochs * len(y_train)

    for k in range(start_epoch, epochs):
        X_ep, y_ep = get_epoch_batch(X_train, y_train, seed*1000 + k, batch_mode)
        s = len(y_ep); total_cost = 0; num_wrong = 0
        for i in range(s):
            x_vec, d = X_ep[i], y_ep[i]
            p1 = qnn_output(qc, x_vec, w_vec, shots=shots)
            total_cost += cost_function(p1, d, class_weight) / s
            num_wrong += (round(p1) != d)
            a_k = lr / (global_step + 1 + A) ** alpha
            c_k = c / (global_step + 1) ** gamma
            if gradient_method == 'spsa':
                grad = spsa_gradient(qc, x_vec, w_vec, d, shots, c_k, class_weight)
            else:
                grad = numerical_gradient(qc, x_vec, w_vec, d, shots=shots, class_weight=class_weight)
            w_vec = w_vec - a_k * grad
            global_step += 1
            if verbose:
                print(f"  epoch {k+1} | {i+1}/{s}", end='\r')
        cost_history.append(total_cost); error_history.append(100*num_wrong/s)
        print(f"Epoch {k+1:2d} | cost = {total_cost:.4f} | error = {error_history[-1]:.1f}%" + " "*20)
        if checkpoint_path:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dict(w_vec=w_vec, cost_history=cost_history,
                                 error_history=error_history, epoch=k+1,
                                 global_step=global_step), f)
    return w_vec, cost_history, error_history

def evaluate_full(qc, X_test, y_test, w_vec, shots=500):
    y_prob = predict_probs(qc, X_test, w_vec, shots=shots)
    y_pred = np.round(y_prob)
    return {
        'accuracy': float(np.mean(y_pred == y_test)),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
    }
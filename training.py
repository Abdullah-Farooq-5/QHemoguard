import numpy as np
from qiskit_aer import AerSimulator
from sklearn.metrics import (precision_score, recall_score, f1_score,
                               balanced_accuracy_score, roc_auc_score,
                               average_precision_score)

_backend = AerSimulator()

def qnn_output(qc_transpiled, x_vec, omega_vec, shots=200):
    """qc_transpiled MUST already be transpiled before calling this — see transpile_circuit()."""
    param_values = np.hstack((omega_vec, x_vec))
    parameter_dict = {p: v for p, v in zip(qc_transpiled.parameters, param_values)}
    bound_qc = qc_transpiled.assign_parameters(parameter_dict)

    result = _backend.run(bound_qc, shots=shots).result()
    counts_bit = result.get_counts()
    total = sum(counts_bit.values())
    ones = sum(v for k, v in counts_bit.items() if k.strip().endswith('1'))
    return ones / total

def transpile_circuit(qc):
    """Call this ONCE per circuit, before training. Do not call transpile() anywhere else."""
    from qiskit import transpile
    return transpile(qc, _backend)

def cost_function(p1, d_class):
    return 0.5 * (d_class - p1)**2

def numerical_gradient(qc, x_vec, omega_vec, d_class, shots=2000, delta=0.1):
    num_weights = len(omega_vec)
    grad = np.zeros(num_weights)
    for j in range(num_weights):
        omega_plus = omega_vec.copy()
        omega_minus = omega_vec.copy()
        omega_plus[j] += delta
        omega_minus[j] -= delta
        p1_plus = qnn_output(qc, x_vec, omega_plus, shots=shots)
        p1_minus = qnn_output(qc, x_vec, omega_minus, shots=shots)
        cost_plus = cost_function(p1_plus, d_class)
        cost_minus = cost_function(p1_minus, d_class)
        grad[j] = (cost_plus - cost_minus) / (2 * delta)
    return grad

def spsa_gradient(qc, x_vec, omega_vec, d_class, shots=300, c_k=0.1):
    num_weights = len(omega_vec)
    delta = np.random.choice([-1, 1], size=num_weights)

    omega_plus = omega_vec + c_k * delta
    omega_minus = omega_vec - c_k * delta

    p1_plus = qnn_output(qc, x_vec, omega_plus, shots=shots)
    p1_minus = qnn_output(qc, x_vec, omega_minus, shots=shots)

    cost_plus = cost_function(p1_plus, d_class)
    cost_minus = cost_function(p1_minus, d_class)

    grad = (cost_plus - cost_minus) / (2 * c_k * delta)
    return grad

def train(qc, X_train, y_train, num_weights, epochs=20, lr=1.0, shots=2000,
          w_vec=None, verbose=True, gradient_method='spsa',
          a=None, c=0.1, A=None, alpha=0.602, gamma=0.101):
    """
    a, c: SPSA hyperparameters (standard Spall defaults for alpha/gamma).
    a defaults to lr if not given. A (stability constant) defaults to 10% of total epochs.
    """
    if w_vec is None:
        w_vec = 2*np.pi*np.random.random(num_weights)
    if a is None:
        a = lr
    if A is None:
        A = 0.1 * epochs

    s = len(X_train)
    cost_history = []
    error_history = []
    global_step = 0

    for k in range(epochs):
        total_cost = 0
        num_wrong = 0
        for i in range(s):
            x_vec = X_train[i]
            d_class = y_train[i]

            p1 = qnn_output(qc, x_vec, w_vec, shots=shots)
            cost = cost_function(p1, d_class)
            total_cost += cost / s
            if round(p1) != d_class:
                num_wrong += 1

            # Decaying step sizes — this is the actual SPSA algorithm, not a fixed-step approximation
            a_k = a / (global_step + 1 + A) ** alpha
            c_k = c / (global_step + 1) ** gamma

            if gradient_method == 'spsa':
                grad = spsa_gradient(qc, x_vec, w_vec, d_class, shots=shots, c_k=c_k)
                w_vec = w_vec - a_k * grad
            else:
                grad = numerical_gradient(qc, x_vec, w_vec, d_class, shots=shots)
                w_vec = w_vec - lr * grad

            global_step += 1

            if verbose:
                print(f"  epoch {k+1} | sample {i+1}/{s} | p1={p1:.3f} | true={d_class}", end='\r')

        cost_history.append(total_cost)
        error_pct = 100 * num_wrong / s
        error_history.append(error_pct)
        print(f"Epoch {k+1:2d} | cost = {total_cost:.4f} | error = {error_pct:.1f}%" + " "*20)

    return w_vec, cost_history, error_history

def evaluate(qc, X_test, y_test, w_vec, shots=5000):
    correct = 0
    for i in range(len(X_test)):
        p1 = qnn_output(qc, X_test[i], w_vec, shots=shots)
        if round(p1) == y_test[i]:
            correct += 1
    return correct / len(X_test)

def evaluate_full(qc, X_test, y_test, w_vec, shots=500):
    y_pred = []
    y_prob = []
    for i in range(len(X_test)):
        p1 = qnn_output(qc, X_test[i], w_vec, shots=shots)
        y_prob.append(p1)
        y_pred.append(round(p1))
    return {
        'accuracy': sum(np.array(y_pred) == y_test) / len(y_test),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
    }
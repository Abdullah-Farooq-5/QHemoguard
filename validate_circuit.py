import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from qiskit.quantum_info import Statevector
from fast_train import FULL, CBC, LABEL, forward
from pqc import build_full_pqc

DATA, EXP, JSON = sys.argv[1], sys.argv[2], sys.argv[3]      # csv, full|cbc, results json
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
feats = FULL if EXP == 'full' else CBC

df = pd.read_csv(DATA); df.columns = df.columns.str.strip()
X = df[feats].values.astype(float); y = df[LABEL].values.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)
lo, hi = X_tr.min(0), X_tr.max(0)                    # same train-only scaling as fast_train.py
X_te = 2 * np.pi * np.clip((X_te - lo) / (hi - lo), 0, 1)

w = np.array(json.load(open(JSON))[0]['weights'])    # seed 1's trained weights
qc, xp, wp = build_full_pqc(5)
qc = qc.remove_final_measurements(inplace=False)     # exact statevector, no shot noise

fast, circ = [], []
for x in X_te[:N]:
    fast.append(forward(x[None], w)[0][0])
    bind = {xp[i]: x[i] for i in range(5)}
    bind.update({wp[j]: w[j] for j in range(10)})
    circ.append(Statevector(qc.assign_parameters(bind)).probabilities([0])[1])
fast, circ = np.array(fast), np.array(circ)
print(f"test patients checked        : {N}")
print(f"max |fast - Qiskit circuit|  : {np.abs(fast - circ).max():.2e}")
print(f"same prediction at thr 0.5   : {np.mean((fast >= .5) == (circ >= .5)) * 100:.1f}%")

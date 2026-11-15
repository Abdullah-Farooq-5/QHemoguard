"""
Fast exact simulation of the Hemoguard Hadamard-test QNN.
Output probability of the circuit (verified against Qiskit statevector to ~1e-13):
    P(1) = 0.5 * (1 - Re prod_i 0.5*(exp(i*x_i*w[2i]) + exp(i*x_i*w[2i+1])))
Everything is NumPy; runs on any CPU in seconds-minutes.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, accuracy_score, average_precision_score)

FULL = ['Hemoglobin A2 (%)', 'Hemoglobin A (%)', 'RBC Count (mil/cmm)', 'MCV (fL)', 'MCH (pg)']
CBC  = ['RBC Count (mil/cmm)', 'Hemoglobin Level (g/dL)', 'Hematocrit (%)', 'MCV (fL)', 'MCH (pg)']
LABEL = 'Hematologist Remarks'

def forward(X, w):
    n = X.shape[1]
    E = np.exp(1j * X[:, :, None] * w.reshape(n, 2)[None])      # (B,n,2)
    F = 0.5 * (E[:, :, 0] + E[:, :, 1])                          # (B,n)
    return 0.5 * (1 - np.prod(F, axis=1).real), F, E

def loss_grad(X, y, w, cw):
    p, F, E = forward(X, w)
    n = X.shape[1]
    wc = np.where(y == 1, cw[1], cw[0])
    loss = np.mean(0.5 * wc * (y - p) ** 2)
    others = np.stack([np.prod(np.delete(F, i, axis=1), axis=1) for i in range(n)], axis=1)
    dP = -0.5 * np.real(others[:, :, None] * 0.5j * X[:, :, None] * E)   # dP1/dw, (B,n,2)
    g = np.mean((wc * (p - y))[:, None, None] * dP, axis=0)
    return loss, g.reshape(-1)

def epoch_batch(X, y, rng, mode):
    if mode == 'all':
        idx = rng.permutation(len(y))
    else:   # all carriers + fresh equal-size draw of normals each epoch
        i1 = np.where(y == 1)[0]
        i0 = rng.choice(np.where(y == 0)[0], size=len(i1), replace=False)
        idx = rng.permutation(np.concatenate([i1, i0]))
    return X[idx], y[idx]

def predict(X, w, shots, rng):
    p = forward(X, w)[0]
    return p if shots <= 0 else rng.binomial(shots, np.clip(p, 0, 1)) / shots   # mimics shot noise

def train(Xtr, ytr, Xva, yva, seed, epochs, lr, batch, init_std, mode):
    rng = np.random.RandomState(seed)
    n = Xtr.shape[1]
    w = rng.normal(0, init_std, 2 * n)
    cw = {0: 1.0, 1: (ytr == 0).sum() / (ytr == 1).sum()} if mode == 'all' else {0: 1.0, 1: 1.0}
    m, v, t = np.zeros_like(w), np.zeros_like(w), 0
    best_auc, best_w, hist = -1, w.copy(), []
    for ep in range(epochs):
        Xe, ye = epoch_batch(Xtr, ytr, rng, mode)
        for s in range(0, len(ye), batch):
            _, g = loss_grad(Xe[s:s+batch], ye[s:s+batch], w, cw)
            t += 1
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
            w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        auc = roc_auc_score(yva, forward(Xva, w)[0])
        hist.append(float(auc))
        if auc > best_auc: best_auc, best_w = auc, w.copy()      # best epoch chosen on VALIDATION only
    return best_w, hist

def best_threshold(y, p, metric):
    ts = np.arange(0.02, 0.99, 0.01)
    return float(ts[int(np.argmax([metric(y, (p >= t).astype(int)) for t in ts]))])

def summarize(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(threshold=thr, accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)))

def run(df, feats, a):
    X = df[feats].values.astype(float); y = df[LABEL].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)
    lo, hi = X_tr.min(0), X_tr.max(0)                       # scaler fitted on TRAIN only -> no leakage
    sc = lambda Z: 2 * np.pi * np.clip((Z - lo) / (hi - lo), 0, 1)
    X_tr, X_va, X_te = sc(X_tr), sc(X_va), sc(X_te)
    rows = []
    for seed in range(1, a.seeds + 1):
        t0 = time.time()
        w, hist = train(X_tr, y_tr, X_va, y_va, seed, a.epochs, a.lr, a.batch, a.init_std, a.mode)
        rng = np.random.RandomState(1000 + seed)
        pv, pt = predict(X_va, w, a.eval_shots, rng), predict(X_te, w, a.eval_shots, rng)
        t_bal = best_threshold(y_va, pv, balanced_accuracy_score)
        t_acc = best_threshold(y_va, pv, accuracy_score)
        r = dict(seed=seed, roc_auc=float(roc_auc_score(y_te, pt)), pr_auc=float(average_precision_score(y_te, pt)),
                 at_0p5=summarize(y_te, pt, 0.5), tuned_balacc=summarize(y_te, pt, t_bal),
                 tuned_acc=summarize(y_te, pt, t_acc), seconds=round(time.time() - t0, 1), weights=[float(x) for x in w])
        rows.append(r)
        print(f"  seed {seed}: AUC {r['roc_auc']:.3f} | acc@tuned-acc {r['tuned_acc']['accuracy']:.3f} | "
              f"balacc@tuned {r['tuned_balacc']['balanced_accuracy']:.3f} | {r['seconds']}s")
    return rows

def agg(rows, f):
    v = np.array([f(r) for r in rows])
    return f"{v.mean():.3f} +/- {v.std():.3f}  (min {v.min():.3f}, max {v.max():.3f})"

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--exp', choices=['full', 'cbc', 'both'], default='both')
    ap.add_argument('--mode', choices=['balanced', 'all'], default='balanced')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--init_std', type=float, default=0.5)
    ap.add_argument('--eval_shots', type=int, default=0, help='0 = exact probabilities, else binomial shot noise')
    ap.add_argument('--out', default='results_fast')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data); df.columns = df.columns.str.strip()
    for name, feats in [('full', FULL), ('cbc', CBC)]:
        if a.exp not in (name, 'both'): continue
        print(f"\n=== {name} | mode={a.mode} | init_std={a.init_std} ===")
        rows = run(df, feats, a)
        json.dump(rows, open(f"{a.out}/{name}_{a.mode}.json", 'w'), indent=1)
        print("  ROC-AUC            " + agg(rows, lambda r: r['roc_auc']))
        print("  Accuracy (tuned)   " + agg(rows, lambda r: r['tuned_acc']['accuracy']))
        print("  Balanced acc       " + agg(rows, lambda r: r['tuned_balacc']['balanced_accuracy']))
        print("  F1 (bal-tuned)     " + agg(rows, lambda r: r['tuned_balacc']['f1']))

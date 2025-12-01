import os
import numpy as np
from tqdm import tqdm
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ================== config ==================

DATA_DIR = r"F:\dataset\sleep-edf\preprocessed"

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

N_FOLDS = 5  # how many folds for cross-validation


# ================== feature extraction ==================

def extract_features_epoch(epoch_signal, fs=100.0):
    """
    epoch_signal: shape (T,) where T ~ 3000 samples (30s @100Hz)
    returns: 1D feature vector (float list)
    We'll compute:
    - time domain stats
    - band power ratios from PSD
    """
    x = epoch_signal

    # --- time-domain features ---
    mean_val = np.mean(x)
    var_val = np.var(x)
    skew_val = np.mean(((x - mean_val) / (np.std(x) + 1e-8)) ** 3)
    kurt_val = np.mean(((x - mean_val) / (np.std(x) + 1e-8)) ** 4)
    zcr = np.mean(np.abs(np.diff(np.sign(x))))  # zero crossing rate approx

    # --- frequency-domain features via Welch PSD ---
    # welch returns power spectral density estimate
    freqs, psd = welch(x, fs=fs, nperseg=256)

    # total power in [0.5, 30] to normalize subbands
    total_mask = (freqs >= 0.5) & (freqs <= 30)
    total_power = np.trapz(psd[total_mask], freqs[total_mask]) + 1e-10

    band_powers = []
    for (lo, hi) in BANDS.values():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = np.trapz(psd[mask], freqs[mask])  # band absolute power
        band_powers.append(bp / total_power)  # relative power

    # final feature vector
    feats = [
        mean_val,
        var_val,
        skew_val,
        kurt_val,
        zcr,
    ] + band_powers  # [delta_rel, theta_rel, alpha_rel, beta_rel]

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(X_raw, fs=100.0):
    """
    X_raw: shape (N, T)
    return: X_feat shape (N, D)
    """
    feats = []
    for i in tqdm(range(len(X_raw)), desc="Extracting features"):
        feats.append(extract_features_epoch(X_raw[i], fs=fs))
    feats = np.stack(feats, axis=0)
    return feats


# ================== main training/eval ==================

def main():
    # ----- load data -----
    X_raw = np.load(os.path.join(DATA_DIR, "X_raw.npy"))   # (N, T)
    y     = np.load(os.path.join(DATA_DIR, "y.npy"))       # (N,)
    groups= np.load(os.path.join(DATA_DIR, "groups.npy"))  # (N,)

    print("Loaded:")
    print("X_raw:", X_raw.shape)
    print("y    :", y.shape)
    print("groups:", groups.shape)

    # ----- feature extraction -----
    # Sleep-EDF Cassette EEG is ~100 Hz, 30s => 3000 samples
    fs = 100.0
    X_feat = build_feature_matrix(X_raw, fs=fs)
    print("Feature matrix:", X_feat.shape)  # (N, D)

    # ----- cross-validation (group-wise) -----
    # We want subject-separated folds.
    # groups[] currently holds something like "SC4001E0-PSG" per epoch.
    # To avoid leakage between different nights of same subject, we can
    # simplify group ID to the first 6 chars "SC4001".
    simple_groups = np.array([g[:6] for g in groups])

    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X_feat, y, groups=simple_groups), start=1
    ):
        X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # ----- train SVM baseline -----
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma=1e-2,
            class_weight="balanced"  # handle class imbalance (N1 is rare)
        )
        clf.fit(X_tr, y_tr)

        # ----- evaluate -----
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average="macro")

        print(f"\n=== Fold {fold_idx} ===")
        print("Accuracy :", acc)
        print("Macro-F1 :", f1_macro)

        # optional: per-class report
        # print(classification_report(y_te, y_pred))

        fold_results.append((acc, f1_macro))

    # ----- summary -----
    accs = [r[0] for r in fold_results]
    f1s  = [r[1] for r in fold_results]

    acc_mean = np.mean(accs)
    acc_std  = np.std(accs)
    f1_mean  = np.mean(f1s)
    f1_std   = np.std(f1s)

    print("\n================ Summary ================")
    for i, (acc, f1m) in enumerate(fold_results, start=1):
        print(f"Fold {i}: Acc={acc:.4f}, MacroF1={f1m:.4f}")
    print("-----------------------------------------")
    print(f"Avg Acc   = {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Avg F1    = {f1_mean:.4f} ± {f1_std:.4f}")

    # also dump a tiny CSV 
    results_csv = os.path.join(DATA_DIR, "svm_baseline_results.csv")
    with open(results_csv, "w") as f:
        f.write("fold,acc,macro_f1\n")
        for i,(acc,f1m) in enumerate(fold_results, start=1):
            f.write(f"{i},{acc:.6f},{f1m:.6f}\n")
        f.write(f"mean,{acc_mean:.6f},{f1_mean:.6f}\n")
        f.write(f"std,{acc_std:.6f},{f1_std:.6f}\n")

    print(f"\nSaved per-fold results to {results_csv}")


if __name__ == "__main__":
    main()

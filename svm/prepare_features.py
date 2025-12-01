import os
import glob
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ================== config ==================

DATA_ROOT = r"F:\dataset\sleep-edf\physionet.org\files\sleep-edfx\1.0.0\sleep-cassette"
OUT_DIR   = r"F:\dataset\sleep-edf\preprocessed"

EEG_CHANNEL_CANDIDATES = ["Fpz-Cz", "Pz-Oz", "EEG Fpz-Cz", "EEG Pz-Oz"]
EPOCH_LEN_SEC = 30.0

# 按proposal保留6类
label_map = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "R": 5,
    "REM": 5
}

# ================== helpers ==================

def bandpass_filter(sig, fs, low=0.5, high=35.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def find_eeg_channel(raw):
    ch_names_lower = [ch.lower() for ch in raw.ch_names]
    for cand in EEG_CHANNEL_CANDIDATES:
        c_lower = cand.lower()
        if c_lower in ch_names_lower:
            idx = ch_names_lower.index(c_lower)
            return raw.ch_names[idx]
    # fallback: first channel
    return raw.ch_names[0]

def clean_stage_label(stage_str):
    # "Sleep stage W", "Sleep stage 1", etc. -> "W","N1",...
    s = stage_str.strip().upper()
    s = s.replace("SLEEP STAGE ", "")  # remove prefix if present

    if s == "W":
        return "W"
    if s == "R":
        return "R"
    if s == "1":
        return "N1"
    if s == "2":
        return "N2"
    if s == "3":
        return "N3"
    if s == "4":
        return "N4"
    # movement, "?", etc -> drop
    return None

def build_epoch_labels_from_annotations(ann_onset, ann_desc, total_duration_sec, epoch_len_sec):
    """
    ann_onset: array of onset times in seconds
    ann_desc: array of strings ("Sleep stage W", etc.)
    total_duration_sec: float
    return: list[str or None] length = n_epochs_total
    """
    order = np.argsort(ann_onset)
    ann_onset = ann_onset[order]
    ann_desc  = np.array(ann_desc)[order]

    n_epochs_total = int(total_duration_sec // epoch_len_sec)
    epoch_labels = []

    for i in range(n_epochs_total):
        t0 = i * epoch_len_sec
        # 找到最后一个 onset <= t0
        idx = np.where(ann_onset <= t0)[0]
        if len(idx) == 0:
            epoch_labels.append(None)
            continue
        last_idx = idx[-1]
        lab = clean_stage_label(ann_desc[last_idx])
        epoch_labels.append(lab)

    return epoch_labels

def process_one_subject(psg_path, hyp_path):
    """
    psg_path: ...-PSG.edf  (signals)
    hyp_path: ...-Hypnogram.edf (annotations)
    returns epochs, labels, groups
    """

    # 1. 读 PSG (信号)
    raw_psg = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    fs = raw_psg.info["sfreq"]

    ch_name = find_eeg_channel(raw_psg)
    eeg = raw_psg.get_data(picks=[ch_name])[0]  # shape [n_samples]

    # 滤波
    eeg = bandpass_filter(eeg, fs)

    # z-score 归一化 (per subject/night)
    eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-8)

    total_duration_sec = len(eeg) / fs

    # 2. 读 Hypnogram (标签)
    #   对某些版本，Hypnogram.edf 不是真正时间序列，而是一个EDF+annotations
    #   最干脆的方式：用 mne.read_annotations() 直接解析
    try:
        ann = mne.read_annotations(hyp_path)
    except Exception:
        # 如果 read_annotations 不行, 尝试当成raw再取 annotations
        raw_hyp = mne.io.read_raw_edf(hyp_path, preload=False, verbose=False)
        ann = raw_hyp.annotations

    ann_onset = ann.onset  # seconds
    ann_desc  = ann.description

    # 3. 把 annotation 转成每30秒一个label
    epoch_labels = build_epoch_labels_from_annotations(
        ann_onset, ann_desc,
        total_duration_sec=total_duration_sec,
        epoch_len_sec=EPOCH_LEN_SEC
    )

    # 4. 丢掉无效label的 epoch，并把信号切块
    samples_per_epoch = int(EPOCH_LEN_SEC * fs)

    valid_epochs = []
    valid_labels = []

    for i, lab in enumerate(epoch_labels):
        if lab not in label_map:
            continue
        start = i * samples_per_epoch
        end   = start + samples_per_epoch
        if end <= len(eeg):
            valid_epochs.append(eeg[start:end])
            valid_labels.append(label_map[lab])

    if len(valid_epochs) == 0:
        return None

    valid_epochs = np.array(valid_epochs)   # [num_epochs, samples_per_epoch]
    valid_labels = np.array(valid_labels)   # [num_epochs]

    return valid_epochs, valid_labels, fs, ch_name

# ================== main ==================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 只抓 PSG.edf
    psg_files = glob.glob(os.path.join(DATA_ROOT, "*-PSG.edf"))
    print("Found PSG files:", len(psg_files))

    all_epochs = []
    all_labels = []
    all_groups = []

    for psg_path in tqdm(psg_files):
        pbase = os.path.basename(psg_path)

        # 尝试推断对应的 Hypnogram 文件名
        # 规则来源于 Sleep-EDF Cassette:
        #   SCxxxxE0-PSG.edf  -> SCxxxxEC-Hypnogram.edf
        #   SCxxxxE1-PSG.edf  -> SCxxxxHC-Hypnogram.edf
        hyp_path = None

        # 第一晚: E0 -> EC
        alt1 = psg_path.replace("E0-PSG.edf", "EC-Hypnogram.edf")
        # 第二晚: E1 -> HC
        alt2 = psg_path.replace("E1-PSG.edf", "HC-Hypnogram.edf")

        if os.path.exists(alt1):
            hyp_path = alt1
        elif os.path.exists(alt2):
            hyp_path = alt2
        else:
            print("⚠ Missing hypnogram for:", pbase)
            continue

        subject_id = os.path.splitext(pbase)[0]

        out = process_one_subject(psg_path, hyp_path)
        if out is None:
            print("⚠ No valid epochs for:", pbase)
            continue

        epochs, labels, fs, ch_name = out

        all_epochs.append(epochs)
        all_labels.append(labels)
        all_groups.append(np.array([subject_id] * len(labels)))

    if len(all_epochs) == 0:
        print("No valid data found.")
        return


    X = np.concatenate(all_epochs, axis=0)         # [N, T]
    y = np.concatenate(all_labels, axis=0)         # [N]
    g = np.concatenate(all_groups, axis=0)         # [N]

    X_cnn = X[:, np.newaxis, :]                    # [N, 1, T]

    np.save(os.path.join(OUT_DIR, "X_raw.npy"), X)
    np.save(os.path.join(OUT_DIR, "X_cnn.npy"), X_cnn)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "groups.npy"), g)

    print("✅ Done.")
    print("X_raw.npy:", X.shape)
    print("X_cnn.npy:", X_cnn.shape)
    print("y.npy:", y.shape)
    print("groups.npy:", g.shape)

if __name__ == "__main__":
    main()

import os, re
import numpy as np
import pandas as pd
import mne
mne.set_log_level('ERROR')
from collections import Counter

# ============================
# CONFIGURACIÓN
# ============================
BASE_PATH = r'C:\Users\ElPresio\Desktop\sleep-edf-database-expanded-1.0.0'
WORK_DIR = os.path.join(BASE_PATH, 'processed')
SUBJECT_DIR = os.path.join(WORK_DIR, 'subjects')
os.makedirs(SUBJECT_DIR, exist_ok=True)

TARGET_FS = 100
WINDOW_SEC = 30
STRIDE_SEC = 30  # sin solape
BANDPASS = (0.3, 35.0)
DEBUG = True  # Activa diagnóstico rápido

EEG_PRIORITY = ['EEG Fpz-Cz', 'Fpz-Cz', 'EEG Pz-Oz', 'Pz-Oz']
EXTRA_CHANNELS = ['EOG horizontal', 'EMG submental']  # opcional

STAGE_MAP_INT = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1, 'Sleep stage 1': 1,
    'Sleep stage N2': 2, 'Sleep stage 2': 2,
    'Sleep stage N3': 3, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
    'Sleep stage R': 4,
}
INVALID = {'Movement time','Sleep stage ?','Sleep stage ? (Artefact)','?','UNKNOWN',
           'Lights off','Lights on','Marker','Start','End'}

# ============================
# EMPAREJAR PSG-Hypnogram
# ============================
pairs = []
for subfolder in ['sleep-cassette','sleep-telemetry']:
    root = os.path.join(BASE_PATH, subfolder)
    if not os.path.exists(root): continue
    files = os.listdir(root)
    psg_files = [f for f in files if f.endswith('-PSG.edf')]
    hyp_files = [f for f in files if f.endswith('-Hypnogram.edf')]

    hyp_by_subject = {}
    for h in hyp_files:
        match = re.search(r'(SC\d+|ST\d+)', h)
        if match:
            subj = match.group(1)
            hyp_by_subject.setdefault(subj, []).append(h)

    for psg in psg_files:
        match = re.search(r'(SC\d+|ST\d+)', psg)
        if match:
            subj = match.group(1)
            candidates = hyp_by_subject.get(subj, [])
            if not candidates: continue
            pairs.append({
                'subject': subj,
                'psg_path': os.path.join(root, psg),
                'hyp_candidates': [os.path.join(root, h) for h in sorted(candidates)]
            })

print(f"Pares encontrados: {len(pairs)}")

# ============================
# DIAGNÓSTICO OPCIONAL
# ============================
if DEBUG and pairs:
    print("\n[DEBUG] Inspección rápida de etiquetas y cobertura:")
    sample_pairs = pairs[:3]
    for pair in sample_pairs:
        print("\n[Sujeto]", pair['subject'])
        for hyp in pair['hyp_candidates']:
            ann = mne.read_annotations(hyp)
            descs = [str(d).strip() for d in ann.description]
            c = Counter(descs)
            print(" ", os.path.basename(hyp), "->", {k: c[k] for k in sorted(c) if k.startswith('Sleep stage') or k in INVALID})
            raw_psg = mne.io.read_raw_edf(pair['psg_path'], preload=False, verbose='ERROR')
            dur_psg = raw_psg.times[-1]
            total_valid = sum(d for o,d,desc in zip(ann.onset, ann.duration, ann.description)
                              if STAGE_MAP_INT.get(str(desc).strip()) and str(desc).strip() not in INVALID)
            pct = 100.0 * total_valid / max(dur_psg, 1e-6)
            print(f"    Cobertura={pct:.2f}%")

# ============================
# FUNCIONES AUXILIARES
# ============================
def read_ann(hyp_path):
    return mne.read_annotations(hyp_path)

def seleccionar_canales(raw):
    chs = raw.ch_names
    selected = [c for c in EEG_PRIORITY if c in chs]
    extras = [c for c in EXTRA_CHANNELS if c in chs]
    if selected or extras:
        return raw.copy().pick_channels(selected + extras), selected + extras
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    return raw.copy().pick(eeg_picks), [raw.ch_names[i] for i in eeg_picks]

def preprocesar_eeg(psg_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose='ERROR')
    raw_sel, used_channels = seleccionar_canales(raw)
    raw_sel.filter(BANDPASS[0], BANDPASS[1], method='fir', verbose='ERROR')
    if int(raw_sel.info['sfreq']) != TARGET_FS:
        raw_sel.resample(TARGET_FS, npad='auto', verbose='ERROR')
    data = raw_sel.get_data()
    stds = data.std(axis=1, ddof=1); stds[stds==0] = 1.0
    raw_sel._data = data / stds[:,None]
    return raw_sel, used_channels

def etiquetas_por_muestra(ann, fs, n):
    y = np.full(n, -1, dtype=np.int16)
    for o,d,desc in zip(ann.onset, ann.duration, ann.description):
        desc = str(desc).strip()
        if desc in INVALID: continue
        lab = STAGE_MAP_INT.get(desc)
        if lab is None: continue
        s = max(0, int(o*fs)); e = min(n, int((o+d)*fs))
        y[s:e] = lab
    return y

def hacer_ventanas(raw_eeg, labels, subject, used_channels):
    fs = int(raw_eeg.info['sfreq'])
    x = raw_eeg.get_data()
    n = x.shape[1]
    win = fs * WINDOW_SEC
    stride = fs * STRIDE_SEC
    rows = []
    if len(labels) < n:
        labels = np.pad(labels, (0, n - len(labels)), constant_values=-1)
    for start in range(0, n - win + 1, stride):
        seg_lab = labels[start:start+win]
        valid = seg_lab[seg_lab >= 0]
        if valid.size == 0: continue
        lab = int(np.bincount(valid).argmax())
        rows.append({
            'subject': subject,
            'channels': used_channels,
            'fs': fs,
            'window_sec': WINDOW_SEC,
            'stage_int': lab,
            'signal': x[:, start:start+win].astype(np.float32)
        })
    return rows

# ============================
# PROCESAMIENTO POR SUJETO
# ============================
processed_count = 0
for pair in pairs:
    subj = pair['subject']
    psg = pair['psg_path']
    best_hyp, best_score = None, (-1.0, -1)
    for hyp in pair['hyp_candidates']:
        raw_psg = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
        dur_psg = raw_psg.times[-1]
        ann = read_ann(hyp)
        total_valid = sum(d for o,d,desc in zip(ann.onset, ann.duration, ann.description)
                          if STAGE_MAP_INT.get(str(desc).strip()) and str(desc).strip() not in INVALID)
        pct = 100.0 * total_valid / max(dur_psg, 1e-6)
        stages = {STAGE_MAP_INT.get(str(desc).strip()) for desc in ann.description if STAGE_MAP_INT.get(str(desc).strip())}
        score = (pct, len(stages))
        if score > best_score:
            best_score, best_hyp = score, hyp
    if best_hyp is None or best_score[0] < 1.0:
        print(f"[SKIP] {subj}: cobertura {best_score[0]:.2f}%")
        continue

    print(f"[INFO] {subj}: {os.path.basename(best_hyp)} cobertura={best_score[0]:.2f}% etapas={best_score[1]}")
    raw_eeg, used_channels = preprocesar_eeg(psg)
    ann = read_ann(best_hyp)
    labels = etiquetas_por_muestra(ann, int(raw_eeg.info['sfreq']), raw_eeg.n_times)
    rows = hacer_ventanas(raw_eeg, labels, subj, used_channels)

    if rows:
        df_subj = pd.DataFrame(rows)
        # Guardar CSV sin la señal
        out_csv = os.path.join(SUBJECT_DIR, f'{subj}_features.csv')
        df_subj.drop(columns=['signal']).to_csv(out_csv, index=False)
        # Guardar señales en NPZ
        X = np.stack(df_subj['signal'].values).transpose(0, 2, 1)
        y = df_subj['stage_int'].values
        out_npz = os.path.join(SUBJECT_DIR, f'{subj}_Xy.npz')
        np.savez_compressed(out_npz, X=X, y=y)
        print(f"[OK] Guardado sujeto {subj}: {out_csv}, {out_npz}")
        processed_count += 1
    else:
        print(f"[WARN] {subj}: no se generaron ventanas.")

print(f"\n[FIN] Sujetos procesados: {processed_count}/{len(pairs)}")
print(f"[INFO] Archivos guardados en: {SUBJECT_DIR}")
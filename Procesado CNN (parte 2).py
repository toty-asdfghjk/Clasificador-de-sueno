# ⬅️ Ejecuta esta celda en Google Colab
from google.colab import drive
drive.mount('/content/drive')

# ⚠️ CAMBIA ESTA RUTA a la carpeta que contiene tus archivos por sujeto:
# Debe ser la carpeta donde están SCxxxx_Xy.npz y SCxxxx_features.csv
SUBJECT_DIR = '/content/drive/MyDrive/subjects'  # <-- EDITA AQUÍ

import os, numpy as np, pandas as pd
N_CLASSES = 5  # W, N1, N2, N3, R
RATIOS = (0.6, 0.2, 0.2)  # train / val / test
SEED = 42

# Sanity check
print("[INFO] SUBJECT_DIR existe:", os.path.exists(SUBJECT_DIR))
print("    Ejemplos en la carpeta:", sorted([f for f in os.listdir(SUBJECT_DIR)][:10]))

import numpy as np
import os

def list_subjects(subject_dir=SUBJECT_DIR):
    """IDs de sujetos con NPZ disponible."""
    files = [f for f in os.listdir(subject_dir) if f.endswith('_Xy.npz')]
    ids = sorted([f.split('_')[0] for f in files])  # 'SC4001' de 'SC4001_Xy.npz'
    if not ids:
        raise FileNotFoundError("No se encontraron '*_Xy.npz' en SUBJECT_DIR.")
    return ids

def load_counts_for_subject(sid, subject_dir=SUBJECT_DIR, n_classes=N_CLASSES):
    """Bincount de y para el sujeto."""
    path = os.path.join(subject_dir, f"{sid}_Xy.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ no encontrado para sujeto {sid}: {path}")
    y = np.load(path)['y']
    return np.bincount(y, minlength=n_classes)

def entropy(counts):
    """Entropía de la distribución (para ordenar sujetos con más diversidad primero)."""
    p = counts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())

def stratified_subject_split(subject_dir=SUBJECT_DIR, ratios=(0.6, 0.2, 0.2), seed=SEED, n_classes=N_CLASSES):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Los ratios deben sumar 1.0"
    subjects = list_subjects(subject_dir)
    dist = [(sid, load_counts_for_subject(sid, subject_dir, n_classes)) for sid in subjects]

    # Ordenar por entropía descendente (sujetos más "ricos" en diversidad primero)
    dist.sort(key=lambda t: entropy(t[1]), reverse=True)

    total = np.sum([c for _, c in dist], axis=0)
    target = {
        'train': total * ratios[0],
        'val':   total * ratios[1],
        'test':  total * ratios[2]
    }
    sets = {'train': [], 'val': [], 'test': []}
    sums = {'train': np.zeros(n_classes), 'val': np.zeros(n_classes), 'test': np.zeros(n_classes)}

    # Semilla inicial para evitar algún split vacío
    preset = ['train', 'val', 'test']
    for i, grp in enumerate(preset):
        if i < len(dist):
            sid, c = dist[i]
            sets[grp].append(sid); sums[grp] += c

    # Greedy: asigna cada sujeto al split cuya suma se acerque más al target
    for i in range(len(preset), len(dist)):
        sid, c = dist[i]
        diffs = {
            'train': np.linalg.norm((sums['train'] + c) - target['train']),
            'val':   np.linalg.norm((sums['val']   + c) - target['val']),
            'test':  np.linalg.norm((sums['test']  + c) - target['test'])
        }
        pick = min(diffs, key=diffs.get)
        sets[pick].append(sid); sums[pick] += c

    # Reequilibrar por cantidad de sujetos (además de clases)
    n_total = len(subjects)
    target_n = {
        'train': int(round(ratios[0] * n_total)),
        'val':   int(round(ratios[1] * n_total)),
        'test':  n_total - int(round(ratios[0] * n_total)) - int(round(ratios[1] * n_total))
    }
    def move_one(src, dst):
        if not sets[src]:
            return False
        sid = sets[src].pop()  # mover el último (costo computacional simple)
        c = load_counts_for_subject(sid, subject_dir, n_classes)
        sets[dst].append(sid)
        sums[src] -= c; sums[dst] += c
        return True

    changed = True
    while changed:
        changed = False
        for grp in ['train', 'val', 'test']:
            while len(sets[grp]) > target_n[grp]:
                dest = min(['train','val','test'], key=lambda g: len(sets[g]) - target_n[g])
                if dest == grp:
                    break
                if move_one(grp, dest):
                    changed = True

    return sets['train'], sets['val'], sets['test'], sums

def save_manifests(train_ids, val_ids, test_ids, subject_dir=SUBJECT_DIR):
    pd.DataFrame({'subject': train_ids}).to_csv(os.path.join(subject_dir, 'train_manifest.csv'), index=False)
    pd.DataFrame({'subject': val_ids}).to_csv(os.path.join(subject_dir, 'val_manifest.csv'), index=False)
    pd.DataFrame({'subject': test_ids}).to_csv(os.path.join(subject_dir, 'test_manifest.csv'), index=False)

def split_counts(ids, subject_dir=SUBJECT_DIR, n_classes=N_CLASSES):
    acc = np.zeros(n_classes, dtype=np.int64)
    for sid in ids:
        acc += load_counts_for_subject(sid, subject_dir, n_classes)
    return acc

# Ejecutar split y guardar
train_ids, val_ids, test_ids, sums = stratified_subject_split(SUBJECT_DIR, RATIOS, SEED, N_CLASSES)
save_manifests(train_ids, val_ids, test_ids, SUBJECT_DIR)

ct, cv, tt = split_counts(train_ids), split_counts(val_ids), split_counts(test_ids)
print(f"[SPLIT] subjects -> train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
print("[CLASSES] train:", ct)
print("[CLASSES] val:  ", cv)
print("[CLASSES] test: ", tt)
print(f"[OK] Manifests guardados en: {SUBJECT_DIR}")
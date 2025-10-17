##############################################
###### Extracción de features Sleep-EDF ######
##############################################

#Librerías
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, entropy, skew
import mne
from tqdm import tqdm

#########################################
###### Rutas de carpetas definidas ######
#########################################
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # o Path.cwd() si usas Colab
DATAPATH = BASE_DIR / "sleep-edf-database-expanded-1.0.0" / "sleep-cassette"
OUT_FOLDER = BASE_DIR / "sleep-edf-processed"
OUT_FOLDER.mkdir(exist_ok=True, parents=True)

print(f"Carpeta de entrada: {DATAPATH}")
print(f"Carpeta de salida: {OUT_FOLDER}")

#######################################
###### Funcion de mapeo de clases #####
#######################################
"""funcion que va a convertir las annotations a codigos numericos (labels para el dataframe)"""
def annotation_to_label(descript):
    """Convierte descripción textual a etiqueta y código numérico."""
    if descript is None:
        return None, None
    d = descript.lower()

    #se toman los stage de annotations
    if "sleep stage w" in d:
        return "W", 0
    if "sleep stage 1" in d:
        return "N1", 1
    if "sleep stage 2" in d:
        return "N2", 2
    #tomamos sleep stage 4 como 3 (se puede hacer)
    if "sleep stage 3" in d or "sleep stage 4" in d:
        return "N3", 3
    if "sleep stage r" in d:
        return "REM", 4
    
    #casos de no interes como ?
    return None, None

######################################
##### Construcción del DataFrame #####
######################################
"""Crea DataFrame de anotaciones y etiquetas de 30s"""
def build_epoch_df(datapath, out_folder):
    #se buscan los archivos de hipnograma
    files = sorted(os.listdir(datapath))
    hypno_files = [os.path.join(datapath, f) for f in files if "Hypnogram" in f]

    #almacenamiento de datos
    epoch_row, annotation_row, epoch_count = [], [], {}

    for hypno_file in tqdm(hypno_files, desc="Leyendo anotaciones"):
        try:
            anotacion = mne.read_annotations(hypno_file)
        except Exception as e:
            print(f"Error al leer {hypno_file}: {e}")
            continue

        #ID del sujeto
        match = re.search(r"(SC\d+)", os.path.basename(hypno_file))
        subject = match.group(1) if match else os.path.basename(hypno_file).split("-")[0]
        epoch_count.setdefault(subject, 0)

        #itera sobre las annotaciones (data relevante)
        for onset, duration, descript in zip(anotacion.onset, anotacion.duration, anotacion.description):
            stage, code = annotation_to_label(descript)
            if stage is None:
                continue
            
            #se guardan datos relevantes de la anotacion
            annotation_row.append({
                "sujeto": subject,
                "hypno_file": os.path.basename(hypno_file),
                "onset": onset,
                "duration": duration,
                "stage": stage,
                "code": code
            })

            #se divide en epocas de 30s
            n_epochs = int(round(duration / 30))

            #se guardan datos relevantes de la epoca
            for i in range(n_epochs):
                epoch_row.append({
                    "sujeto": subject,
                    "epoch_onset": onset + i * 30,
                    "epoch_duration": 30.0,
                    "stage": stage,
                    "code": code,
                    "epoch_index_subject": epoch_count[subject]
                })
                epoch_count[subject] += 1
    #creacion de dataframes
    ann_df = pd.DataFrame(annotation_row)
    epoch_df = pd.DataFrame(epoch_row)

    #se guarda csv para uso en ML
    ann_df.to_csv(out_folder / "annotations_summary.csv", index=False)
    epoch_df.to_csv(out_folder / "epochs_30s_labels.csv", index=False)

    print(f"Guardados en {out_folder}:")
    print(" - annotations_summary.csv")
    print(" - epochs_30s_labels.csv")
    print("Distribución por clase:")
    print(epoch_df["stage"].value_counts())

    return epoch_df, ann_df


#################################
##### Funciones de features #####
#################################
"""funcion auxiliar"""
def bandpower(freqs, psd, low, high):
    idx = np.logical_and(freqs >= low, freqs <= high)
    
    return np.trapezoid(psd[idx], freqs[idx])

"""funciones para extraer features (EEG, EMG, EOG)"""
def extract_features_eeg(signal, fs, canal="EEG"):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    total_pow = np.trapezoid(psd, freqs) or 1e-23 #valor pequeño para evitar 0
    #bandas de frecuencia
    bands = {
        "delta": (0.5, 4.5),
        "theta": (4.5, 8.5),
        "alpha": (8.5, 11.5),
        "sigma": (11.5, 15.5),
        "beta": (15.5, 30),
    }
    #se almacenan los features
    feats = {f"{canal}_{b}": bandpower(freqs, psd, low, high) / total_pow
             for b, (low, high) in bands.items()} #se asignan los features por cada banda para EEG

    #se asignan features de std, mean, entropy, kurtosis, skew (features extras)
    feats.update({
        f"{canal}_mean": np.mean(signal),
        f"{canal}_std": np.std(signal),
        f"{canal}_entropy": float(entropy(np.abs(signal) / np.sum(np.abs(signal)))),
        f"{canal}_kurtosis": kurtosis(signal),
        f"{canal}_skew": skew(signal)
    })

    return feats


def extract_features_eog(signal, fs, canal="EOG"):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    total_pow = np.trapezoid(psd, freqs) or 1e-23 #potencia de señal (integral de potencia)

    #se almacenan los features
    feats = {
        f"{canal}_lf_power": bandpower(freqs, psd, 0.1, 5) / total_pow, #se toman valores de low - high
        f"{canal}_mean": np.mean(signal),
        f"{canal}_std": np.std(signal),
        f"{canal}_entropy": entropy(np.abs(signal) / np.sum(np.abs(signal))),
        f"{canal}_kurtosis": kurtosis(signal),
    }

    return feats


def extract_features_emg(signal, fs, canal="EMG"):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    total_pow = np.trapezoid(psd, freqs) or 1e-23 #potencia de señal (integral de potencia)

    #se almacenan los features
    feats = {
        f"{canal}_total_power": total_pow,
        f"{canal}_hf_power": bandpower(freqs, psd, 30, 100) / total_pow, #se toman valores de low - high
        f"{canal}_rms": np.sqrt(np.mean(signal**2)),
        f"{canal}_std": np.std(signal),
        f"{canal}_entropy": entropy(np.abs(signal) / np.sum(np.abs(signal))),
        f"{canal}_kurtosis": kurtosis(signal)
    }

    return feats

"""funcion que integra todas las features"""
def extract_features_multichannel(raw, start, end, fs):
    #se almacenan los features
    feats = {}
    if "EEG Fpz-Cz" in raw.ch_names:
        feats.update(extract_features_eeg(raw.get_data(picks=["EEG Fpz-Cz"], start=start, stop=end)[0], fs, "EEG_FpzCz"))
    if "EEG Pz-Oz" in raw.ch_names:
        feats.update(extract_features_eeg(raw.get_data(picks=["EEG Pz-Oz"], start=start, stop=end)[0], fs, "EEG_PzOz"))
    if "EOG horizontal" in raw.ch_names:
        feats.update(extract_features_eog(raw.get_data(picks=["EOG horizontal"], start=start, stop=end)[0], fs, "EOG"))
    if "EMG submental" in raw.ch_names:
        feats.update(extract_features_emg(raw.get_data(picks=["EMG submental"], start=start, stop=end)[0], fs, "EMG"))

    return feats


###################################
##### Procesamiento principal #####
###################################
"""Genera features por sujeto combinando EEG/EOG/EMG."""
def process_all_subjects(datapath, epoch_df, out_folder):
    psg_files = sorted(f for f in os.listdir(datapath) if f.endswith("PSG.edf"))
    subjects = sorted(list(set(re.findall(r"(SC\d+)", " ".join(psg_files)))))

    print(f"Encontrados {len(subjects)} sujetos.")
    #se itera sobre todos los sujetos
    for sujeto in tqdm(subjects, desc="Procesando sujetos"):
        try:
            #algunos sujetos tienen F0 o G0 en vez de E0
            possible_files = [
                datapath / f"{sujeto}E0-PSG.edf",
                datapath / f"{sujeto}F0-PSG.edf",
                datapath / f"{sujeto}G0-PSG.edf"
            ]
            psg_file = next((p for p in possible_files if p.exists()), None)
            if psg_file is None:
                print(f"{sujeto}: no se encontró archivo EDF.")
                continue

            #se lee el archivo
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            fs = int(raw.info["sfreq"])
            samples_epoch = 30 * fs

            subject_epochs = epoch_df[epoch_df["sujeto"] == sujeto].sort_values("epoch_onset")
            rows = []

            #se integran los features
            for _, row in subject_epochs.iterrows():
                start, end = int(row["epoch_onset"] * fs), int(row["epoch_onset"] * fs) + samples_epoch
                feats = {"sujeto": sujeto,
                    "epoch_index": row["epoch_index_subject"],
                    "stage": row["stage"],
                    "code": row["code"]}
                feats.update(extract_features_multichannel(raw, start, end, fs))
                rows.append(feats)

            #se arman los dataframes
            df = pd.DataFrame(rows)
            df.to_csv(out_folder / f"{sujeto}_features_multichannel.csv", index=False)
            print(f"{sujeto} guardado ({df.shape[0]} épocas, {df.shape[1]-4} features)")

        except Exception as e:
            print(f"Error procesando {sujeto}: {e}")


###########################################
##### Concatenación y dataframe final #####
###########################################
"""se arma el dataframe definitivo"""
def combine_all_features(out_folder):
    all_files = list(out_folder.glob("*_features_multichannel.csv"))
    if not all_files:
        raise RuntimeError("No se encontraron archivos de features multicanal.")
    all_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    final_path = out_folder / "all_subjects_features_multichannel.csv"
    all_df.to_csv(final_path, index=False)
    print(f"\nDataFrame total guardado: {final_path} ({all_df.shape})")

    return all_df


########################
##### Funcion main #####
########################
if __name__ == "__main__":
    epoch_df, ann_df = build_epoch_df(DATAPATH, OUT_FOLDER)
    print(epoch_df.head())
    process_all_subjects(DATAPATH, epoch_df, OUT_FOLDER)
    df_all = combine_all_features(OUT_FOLDER)
# ==== Configuración de paths ====
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import time
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# ==== Utilidades ====
def infer_dims(subject_dir, subject_id=None):
    """Inspecciona un NPZ para inferir (time_steps, n_channels)."""
    files = sorted([f for f in os.listdir(subject_dir) if f.endswith('_Xy.npz')])
    if not files:
        raise FileNotFoundError("No hay NPZ en SUBJECT_DIR.")
    f = files[0] if subject_id is None else f"{subject_id}_Xy.npz"
    data = np.load(os.path.join(subject_dir, f))
    X = data['X']  # (N_win, time, C)
    return X.shape[1], X.shape[2]

class SleepEDFSeqDataset(Dataset):
    """
    Forma samples tipo:
      x: [1, T_concat] donde T_concat = seq_len * time_steps (solo 1 canal)
      y: etiqueta de la secuencia (mayoría o última ventana)
    """
    def __init__(self, subject_ids, subject_dir, seq_len=10, seq_stride=5,
                 primary_channel=0, label_mode='majority', 
                 balance_wake=True):  # <--- NUEVO PARÁMETRO
        
        self.subject_ids = list(subject_ids)
        self.subject_dir = subject_dir
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.primary_channel = int(primary_channel)
        assert label_mode in ('majority', 'last')
        self.label_mode = label_mode

        # Construir índice global (sid, start_idx)
        self.index = []
        self.lengths = {}
        
        # 30 minutos en épocas (asumiendo épocas de 30s)
        # 30 min * 60 seg / 30 seg_por_epoca = 60 épocas
        WAKE_BUFFER = 180 

        for sid in self.subject_ids:
            # Cargar solo etiquetas para calcular índices (rápido)
            # Usamos mmap_mode='r' para no saturar RAM leyendo la matriz X
            data = np.load(os.path.join(self.subject_dir, f"{sid}_Xy.npz"), mmap_mode='r')
            y = data['y']
            n = len(y)
            self.lengths[sid] = n
            
            start_index = 0
            end_index = n

            if balance_wake:
                # Lógica de Recorte: Encontrar primer y último momento de sueño
                # Asumimos que Wake es 0. Buscamos cualquier cosa != 0
                is_sleep = np.where(y != 0)[0]
                
                if len(is_sleep) > 0:
                    first_sleep = is_sleep[0]
                    last_sleep = is_sleep[-1]
                    
                    # Definir nuevos límites con margen (buffer)
                    start_index = max(0, first_sleep - WAKE_BUFFER)
                    end_index = min(n, last_sleep + WAKE_BUFFER)
                else:
                    # Si el sujeto NUNCA durmió (raro, pero posible), 
                    # tomamos un trozo central o lo ignoramos.
                    # Aquí lo dejamos igual para no romper nada.
                    pass

            # Solo generamos secuencias dentro del rango útil
            # El límite superior del range debe asegurar que cabe una secuencia completa
            max_start = end_index - self.seq_len + 1
            
            if max_start > start_index:
                for s in range(start_index, max_start, self.seq_stride):
                    self.index.append((sid, s))

        # Inferir dims (igual que antes)
        self.time_steps, self.n_channels = infer_dims(self.subject_dir)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, s = self.index[idx]
        path = os.path.join(self.subject_dir, f"{sid}_Xy.npz")
        data = np.load(path, mmap_mode='r')
        X, y = data['X'], data['y']            # X: (Nwin, time, C)
        Xseq = X[s:s+self.seq_len, :, :]       # (seq_len, time, C)
        yseq = y[s:s+self.seq_len]             # (seq_len,)

        # Tomar solo el canal principal y concatenar en tiempo
        x_1c = Xseq[:, :, self.primary_channel]        # (seq_len, time)
        x_concat = x_1c.reshape(-1)                    # (seq_len*time,)
        x_tensor = torch.tensor(x_concat, dtype=torch.float32).unsqueeze(0)  # [1, T_concat]

        # Etiqueta de secuencia
        if self.label_mode == 'last':
            ylab = int(yseq[-1])
        else:
            ylab = int(np.bincount(yseq).argmax())

        y_tensor = torch.tensor(ylab, dtype=torch.long)
        return x_tensor, y_tensor


class EarlyStopping:
        def __init__(self, n_epochs_tolerance=5):
            self.patience = n_epochs_tolerance
            self.best_loss = float('inf')
            self.counter = 0
            self.should_stop = False

        def __call__(self, val_loss):
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            return self.should_stop
        

class CNN_LSTM_Seq(nn.Module):
    """
    Acepta entradas [B, 1, T_concat].
    Conv1d -> Conv1d -> permute a [B, T', F] -> LSTM -> Clasificador.
    """
    def __init__(self, num_classes=5, lstm_hidden=128, dropout_p=0.35):
        super().__init__()
        # --- 1. CNN EXTRACTOR (128 features) ---
        self.cnn = nn.Sequential(
            # Bloque 1
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Bloque 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Bloque 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten_dim = 128

        # --- 2. MLP COMPRESOR (128 -> 32 features) ---
        # Cortamos justo antes de la capa final.
        self.mlp_compressor = nn.Sequential(
            nn.Flatten(),
            
            # Capa 1: 128 -> 64
            nn.Linear(self.flatten_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            # Capa 2: 64 -> 32 (Aquí se generan los features comprimidos)
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout_p) 
        )

        # --- 3. CLASIFICADOR FINAL (32 -> 5 clases) ---
        # Esta capa solo se usa para entrenar, no para extraer features "mlp"
        self.classifier_head = nn.Linear(32, num_classes)

    def forward(self, x):
        # Flujo normal de entrenamiento
        x = self.cnn(x)             # [B, 128, T]
        x = self.global_pool(x)     # [B, 128, 1]
        
        x = self.mlp_compressor(x)  # [B, 32]  <-- Pasamos por la parte densa
        logits = self.classifier_head(x) # [B, 5] <-- Clasificación final
        return logits

    def extract_features(self, x, mode="cnn"):
        """
        Extrae características según el modo elegido.
        Args:
            x (tensor): Input batch [B, 1, T]
            mode (str): "cnn" para vector de 128 dim, "mlp" para vector de 32 dim.
        """
        with torch.no_grad():
            # Paso 1: Siempre pasamos por la CNN
            x = self.cnn(x)
            x = self.global_pool(x) # [B, 128, 1]
            
            if mode == "cnn":
                return x.flatten(1) # Retorna [B, 128]
            
            elif mode == "mlp":
                # Paso 2: Si queremos MLP, pasamos por el compresor
                feat_32 = self.mlp_compressor(x)
                return feat_32 # Retorna [B, 32]
            
            else:
                raise ValueError("Mode debe ser 'cnn' o 'mlp'")


def train_step(x_batch, y_batch, model, optimizer, criterion, device):
    """
    Paso de entrenamiento independiente (si quieres usarlo en bucles manuales).
    """
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    y_predicted = model(x_batch)
    loss = criterion(y_predicted, y_batch)
    loss.backward()
    optimizer.step()

    return y_predicted, loss

def calcular_pesos_clases(subject_ids, subject_dir, num_classes=5):
    """
    Calcula pesos leyendo directamente los archivos .npz (Instantáneo).
    Replica la lógica de recorte de Wake para que los pesos sean precisos.
    """
    print(f"Calculando pesos (Modo Rápido) para {len(subject_ids)} sujetos...")
    
    # Acumulador de conteos por clase
    total_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Lógica de buffer (debe coincidir con tu Dataset)
    WAKE_BUFFER = 180 # 30 min * 2 épocas/min (ajusta según tu dataset)

    for sid in tqdm(subject_ids, desc="Escaneando archivos raw"):
        path = os.path.join(subject_dir, f"{sid}_Xy.npz")
        
        # Cargar SOLO la etiqueta 'y' (muy liviano)
        try:
            # mmap_mode='r' es clave para no explotar la RAM
            data = np.load(path, mmap_mode='r') 
            y = data['y']
        except Exception as e:
            print(f"Advertencia: No se pudo leer {sid}: {e}")
            continue
        
        # --- APLICAR RECORTE (CROP) ---
        # Tenemos que contar solo lo que el modelo va a ver realmente
        # Si contamos todo el Wake del archivo original, los pesos saldrán mal.
        
        # Buscar índices de sueño real
        is_sleep = np.where(y != 0)[0]
        
        if len(is_sleep) > 0:
            first_sleep = is_sleep[0]
            last_sleep = is_sleep[-1]
            
            start_index = max(0, first_sleep - WAKE_BUFFER)
            end_index = min(len(y), last_sleep + WAKE_BUFFER)
            
            # Cortamos el array
            y_cropped = y[start_index:end_index]
            
            # Contamos clases en este sujeto
            counts = np.bincount(y_cropped, minlength=num_classes)
            
            # Sumar al total global
            # Nota: bincount puede devolver menos de 5 si faltan clases, aseguramos shape
            if len(counts) < num_classes:
                counts = np.pad(counts, (0, num_classes - len(counts)), 'constant')
            elif len(counts) > num_classes:
                counts = counts[:num_classes]
                
            total_counts += counts
        else:
            # Si es puro Wake, tomamos todo (o nada, según tu lógica)
            pass

    print(f"Conteos totales (Train): {total_counts}")
    
    # Evitar división por cero
    total_counts = np.maximum(total_counts, 1) 
    total_samples = np.sum(total_counts)
    
    # Fórmula de pesos balanceados
    weights = total_samples / (num_classes * total_counts.astype(float))
    
    return torch.tensor(weights, dtype=torch.float32)

# ============================================
# Entrenamiento con DataLoaders (Opción B)
# ============================================

def train_model_with_loaders(
    model,
    train_loader,
    val_loader,
    max_epochs=30,
    lr=5e-4,
    criterion=nn.CrossEntropyLoss(),
    n_evaluations_per_epoch=6,
    early_stopping_tolerance=25,
    use_gpu=False,
    save_path="best_model_seq.pth",
    num_classes=5,
    class_weights=None
):
    # Optimización 1: CuDNN Benchmark (busca el mejor algoritmo de convolución)
    #if torch.cuda.is_available():
        #torch.backends.cudnn.benchmark = True 
    
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Inicializar Scaler para Mixed Precision
    scaler = GradScaler() # <--- NUEVO

    if isinstance(criterion, nn.CrossEntropyLoss) and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if hasattr(class_weights, 'to') else class_weights)

    curves = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": [], "train_cm": [], "val_cm": []}

    best_val_loss = float('inf')
    early_stopping_counter = 0
    t0 = time.perf_counter()
    iteration = 0
    n_batches = len(train_loader)
    eval_every = max(1, n_batches // max(1, n_evaluations_per_epoch))

    for epoch in range(1, max_epochs + 1):
        # ===== TRAIN =====
        model.train()
        cumulative_train_loss = 0.0
        cumulative_train_corrects = 0
        train_loss_count = 0
        train_acc_count = 0

        epoch_train_preds = []
        epoch_train_labels = []

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [TRAIN]", leave=False)
        
        for i, (x_batch, y_batch) in enumerate(pbar_train):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # --- BLOQUE OPTIMIZADO CON AMP ---
            with autocast(device_type='cuda', dtype=torch.float16): # <--- Python usa float16 donde es seguro
                y_predicted = model(x_batch)
                loss = criterion(y_predicted, y_batch)
            
            scaler.scale(loss).backward() # <--- Escalado de gradientes
            scaler.step(optimizer)
            scaler.update()
            # ---------------------------------

            cumulative_train_loss += loss.item()
            train_loss_count += 1
            train_acc_count += y_batch.shape[0]

            preds = torch.argmax(y_predicted, dim=1).long()
            cumulative_train_corrects += (preds == y_batch).sum().item()
            
            # (Opcional: Reduce la frecuencia de updates a numpy para ahorrar CPU)
            epoch_train_preds.append(preds.detach().cpu().numpy())
            epoch_train_labels.append(y_batch.detach().cpu().numpy())

            iteration += 1
            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

            # Impresión intermedia (opcional)
            if (i % eval_every == 0) and (i > 0):
                train_loss_inter = cumulative_train_loss / max(1, train_loss_count)
                train_acc_inter  = cumulative_train_corrects / max(1, train_acc_count)
                print(f"Epoch {epoch}/{max_epochs} Iter {iteration} - Train loss: {train_loss_inter:.4f}, Train acc: {train_acc_inter:.4f}")

        # Métricas de TRAIN (por época)
        epoch_train_preds  = np.concatenate(epoch_train_preds) if len(epoch_train_preds) > 0 else np.array([])
        epoch_train_labels = np.concatenate(epoch_train_labels) if len(epoch_train_labels) > 0 else np.array([])
        if len(epoch_train_labels) > 0:
            train_cm  = confusion_matrix(epoch_train_labels, epoch_train_preds, labels=np.arange(num_classes))
        else:
            train_cm = np.zeros((num_classes, num_classes), dtype=int)

        train_loss = cumulative_train_loss / max(1, train_loss_count)
        train_acc  = cumulative_train_corrects / max(1, train_acc_count)
        curves["train_cm"].append(train_cm)

        # ===== VALIDATION =====
        model.eval()
        val_loss_total = 0.0
        val_corrects = 0
        val_preds_all, val_labels_all = [], []

        # Barra de progreso para VAL
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} [VAL]  ", leave=False)
            for x_val, y_val in pbar_val:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                y_val_pred = model(x_val)
                batch_loss = criterion(y_val_pred, y_val).item()
                val_loss_total += batch_loss

                preds = torch.argmax(y_val_pred, dim=1)
                val_corrects += (preds == y_val).sum().item()

                val_preds_all.append(preds.detach().cpu().numpy())
                val_labels_all.append(y_val.detach().cpu().numpy())

                # Actualizar barra con métricas instantáneas de validación
                pbar_val.set_postfix({
                    "loss_batch": f"{batch_loss:.4f}",
                    "acc_batch": f"{(preds == y_val).float().mean().item():.4f}"
                })

        val_loss = val_loss_total / max(1, len(val_loader))
        total_val_samples = sum(len(lbl) for lbl in val_labels_all) if len(val_labels_all) > 0 else 0
        val_acc  = (val_corrects / max(1, total_val_samples)) if total_val_samples > 0 else 0.0
        val_preds_all  = np.concatenate(val_preds_all) if len(val_preds_all) > 0 else np.array([])
        val_labels_all = np.concatenate(val_labels_all) if len(val_labels_all) > 0 else np.array([])
        if len(val_labels_all) > 0:
            val_cm = confusion_matrix(val_labels_all, val_preds_all, labels=np.arange(num_classes))
        else:
            val_cm = np.zeros((num_classes, num_classes), dtype=int)

        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["val_cm"].append(val_cm)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")

        # ===== Guardar mejor por val_loss =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_model_state, save_path)
            print(f"✔ Mejor modelo guardado en {save_path} (Val loss={best_val_loss:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= early_stopping_tolerance:
            print("Early stopping activado.")
            break

    print(f"\nTiempo total: {time.perf_counter() - t0:.2f} s")
    model.to('cpu')
    return curves

def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    SUBJECT_DIR = r'C:\Users\SirTo\Desktop\Proyecto Sleep Classifier\subjects'  # <-- AJUSTA A TU RUTA
    # ==== Cargar manifests ====
    train_ids = pd.read_csv(os.path.join(SUBJECT_DIR, 'train_manifest.csv'))['subject'].tolist()
    val_ids   = pd.read_csv(os.path.join(SUBJECT_DIR, 'val_manifest.csv'))['subject'].tolist()
    test_ids  = pd.read_csv(os.path.join(SUBJECT_DIR, 'test_manifest.csv'))['subject'].tolist()
    print(f"[SPLIT] train/val/test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    # 1. CALCULAR PESOS RÁPIDO (Antes de crear el Dataset incluso)
    pesos_calculados = calcular_pesos_clases(train_ids, SUBJECT_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pesos_calculados = pesos_calculados.to(device)
    print(f"Pesos finales: {pesos_calculados}")

    # ==== Crear datasets ====
    SEQ_LEN = 5        # 10 x 30s = 5 min de contexto
    SEQ_STRIDE = 5      # 50% de solape entre secuencias
    PRIMARY_CH = 0      # usa EEG principal (ajusta si quieres otro canal)
    LABEL_MODE = 'majority'

    train_ds = SleepEDFSeqDataset(train_ids, SUBJECT_DIR, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE,
                                primary_channel=PRIMARY_CH, label_mode=LABEL_MODE)
    val_ds   = SleepEDFSeqDataset(val_ids,   SUBJECT_DIR, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE,
                                primary_channel=PRIMARY_CH, label_mode=LABEL_MODE)
    test_ds  = SleepEDFSeqDataset(test_ids,  SUBJECT_DIR, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE,
                                primary_channel=PRIMARY_CH, label_mode=LABEL_MODE)

    # ==== DataLoaders ====
    BATCH_SIZE = 64
    USE_GPU = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=USE_GPU)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=USE_GPU)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=USE_GPU)

    # Sanity check de shapes
    xb, yb = next(iter(train_loader))
    print("[SHAPES] xb:", tuple(xb.shape), "yb:", tuple(yb.shape))  # esperado: xb [B, 1, T_concat], yb [B]

    # ============================================
    # Loop de entrenamiento Multi-Run (Opción B)
    # ============================================

    # Parámetros
    lr = 5e-4
    dropout_p = 0.35
    epochs = 30
    use_gpu = True
    num_classes = 5
    n_reps = 5

    # Historiales
    curves_history = []
    val_acc_history = []
    val_cm_history = []

    device = torch.device('cuda' if use_gpu else 'cpu')

    for i in range(n_reps):
        print(f"\nIteración CNN-LSTM Seq N° {i+1}")

        # Instanciar modelo compatible con [B, 1, T_concat]
        model = CNN_LSTM_Seq(
            num_classes=num_classes,
            lstm_hidden=128,
            dropout_p=dropout_p
        )

        # Entrenar con DataLoaders existentes
        curves = train_model_with_loaders(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=epochs,
            lr=lr,
            criterion=nn.CrossEntropyLoss(),    # si quieres class_weights, pásalos aquí
            n_evaluations_per_epoch=5,
            early_stopping_tolerance=5,
            use_gpu=use_gpu,
            save_path=f"best_model_seq_run{i+1}.pth",
            num_classes=num_classes,
            class_weights=pesos_calculados                  # opcional: tensor de pesos por clase
        )

        # Evaluación final en validación
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_true.append(yb.numpy())
                all_pred.append(preds)
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        val_acc = (y_true == y_pred).mean().item()
        cm_val = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes), normalize='true')

        val_acc_history.append(val_acc)
        val_cm_history.append(cm_val)
        curves_history.append(curves)

    # Estadísticas finales
    val_acc_history = np.array(val_acc_history)

    #guardar pesos (RECOMENDADO)
    torch.save(model.state_dict(), "cnn_lstm_weights.pth")

    #guardar copia completa (RESPALDO mas que nada)
    torch.save(model, "cnn_lstm_entire_model.pt")


if __name__ == "__main__":
    main()
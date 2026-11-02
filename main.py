import os
import json
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
from keras.layers import Layer
import uvicorn

# ==================== PATCH KOMPATIBILITAS KERAS ====================
# Model disimpan dengan Keras versi lebih baru yang menyertakan
# `quantization_config` di Dense. Versi terpasang tidak mengenali field
# tersebut, sehingga perlu dihapus saat deserialisasi.
_dense_from_config_orig = keras.layers.Dense.from_config.__func__

@classmethod  # type: ignore[misc]
def _dense_from_config_patched(cls, config):
    config = dict(config)
    config.pop('quantization_config', None)
    return _dense_from_config_orig(cls, config)

keras.layers.Dense.from_config = _dense_from_config_patched

# ==================== PENGATURAN AWAL ====================
app = FastAPI(title="SIBI SPOK Realtime Backend - 20 Kelas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== KELAS CUSTOM KERAS ====================
@keras.saving.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='custom')
class FocalLossWithSmoothing(tf.keras.losses.Loss):
    def __init__(self, gamma=2.5, alpha=None, label_smoothing=0.05,
                 name='focal_loss_with_smoothing', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else [1.0] * 20
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + \
                        self.label_smoothing / tf.cast(n_classes, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_smooth * tf.math.log(y_pred)
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        alpha_w = tf.reduce_sum(y_true * alpha_tensor, axis=-1, keepdims=True)
        focal_w = alpha_w * tf.pow(1.0 - pt, self.gamma)
        return tf.reduce_mean(focal_w * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
        })
        return cfg


# ==================== KONFIGURASI ====================
ACTIONS = ['Saya', 'Makan', 'Obat', 'Agar', 'Kuat', 'Buah', 'Sayur',
           'Ibu', 'An', 'Sabar', 'Siap', 'Gelas', 'Harus', 'Kakak',
           'Kan', 'Kue', 'Malam', 'Rajin', 'Siram', 'Untuk']
SEQUENCE_LENGTH = 45

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "public", "models", "model20class.keras")

model = None

# ==================== STARTUP EVENT ====================
# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def load_model_on_startup():
    global model
    print(f"[INFO] Memuat model dari: {MODEL_PATH}")

    if os.path.exists(MODEL_PATH):
        try:
            # === PERUBAHAN ADA DI SINI ===
            # Kita gunakan compile=False agar Keras tidak mencoba merekonstruksi
            # optimizer (AdamW) dan custom loss (FocalLossWithSmoothing).
            # Kita tetap harus meregister AttentionLayer karena itu adalah 
            # bagian dari arsitektur jaringannya (layer).
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,  # <-- KUNCI SOLUSINYA
                custom_objects={
                    'AttentionLayer': AttentionLayer,
                    # Hapus FocalLossWithSmoothing dari custom_objects
                    # karena tidak lagi dibutuhkan saat compile=False
                }
            )
            print("[INFO] Model 20 kelas berhasil dimuat.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model: {e}")
    else:
        print(f"[ERROR] File model tidak ditemukan di: {MODEL_PATH}")


# ==================== ENDPOINTS ====================
@app.get("/")
def home():
    return {"message": "Server FastAPI SIBI 20 Kelas aktif."}


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Client (NextJS) terhubung via WebSocket.")

    try:
        while True:
            data_str = await websocket.receive_text()

            if model is None:
                await websocket.send_text(json.dumps(
                    {"error": "Model gagal di-load di server."}
                ))
                continue

            try:
                payload = json.loads(data_str)
                sequence = payload.get("sequence", [])

                if len(sequence) == SEQUENCE_LENGTH:
                    input_seq = np.expand_dims(np.array(sequence), axis=0)
                    res = model.predict(input_seq, verbose=0)[0]

                    best_idx = int(np.argmax(res))
                    confidence = float(res[best_idx])
                    word = ACTIONS[best_idx] if best_idx < len(ACTIONS) else "Unknown"

                    response = {
                        "prediction": word,
                        "confidence": confidence,
                        "probabilities": res.tolist()
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "error": f"Sekuens tidak lengkap: {len(sequence)}/45"
                    }))

            except Exception as e:
                print(f"[ERROR] Prediksi: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        print("[INFO] Client terputus.")
    except Exception as e:
        print(f"[ERROR] WebSocket: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
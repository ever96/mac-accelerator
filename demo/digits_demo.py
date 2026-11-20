"""
Demo de punta a punta: red neuronal 64 -> 32 (ReLU) -> 10 para reconocer dígitos
manuscritos (dataset "digits" de scikit-learn, imágenes de 8x8), cuantizada a int8
y ejecutada en el acelerador simulado (AccelTop) a través del bus Wishbone.

    python demo/digits_demo.py [n_imagenes_en_hardware]

Muestra: exactitud en float, exactitud del modelo entero (NumPy), y verifica que el
hardware dé EXACTAMENTE los mismos resultados que el modelo entero.
"""
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from amaranth.sim import Simulator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from rtl.accel_top import AccelTop
from sw.driver import Accel, WishboneSim
from sw.refmodel import requant

N_HW = int(sys.argv[1]) if len(sys.argv) > 1 else 540     # 540 = todo el set de prueba
CLOCK_MHZ = 50

# ---------------- 1. Entrenar en float ----------------
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=600, random_state=0).fit(Xtr, ytr)
acc_float = mlp.score(Xte, yte)

# ---------------- 2. Cuantizar a int8 ----------------
SX = 1 / 7                                   # píxeles 0..16 -> 0..112
def qx(img): return np.round(img / SX).astype(np.int64)

W1, b1 = mlp.coefs_[0].T, mlp.intercepts_[0]
W2, b2 = mlp.coefs_[1].T, mlp.intercepts_[1]
sw1, sw2 = np.abs(W1).max() / 127, np.abs(W2).max() / 127
W1q, W2q = np.round(W1 / sw1).astype(np.int64), np.round(W2 / sw2).astype(np.int64)
b1q = np.round(b1 / (sw1 * SX)).astype(np.int64)

# Calibrar la escala de la capa oculta (potencia de 2) con los datos de entrenamiento
pre = (qx(Xtr) @ W1q.T + b1q).max()
SHIFT1 = max(0, math.ceil(math.log2(pre / 127)))
SH = sw1 * SX * 2 ** SHIFT1
b2q = np.round(b2 / (sw2 * SH)).astype(np.int64)


def int_model(img):
    h = requant(W1q @ qx(img), b1q, SHIFT1, relu=True)
    return W2q @ h + b2q                     # logits enteros


acc_int = np.mean([int_model(img).argmax() == t for img, t in zip(Xte, yte)])

print(f"Exactitud float32         : {acc_float:.2%}  ({len(yte)} imágenes de prueba)")
print(f"Exactitud int8 (NumPy)    : {acc_int:.2%}  (SHIFT capa oculta = {SHIFT1})")

# ---------------- 3. Ejecutar en el acelerador simulado ----------------
dut = AccelTop(w_depth=1024, x_depth=64, y_depth=64, bias_depth=64)
hw_logits, cycle_log = [], []


async def tb(ctx):
    acc = Accel(WishboneSim(ctx, dut))
    await acc.check_id()
    w2_base = await acc.load_weights(W1q, w_base=0)        # 32 x 16 palabras
    await acc.load_weights(W2q, w_base=w2_base)            # 10 x 8 palabras
    await acc.load_bias(b1q, base=0)
    for img in Xte[:N_HW]:
        await acc.load_input(qx(img))
        c1 = await acc.run(32, 64, SHIFT1, relu=True, w_base=0, bias_base=0)
        h = await acc.read_output(32)
        await acc.load_input(h)
        c2 = await acc.run(10, 32, 0, relu=False, w_base=w2_base, bias_base=0)
        hw_logits.append(await acc.read_acc(10) + b2q)
        cycle_log.append(c1 + c2)


t0 = time.time()
sim = Simulator(dut)
sim.add_clock(1e-9 * 1000 / CLOCK_MHZ)
sim.add_testbench(tb)
sim.run()

ref = [int_model(img) for img in Xte[:N_HW]]
exact = all(np.array_equal(a, b) for a, b in zip(hw_logits, ref))
acc_hw = np.mean([l.argmax() == t for l, t in zip(hw_logits, yte[:N_HW])])
cyc = int(np.mean(cycle_log))
macs = 64 * 32 + 32 * 10

print(f"Hardware == modelo int8   : {'SÍ, bit a bit' if exact else 'NO'} ({N_HW} imágenes)")
print(f"Exactitud en hardware     : {acc_hw:.2%}  ({N_HW} imágenes)")
print(f"Ciclos de cómputo/imagen  : {cyc}  ({macs} MAC -> {macs / cyc:.2f} MAC/ciclo)")
print(f"A {CLOCK_MHZ} MHz                  : {cyc / CLOCK_MHZ:.1f} µs por inferencia "
      f"(sin contar la carga de datos por el bus)")
print(f"(simulación: {time.time() - t0:.1f} s)")
assert exact

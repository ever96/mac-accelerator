"""
DotUnit: unidad de producto punto con signo para inferencia de IA.

  - N carriles (lanes) de 8 bits en paralelo + árbol de sumas.
  - Modo int8: cada carril multiplica un par int8      -> N   MAC/ciclo.
  - Modo int4: cada carril multiplica dos pares int4    -> 2N  MAC/ciclo.
  - Acumulador de 32 bits (no desborda en vectores largos).
  - Pipeline: [multiplicar] -> [árbol + acumular] -> [salida registrada].
  - Streaming: in_valid / in_first (reinicia acumulador) / in_last (emite).
  - Post-proceso: bias + desplazamiento con redondeo + ReLU + saturación int8.

Latencia: out_valid sube 2 ciclos después del ciclo con in_valid & in_last.
"""
from amaranth import *


class DotUnit(Elaboratable):
    def __init__(self, lanes=4, acc_width=32):
        self.lanes, self.acc_width = lanes, acc_width
        W = 8

        # Entrada (streaming)
        self.a = Signal(lanes * W)
        self.b = Signal(lanes * W)
        self.in_valid = Signal()
        self.in_first = Signal()
        self.in_last = Signal()

        # Configuración
        self.int4 = Signal()                     # 0 = int8, 1 = int4 x2 empaquetado
        self.bias = Signal(signed(acc_width))
        self.shift = Signal(5)                   # escala de salida = 2^-shift
        self.relu = Signal()

        # Salida
        self.out_valid = Signal()
        self.out_acc = Signal(signed(acc_width))
        self.out_q = Signal(signed(8))

    def ports(self):
        return [self.a, self.b, self.in_valid, self.in_first, self.in_last,
                self.int4, self.bias, self.shift, self.relu,
                self.out_valid, self.out_acc, self.out_q]

    def elaborate(self, platform):
        m = Module()
        N = self.lanes

        # ---------- Etapa 1: multiplicadores ----------
        prods = []
        for i in range(N):
            a8 = self.a[i * 8:(i + 1) * 8]
            b8 = self.b[i * 8:(i + 1) * 8]
            p8 = a8.as_signed() * b8.as_signed()
            p4 = (a8[0:4].as_signed() * b8[0:4].as_signed() +
                  a8[4:8].as_signed() * b8[4:8].as_signed())
            p = Signal(signed(16), name=f"prod{i}")
            m.d.sync += p.eq(Mux(self.int4, p4, p8))
            prods.append(p)

        v1, f1, l1 = Signal(), Signal(), Signal()
        m.d.sync += [v1.eq(self.in_valid), f1.eq(self.in_first), l1.eq(self.in_last)]

        # ---------- Etapa 2: árbol de sumas + acumulación ----------
        level = prods
        while len(level) > 1:
            nxt = [level[j] + level[j + 1] for j in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        tree_sum = level[0]

        acc = Signal(signed(self.acc_width))
        new_acc = Signal(signed(self.acc_width))
        m.d.comb += new_acc.eq(Mux(f1, 0, acc) + tree_sum)
        with m.If(v1):
            m.d.sync += acc.eq(new_acc)

        # ---------- Etapa 3: salida + re-cuantización ----------
        m.d.sync += self.out_valid.eq(v1 & l1)
        with m.If(v1 & l1):
            m.d.sync += self.out_acc.eq(new_acc)

        Wx = self.acc_width + 2
        biased = Signal(signed(Wx))
        m.d.comb += biased.eq(self.out_acc + self.bias)
        half = (Const(1, Wx) << self.shift) >> 1          # redondeo; 0 si shift == 0
        scaled = Signal(signed(Wx))
        m.d.comb += scaled.eq((biased + half) >> self.shift)
        act = Signal(signed(Wx))
        m.d.comb += act.eq(Mux(self.relu & (scaled < 0), 0, scaled))
        with m.If(act > 127):
            m.d.comb += self.out_q.eq(127)
        with m.Elif(act < -128):
            m.d.comb += self.out_q.eq(-128)
        with m.Else():
            m.d.comb += self.out_q.eq(act[:8])

        return m

"""
SystolicArray: arreglo sistólico NxN "output-stationary" para GEMM int8.

  C[NxN] = A[NxK] @ B[KxN]

Cada ciclo k el host entrega una columna de A (A[:, k]) y una fila de B (B[k, :]).
El arreglo aplica internamente el desfase ("skew") de entrada: la fila i de A se
retrasa i ciclos y la columna j de B se retrasa j ciclos. Cada PE(i, j) acumula
A[i, k] * B[k, j]; los datos de A fluyen hacia la derecha y los de B hacia abajo,
de modo que cada operando que entra se reutiliza N veces (ese es el punto: menos
ancho de banda de memoria por operación).

N^2 MAC por ciclo. Resultado listo DRAIN = 2N - 1 ciclos después de la última
entrada (ver atributo `drain_cycles`). `clear` pone todos los acumuladores en 0.
"""
from amaranth import *


class PE(Elaboratable):
    def __init__(self, acc_width):
        self.a_in = Signal(signed(8)); self.b_in = Signal(signed(8)); self.v_in = Signal()
        self.a_out = Signal(signed(8)); self.b_out = Signal(signed(8)); self.v_out = Signal()
        self.clear = Signal()
        self.acc = Signal(signed(acc_width))

    def elaborate(self, platform):
        m = Module()
        m.d.sync += [self.a_out.eq(self.a_in), self.b_out.eq(self.b_in),
                     self.v_out.eq(self.v_in)]
        with m.If(self.clear):
            m.d.sync += self.acc.eq(0)
        with m.Elif(self.v_in):
            m.d.sync += self.acc.eq(self.acc + self.a_in * self.b_in)
        return m


class SystolicArray(Elaboratable):
    def __init__(self, n=4, acc_width=32):
        self.n, self.acc_width = n, acc_width
        self.drain_cycles = 2 * n - 1
        self.a_col = Signal(8 * n)           # A[:, k] empaquetado, fila 0 en bits bajos
        self.b_row = Signal(8 * n)           # B[k, :] empaquetado, columna 0 en bits bajos
        self.in_valid = Signal()
        self.clear = Signal()
        self.sel = Signal(range(n * n))      # índice i*n + j del resultado a leer
        self.c_out = Signal(signed(acc_width))

    def ports(self):
        return [self.a_col, self.b_row, self.in_valid, self.clear, self.sel, self.c_out]

    def elaborate(self, platform):
        m = Module()
        n = self.n
        pes = [[PE(self.acc_width) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                m.submodules[f"pe_{i}_{j}"] = pes[i][j]
                m.d.comb += pes[i][j].clear.eq(self.clear)

        def delay(sig, cycles, name):
            for c in range(cycles):
                r = Signal.like(sig, name=f"{name}_d{c}")
                m.d.sync += r.eq(sig)
                sig = r
            return sig

        # Skew de entrada + conexiones de borde
        for i in range(n):
            a_i = delay(self.a_col[8 * i:8 * (i + 1)].as_signed(), i, f"a{i}")
            v_i = delay(self.in_valid, i, f"v{i}")
            m.d.comb += [pes[i][0].a_in.eq(a_i), pes[i][0].v_in.eq(v_i)]
        for j in range(n):
            b_j = delay(self.b_row[8 * j:8 * (j + 1)].as_signed(), j, f"b{j}")
            m.d.comb += pes[0][j].b_in.eq(b_j)

        # Malla: A fluye a la derecha (con valid), B fluye hacia abajo
        for i in range(n):
            for j in range(n):
                if j + 1 < n:
                    m.d.comb += [pes[i][j + 1].a_in.eq(pes[i][j].a_out),
                                 pes[i][j + 1].v_in.eq(pes[i][j].v_out)]
                if i + 1 < n:
                    m.d.comb += pes[i + 1][j].b_in.eq(pes[i][j].b_out)

        # Lectura de resultados
        with m.Switch(self.sel):
            for i in range(n):
                for j in range(n):
                    with m.Case(i * n + j):
                        m.d.comb += self.c_out.eq(pes[i][j].acc)
        return m

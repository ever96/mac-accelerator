"""
AccelTop (Core-02): acelerador GEMV int8/int4 con bus Wishbone.

Calcula, para una capa de red neuronal densa:
    y[r] = requant( sum_k W[r, k] * x[k] + bias[r] )     r = 0 .. ROWS-1

Bloques:
  - Esclavo Wishbone clásico de 32 bits (1 estado de espera, sólo escrituras de
    palabra completa; wb_sel se ignora).
  - Registros de control/estado (CSR) y contador de ciclos.
  - Memorias en chip: pesos (W), entrada (X), bias, salida int8 (Y) y salida int32 (YACC).
    W se lee secuencialmente y se reutiliza fila por fila; X se reutiliza en cada fila.
  - FSM que recorre las filas alimentando un DotUnit de 4 carriles
    (4 MAC/ciclo en int8, 8 MAC/ciclo en int4).

Mapa de memoria (direcciones de PALABRA; la dirección en bytes es x4):
  0x000  CTRL      W: bit0 = start          R: bit0 = busy, bit1 = done
  0x001  ROWS      número de filas (salidas)
  0x002  KWORDS    palabras de 32 bits por fila (K/4 en int8, K/8 en int4)
  0x003  SHIFT     escala de salida 2^-SHIFT
  0x004  FLAGS     bit0 = ReLU, bit1 = modo int4
  0x005  W_BASE    palabra inicial de la matriz en la memoria W
  0x006  BIAS_BASE índice inicial en la memoria de bias
  0x007  CYCLES    R: ciclos de la última ejecución
  0x008  ID        R: 0x4D414302 ("MAC" v2)
  0x100+ BIAS      int32, una entrada por fila
  0x200+ X         vector de entrada empaquetado (4 x int8 u 8 x int4 por palabra)
  0x300+ Y         R: salida int8 re-cuantizada (extendida con signo a 32 bits)
  0x400+ YACC      R: acumulador int32 crudo (sin bias)
  0x800+ W         pesos, fila por fila, KWORDS palabras por fila
"""
from amaranth import *
from amaranth.lib.memory import Memory

from rtl.dot_unit import DotUnit

ACCEL_ID = 0x4D414302

CSR_CTRL, CSR_ROWS, CSR_KWORDS, CSR_SHIFT, CSR_FLAGS = 0, 1, 2, 3, 4
CSR_W_BASE, CSR_BIAS_BASE, CSR_CYCLES, CSR_ID = 5, 6, 7, 8
REG_BIAS, REG_X, REG_Y, REG_YACC, REG_W = 0x100, 0x200, 0x300, 0x400, 0x800


class AccelTop(Elaboratable):
    def __init__(self, w_depth=1024, x_depth=64, y_depth=64, bias_depth=64):
        assert w_depth <= 2048 and max(x_depth, y_depth, bias_depth) <= 256
        self.w_depth, self.x_depth = w_depth, x_depth
        self.y_depth, self.bias_depth = y_depth, bias_depth

        self.wb_cyc = Signal()
        self.wb_stb = Signal()
        self.wb_we = Signal()
        self.wb_adr = Signal(12)
        self.wb_dat_w = Signal(32)
        self.wb_sel = Signal(4)
        self.wb_dat_r = Signal(32)
        self.wb_ack = Signal()
        self.irq = Signal()                      # nivel alto mientras done = 1

    def ports(self):
        return [self.wb_cyc, self.wb_stb, self.wb_we, self.wb_adr, self.wb_dat_w,
                self.wb_sel, self.wb_dat_r, self.wb_ack, self.irq]

    def elaborate(self, platform):
        m = Module()
        m.submodules.dot = dot = DotUnit(lanes=4)

        # ---------------- Memorias ----------------
        m.submodules.wmem = wmem = Memory(shape=32, depth=self.w_depth, init=[])
        m.submodules.xmem = xmem = Memory(shape=32, depth=self.x_depth, init=[])
        m.submodules.bmem = bmem = Memory(shape=32, depth=self.bias_depth, init=[])
        m.submodules.ymem = ymem = Memory(shape=8, depth=self.y_depth, init=[])
        m.submodules.amem = amem = Memory(shape=32, depth=self.y_depth, init=[])
        w_wr, w_rd = wmem.write_port(), wmem.read_port()
        x_wr, x_rd = xmem.write_port(), xmem.read_port()
        b_wr, b_rd = bmem.write_port(), bmem.read_port()
        y_wr, y_rd = ymem.write_port(), ymem.read_port()
        a_wr, a_rd = amem.write_port(), amem.read_port()

        # ---------------- CSR ----------------
        rows = Signal(9); kwords = Signal(12); shift = Signal(5)
        relu = Signal(); int4 = Signal()
        w_base = Signal(11); bias_base = Signal(8)
        cycles = Signal(32)
        busy = Signal(); done = Signal()
        start = Signal()
        m.d.comb += self.irq.eq(done)

        # ---------------- Wishbone ----------------
        adr = self.wb_adr
        req = self.wb_cyc & self.wb_stb & ~self.wb_ack
        is_w = adr[11]
        region = Signal(4)
        m.d.comb += region.eq(Mux(is_w, 8, adr[8:12]))
        low = adr[0:8]

        m.d.sync += self.wb_ack.eq(req)
        region_r = Signal(4)
        csr_r = Signal(32)
        with m.If(req):
            m.d.sync += region_r.eq(region)

        # Lectura de CSR (registrada) -- se presenta en el ciclo de ack
        with m.If(req & (region == 0)):
            with m.Switch(low):
                with m.Case(CSR_CTRL):      m.d.sync += csr_r.eq(Cat(busy, done))
                with m.Case(CSR_ROWS):      m.d.sync += csr_r.eq(rows)
                with m.Case(CSR_KWORDS):    m.d.sync += csr_r.eq(kwords)
                with m.Case(CSR_SHIFT):     m.d.sync += csr_r.eq(shift)
                with m.Case(CSR_FLAGS):     m.d.sync += csr_r.eq(Cat(relu, int4))
                with m.Case(CSR_W_BASE):    m.d.sync += csr_r.eq(w_base)
                with m.Case(CSR_BIAS_BASE): m.d.sync += csr_r.eq(bias_base)
                with m.Case(CSR_CYCLES):    m.d.sync += csr_r.eq(cycles)
                with m.Case(CSR_ID):        m.d.sync += csr_r.eq(ACCEL_ID)
                with m.Default():           m.d.sync += csr_r.eq(0)

        # Lectura de memorias de salida (puertos síncronos: dato listo en el ack)
        m.d.comb += [y_rd.addr.eq(low), a_rd.addr.eq(low)]
        with m.Switch(region_r):
            with m.Case(0): m.d.comb += self.wb_dat_r.eq(csr_r)
            with m.Case(3): m.d.comb += self.wb_dat_r.eq(y_rd.data.as_signed())
            with m.Case(4): m.d.comb += self.wb_dat_r.eq(a_rd.data)
            with m.Default(): m.d.comb += self.wb_dat_r.eq(0)

        # Escrituras
        wr = req & self.wb_we
        dat = self.wb_dat_w
        m.d.comb += [
            w_wr.addr.eq(adr[0:11]), w_wr.data.eq(dat), w_wr.en.eq(wr & (region == 8)),
            x_wr.addr.eq(low), x_wr.data.eq(dat), x_wr.en.eq(wr & (region == 2)),
            b_wr.addr.eq(low), b_wr.data.eq(dat), b_wr.en.eq(wr & (region == 1)),
        ]
        with m.If(wr & (region == 0)):
            with m.Switch(low):
                with m.Case(CSR_CTRL):      m.d.comb += start.eq(dat[0])
                with m.Case(CSR_ROWS):      m.d.sync += rows.eq(dat)
                with m.Case(CSR_KWORDS):    m.d.sync += kwords.eq(dat)
                with m.Case(CSR_SHIFT):     m.d.sync += shift.eq(dat)
                with m.Case(CSR_FLAGS):     m.d.sync += [relu.eq(dat[0]), int4.eq(dat[1])]
                with m.Case(CSR_W_BASE):    m.d.sync += w_base.eq(dat)
                with m.Case(CSR_BIAS_BASE): m.d.sync += bias_base.eq(dat)

        # ---------------- Motor GEMV ----------------
        row = Signal(9)
        k = Signal(12)
        waddr = Signal(11)
        p_valid = Signal(); p_first = Signal(); p_last = Signal()

        m.d.comb += [
            w_rd.addr.eq(waddr), x_rd.addr.eq(k), b_rd.addr.eq(bias_base + row),
            dot.a.eq(w_rd.data), dot.b.eq(x_rd.data),
            dot.in_valid.eq(p_valid), dot.in_first.eq(p_first), dot.in_last.eq(p_last),
            dot.int4.eq(int4), dot.relu.eq(relu), dot.shift.eq(shift),
            dot.bias.eq(b_rd.data),
        ]
        m.d.sync += p_valid.eq(0)

        with m.If(busy):
            m.d.sync += cycles.eq(cycles + 1)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(start):
                    m.d.sync += [busy.eq(1), done.eq(0), cycles.eq(0),
                                 row.eq(0), k.eq(0), waddr.eq(w_base)]
                    with m.If((rows == 0) | (kwords == 0)):
                        m.d.sync += [busy.eq(0), done.eq(1)]
                    with m.Else():
                        m.next = "FEED"
            with m.State("FEED"):
                m.d.sync += [p_valid.eq(1), p_first.eq(k == 0),
                             p_last.eq(k == kwords - 1),
                             waddr.eq(waddr + 1), k.eq(k + 1)]
                with m.If(k == kwords - 1):
                    m.d.sync += k.eq(0)
                    m.next = "WAIT"
            with m.State("WAIT"):
                with m.If(dot.out_valid):
                    m.d.sync += row.eq(row + 1)
                    with m.If(row == rows - 1):
                        m.d.sync += [busy.eq(0), done.eq(1)]
                        m.next = "IDLE"
                    with m.Else():
                        m.next = "FEED"

        m.d.comb += [
            y_wr.addr.eq(row), y_wr.data.eq(dot.out_q), a_wr.addr.eq(row),
            a_wr.data.eq(dot.out_acc),
            y_wr.en.eq(dot.out_valid & busy), a_wr.en.eq(dot.out_valid & busy),
        ]
        return m

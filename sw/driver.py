"""
Driver de alto nivel del acelerador.

Funciona sobre cualquier "bus" con métodos async `read(word_addr)` y
`write(word_addr, value)`: en simulación se usa WishboneSim; en hardware real se
reemplaza por un puente (por ejemplo, SPI/UART -> Wishbone, o el bus de Caravel).
"""
import numpy as np

from rtl.accel_top import (ACCEL_ID, CSR_CTRL, CSR_ROWS, CSR_KWORDS, CSR_SHIFT,
                           CSR_FLAGS, CSR_W_BASE, CSR_BIAS_BASE, CSR_CYCLES, CSR_ID,
                           REG_BIAS, REG_X, REG_Y, REG_YACC, REG_W)
from sw.refmodel import pack_int8, pack_int4, to_signed


class WishboneSim:
    """Maestro Wishbone para el simulador de Amaranth."""

    def __init__(self, ctx, dut):
        self.ctx, self.dut = ctx, dut

    async def _cycle(self, addr, we, data=0):
        c, d = self.ctx, self.dut
        c.set(d.wb_cyc, 1); c.set(d.wb_stb, 1); c.set(d.wb_we, we)
        c.set(d.wb_adr, addr); c.set(d.wb_dat_w, data & 0xFFFFFFFF); c.set(d.wb_sel, 0xF)
        await c.tick()
        while not c.get(d.wb_ack):
            await c.tick()
        value = c.get(d.wb_dat_r)
        c.set(d.wb_cyc, 0); c.set(d.wb_stb, 0); c.set(d.wb_we, 0)
        await c.tick()
        return value

    async def write(self, addr, value):
        await self._cycle(addr, 1, value)

    async def read(self, addr):
        return await self._cycle(addr, 0)


class Accel:
    def __init__(self, bus):
        self.bus = bus

    async def check_id(self):
        ident = await self.bus.read(CSR_ID)
        assert ident == ACCEL_ID, hex(ident)

    @staticmethod
    def _pack(vals, int4):
        return pack_int4(vals) if int4 else pack_int8(vals)

    async def load_weights(self, W, w_base=0, int4=False):
        """Carga una matriz [ROWS x K] fila por fila a partir de w_base."""
        addr = REG_W + w_base
        for row in W:
            for word in self._pack(row, int4):
                await self.bus.write(addr, word)
                addr += 1
        return addr - REG_W              # siguiente w_base libre

    async def load_bias(self, bias, base=0):
        for i, b in enumerate(bias):
            await self.bus.write(REG_BIAS + base + i, int(b))

    async def load_input(self, x, int4=False):
        for i, word in enumerate(self._pack(x, int4)):
            await self.bus.write(REG_X + i, word)

    async def run(self, rows, k, shift, relu, int4=False, w_base=0, bias_base=0):
        per = 8 if int4 else 4
        assert k % per == 0
        await self.bus.write(CSR_ROWS, rows)
        await self.bus.write(CSR_KWORDS, k // per)
        await self.bus.write(CSR_SHIFT, shift)
        await self.bus.write(CSR_FLAGS, int(relu) | (int(int4) << 1))
        await self.bus.write(CSR_W_BASE, w_base)
        await self.bus.write(CSR_BIAS_BASE, bias_base)
        await self.bus.write(CSR_CTRL, 1)
        while (await self.bus.read(CSR_CTRL)) & 1:
            pass
        return await self.bus.read(CSR_CYCLES)

    async def read_output(self, rows):
        y = [to_signed(await self.bus.read(REG_Y + i), 32) for i in range(rows)]
        return np.array(y, dtype=np.int64)

    async def read_acc(self, rows):
        a = [to_signed(await self.bus.read(REG_YACC + i), 32) for i in range(rows)]
        return np.array(a, dtype=np.int64)

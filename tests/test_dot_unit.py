"""DotUnit vs modelo de referencia: modos int8 e int4, vectores de 4 a 256 elementos."""
import numpy as np
from amaranth.sim import Simulator

from rtl.dot_unit import DotUnit
from sw.refmodel import pack_int8, pack_int4, requant

N = 4


def run_cases(cases):
    dut = DotUnit(lanes=N)
    results = []

    async def tb(ctx):
        for a, b, bias, shift, relu, int4 in cases:
            pack = pack_int4 if int4 else pack_int8
            wa, wb = pack(a), pack(b)
            ctx.set(dut.bias, bias); ctx.set(dut.shift, shift)
            ctx.set(dut.relu, relu); ctx.set(dut.int4, int4)
            for k, (x, y) in enumerate(zip(wa, wb)):
                ctx.set(dut.a, x); ctx.set(dut.b, y)
                ctx.set(dut.in_valid, 1)
                ctx.set(dut.in_first, k == 0)
                ctx.set(dut.in_last, k == len(wa) - 1)
                await ctx.tick()
            ctx.set(dut.in_valid, 0)
            while not ctx.get(dut.out_valid):
                await ctx.tick()
            results.append((ctx.get(dut.out_acc), ctx.get(dut.out_q)))
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-8)
    sim.add_testbench(tb)
    sim.run()
    return results


def make_cases(seed, int4):
    rng = np.random.default_rng(seed)
    lo, hi, step = (-8, 8, 8) if int4 else (-128, 128, 4)
    cases = []
    for _ in range(30):
        n = step * int(rng.integers(1, 33))
        cases.append((rng.integers(lo, hi, n), rng.integers(lo, hi, n),
                      int(rng.integers(-5000, 5000)), int(rng.integers(0, 16)),
                      int(rng.integers(0, 2)), int4))
    ext = lo  # peor caso: todos los productos al máximo positivo
    cases.append((np.full(256, ext), np.full(256, ext), 0, 15, 0, int4))
    return cases


def check(cases, results):
    assert len(results) == len(cases)
    for (a, b, bias, shift, relu, _), (acc, q) in zip(cases, results):
        g_acc = int(np.dot(a.astype(np.int64), b.astype(np.int64)))
        g_q = int(requant(g_acc, bias, shift, relu))
        assert (acc, q) == (g_acc, g_q), (len(a), bias, shift, relu, acc, q, g_acc, g_q)


def test_int8():
    cases = make_cases(0, int4=False)
    check(cases, run_cases(cases))


def test_int4():
    cases = make_cases(1, int4=True)
    check(cases, run_cases(cases))

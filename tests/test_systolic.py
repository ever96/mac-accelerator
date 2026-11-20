"""SystolicArray vs NumPy: GEMM int8 con K variable, incluido el peor caso."""
import numpy as np
from amaranth.sim import Simulator

from rtl.systolic import SystolicArray
from sw.refmodel import pack_int8, to_signed

N = 4


def run_gemm(pairs):
    dut = SystolicArray(n=N)
    out = []

    async def tb(ctx):
        for A, B in pairs:
            ctx.set(dut.clear, 1); await ctx.tick(); ctx.set(dut.clear, 0)
            for k in range(A.shape[1]):
                ctx.set(dut.a_col, pack_int8(A[:, k])[0])
                ctx.set(dut.b_row, pack_int8(B[k, :])[0])
                ctx.set(dut.in_valid, 1)
                await ctx.tick()
            ctx.set(dut.in_valid, 0)
            await ctx.tick().repeat(dut.drain_cycles)
            C = np.zeros((N, N), dtype=np.int64)
            for idx in range(N * N):
                ctx.set(dut.sel, idx)
                C[idx // N, idx % N] = ctx.get(dut.c_out)
            out.append(C)

    sim = Simulator(dut)
    sim.add_clock(1e-8)
    sim.add_testbench(tb)
    sim.run()
    return out


def test_gemm_random():
    rng = np.random.default_rng(2)
    pairs = []
    for _ in range(20):
        K = int(rng.integers(1, 65))
        pairs.append((rng.integers(-128, 128, (N, K)), rng.integers(-128, 128, (K, N))))
    pairs.append((np.full((N, 256), -128), np.full((256, N), -128)))
    results = run_gemm(pairs)
    assert len(results) == len(pairs)
    for (A, B), C in zip(pairs, results):
        np.testing.assert_array_equal(C, A.astype(np.int64) @ B.astype(np.int64))

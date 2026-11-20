"""AccelTop completo vía Wishbone vs modelo de referencia (int8 e int4, varias capas)."""
import numpy as np
from amaranth.sim import Simulator

from rtl.accel_top import AccelTop, CSR_CTRL
from sw.driver import Accel, WishboneSim
from sw.refmodel import gemv


def simulate(testbench, **kw):
    dut = AccelTop(**kw)

    async def tb(ctx):
        await testbench(Accel(WishboneSim(ctx, dut)), ctx, dut)

    sim = Simulator(dut)
    sim.add_clock(1e-8)
    sim.add_testbench(tb)
    sim.run()


def test_gemv_layers():
    rng = np.random.default_rng(3)
    jobs = []
    for int4 in (False, True):
        lo, hi, per = (-8, 8, 8) if int4 else (-128, 128, 4)
        for _ in range(4):
            rows = int(rng.integers(1, 17))
            k = per * int(rng.integers(1, 9))
            jobs.append(dict(W=rng.integers(lo, hi, (rows, k)), x=rng.integers(lo, hi, k),
                             bias=rng.integers(-3000, 3000, rows),
                             shift=int(rng.integers(0, 12)), relu=bool(rng.integers(0, 2)),
                             int4=int4))
    got = []

    async def tb(acc, ctx, dut):
        await acc.check_id()
        w_base, b_base = 0, 0
        for j in jobs:          # todas las capas residentes a la vez en memoria
            j["w_base"], j["b_base"] = w_base, b_base
            w_base = await acc.load_weights(j["W"], w_base, j["int4"])
            await acc.load_bias(j["bias"], b_base)
            b_base += len(j["bias"])
        for j in jobs:
            await acc.load_input(j["x"], j["int4"])
            rows, k = j["W"].shape
            cycles = await acc.run(rows, k, j["shift"], j["relu"], j["int4"],
                                   j["w_base"], j["b_base"])
            got.append((await acc.read_acc(rows), await acc.read_output(rows), cycles))

    total_w = sum(j["W"].size // (8 if j["int4"] else 4) for j in jobs)
    total_b = sum(len(j["bias"]) for j in jobs)
    assert total_w <= 1024 and total_b <= 256          # todo cabe sin sobrescribir
    simulate(tb, w_depth=1024, bias_depth=256)
    assert len(got) == len(jobs)
    for j, (acc, y, cycles) in zip(jobs, got):
        g_acc, g_y = gemv(j["W"], j["x"], j["bias"], j["shift"], j["relu"])
        np.testing.assert_array_equal(acc, g_acc)
        np.testing.assert_array_equal(y, g_y)
        assert cycles > 0


def test_ctrl_done_and_zero_rows():
    seen = {}

    async def tb(acc, ctx, dut):
        assert await acc.bus.read(CSR_CTRL) == 0           # ni busy ni done al reset
        await acc.run(rows=0, k=4, shift=0, relu=False)
        seen["ctrl"] = await acc.bus.read(CSR_CTRL)
        seen["irq"] = ctx.get(dut.irq)

    simulate(tb)
    assert seen == {"ctrl": 2, "irq": 1}

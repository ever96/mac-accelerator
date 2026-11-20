"""
Genera el Verilog de todos los bloques en build/ y un reporte de síntesis con Yosys.

    python build.py            # Verilog + síntesis
    python build.py --no-synth # sólo Verilog

Presets de AccelTop:
  - accel_top.v        : versión "tapeout" con memorias chicas (en flip-flops)
  - accel_top_demo.v   : versión de la demo (memorias grandes, sólo simulación/FPGA)
"""
import shutil
import subprocess
import sys
from pathlib import Path

from amaranth.back import verilog

from rtl.accel_top import AccelTop
from rtl.dot_unit import DotUnit
from rtl.systolic import SystolicArray

BUILD = Path(__file__).parent / "build"

DESIGNS = {
    "dot_unit": lambda: DotUnit(lanes=4),
    "systolic4x4": lambda: SystolicArray(n=4),
    "accel_top": lambda: AccelTop(w_depth=128, x_depth=16, y_depth=16, bias_depth=16),
    "accel_top_demo": lambda: AccelTop(w_depth=1024, x_depth=64, y_depth=64, bias_depth=64),
}


def yosys_cmd():
    for name in ("yosys", "yowasp-yosys"):
        if shutil.which(name):
            return name
    return None


def main():
    BUILD.mkdir(exist_ok=True)
    for name, make in DESIGNS.items():
        dut = make()
        text = verilog.convert(dut, name=name, ports=dut.ports(), emit_src=False)
        (BUILD / f"{name}.v").write_text(text)
        print(f"build/{name}.v")

    if "--no-synth" in sys.argv:
        return
    yosys = yosys_cmd()
    if not yosys:
        print("Yosys no encontrado: pip install yowasp-yosys")
        return
    report = ["Síntesis genérica (Yosys, sin mapeo a celdas Sky130).",
              "Sirve para comparar bloques; el área real la da OpenLane.", ""]
    for name in DESIGNS:
        if name == "accel_top_demo":
            continue
        out = subprocess.run(
            [yosys, "-q", "-p",
             f"read_verilog {BUILD / name}.v; synth -top {name} -flatten -noabc; "
             f"tee -o {BUILD / name}.stat stat"],
            capture_output=True, text=True)
        stat = (BUILD / f"{name}.stat").read_text() if out.returncode == 0 else out.stderr
        cells = ffs = 0
        for line in stat.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1] == "cells" and parts[0].isdigit():
                cells = int(parts[0])
            elif len(parts) == 2 and "DFF" in parts[1] and parts[0].isdigit():
                ffs += int(parts[0])
        report.append(f"{name:14s} {cells:7d} celdas genéricas  {ffs:6d} flip-flops")
    (BUILD / "synth_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()

#  Vector MAC Accelerator (Core-01)
Primer diseño físico de un circuito integrado digital procesado bajo un flujo industrial estándar (SkyWater 130nm) en Paraguay.

## 📐 Especificaciones Técnicas
- **Arquitectura:** Unidad Multiplicadora-Acumuladora (MAC) para procesamiento de vectores y *embeddings*.
- **Datapath:** Operandos de 8-bit con acumulador de 16-bit.
- **RTL:** Amaranth HDL (Python).
- **Síntesis:** Yosys Open Synthesis Suite.
- **Place & Route:** OpenLane (SkyWater 130nm).
- **Área del Core:** 150 µm × 150 µm.
- **Validación:** 100% limpio de violaciones DRC (Magic / KLayout) y STA verificado.

## 📂 Estructura del Repositorio
- `mac_unit.py`: Código fuente del hardware en Python.
- `mac_unit.v`: RTL generado en Verilog.
- `config.json`: Parámetros de diseño físico para OpenLane.
- `sintesis.ys`: Script de síntesis lógica para Yosys."# mac-accelerator" 

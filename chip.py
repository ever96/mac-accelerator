from amaranth import *
from amaranth.back import verilog

class MACUnit(Elaboratable):
    def __init__(self, width=8):
        # Definimos los 'pines' de entrada y salida del chip (Signals)
        self.vector_a = Signal(width)
        self.vector_b = Signal(width)
        self.accum = Signal(width * 2) # El doble de ancho para evitar desbordamientos
        self.start = Signal()          # Pin para iniciar el cálculo

    def elaborate(self, platform):
        m = Module()
        
        # Dominio síncrono (m.d.sync): Esto ocurre físicamente 
        # cada vez que el reloj del procesador hace "tic".
        with m.If(self.start):
            m.d.sync += self.accum.eq(self.accum + (self.vector_a * self.vector_b))
            
        return m

# === COMPILACIÓN A HARDWARE ===
if __name__ == "__main__":
    mac = MACUnit()
    # Convertimos la clase de Python al estándar industrial de hardware (Verilog)
    with open("mac_unit.v", "w") as f:
        f.write(verilog.convert(mac, ports=[mac.vector_a, mac.vector_b, mac.accum, mac.start]))
    
    print("✅ Hardware compilado: Archivo mac_unit.v generado exitosamente.")
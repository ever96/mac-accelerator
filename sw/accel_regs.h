/*
 * accel_regs.h -- mapa de registros de Core-02 (AccelTop) para firmware en C.
 *
 * ACCEL_BASE es la dirección en BYTES donde el acelerador está conectado al bus
 * Wishbone del sistema (por ejemplo, en Caravel el área de usuario 0x30000000).
 * Todos los registros son de 32 bits; sólo se admiten accesos de palabra completa.
 */
#ifndef ACCEL_REGS_H
#define ACCEL_REGS_H

#include <stdint.h>

#ifndef ACCEL_BASE
#define ACCEL_BASE 0x30000000u
#endif

#define ACCEL_REG(word)     (*(volatile uint32_t *)(ACCEL_BASE + 4u * (word)))

/* Registros de control */
#define ACCEL_CTRL          ACCEL_REG(0x000)   /* W: bit0 start | R: bit0 busy, bit1 done */
#define ACCEL_ROWS          ACCEL_REG(0x001)
#define ACCEL_KWORDS        ACCEL_REG(0x002)
#define ACCEL_SHIFT         ACCEL_REG(0x003)
#define ACCEL_FLAGS         ACCEL_REG(0x004)   /* bit0 ReLU, bit1 int4 */
#define ACCEL_W_BASE        ACCEL_REG(0x005)
#define ACCEL_BIAS_BASE     ACCEL_REG(0x006)
#define ACCEL_CYCLES        ACCEL_REG(0x007)
#define ACCEL_ID            ACCEL_REG(0x008)   /* 0x4D414302 */

#define ACCEL_CTRL_BUSY     (1u << 0)
#define ACCEL_CTRL_DONE     (1u << 1)
#define ACCEL_FLAG_RELU     (1u << 0)
#define ACCEL_FLAG_INT4     (1u << 1)
#define ACCEL_ID_VALUE      0x4D414302u

/* Memorias (índice de palabra) */
#define ACCEL_BIAS(i)       ACCEL_REG(0x100 + (i))
#define ACCEL_X(i)          ACCEL_REG(0x200 + (i))
#define ACCEL_Y(i)          ((int32_t)ACCEL_REG(0x300 + (i)))   /* int8 con signo */
#define ACCEL_YACC(i)       ((int32_t)ACCEL_REG(0x400 + (i)))   /* int32 sin bias */
#define ACCEL_W(i)          ACCEL_REG(0x800 + (i))

/* Ejecuta una capa densa ya cargada. k_words = K/4 (int8) o K/8 (int4). */
static inline uint32_t accel_run(uint32_t rows, uint32_t k_words, uint32_t shift,
                                 uint32_t flags, uint32_t w_base, uint32_t bias_base)
{
    ACCEL_ROWS = rows;
    ACCEL_KWORDS = k_words;
    ACCEL_SHIFT = shift;
    ACCEL_FLAGS = flags;
    ACCEL_W_BASE = w_base;
    ACCEL_BIAS_BASE = bias_base;
    ACCEL_CTRL = 1u;
    while (ACCEL_CTRL & ACCEL_CTRL_BUSY) { }
    return ACCEL_CYCLES;
}

#endif /* ACCEL_REGS_H */

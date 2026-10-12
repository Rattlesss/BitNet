/* Checks a NEON I2_S dot product against the scalar reference, then times both.
 * Build: clang -O3 -mcpu=native+dotprod+i8mm scripts/test_i2s_neon.c -o /tmp/test_i2s_neon
 * The timing uses a weight matrix that mostly fits in cache, so it measures
 * compute speed, not DRAM bandwidth. */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t rng = 88172645463325252ULL;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

__attribute__((noinline))
static void ref_dot(int n, float * s, size_t bx, const uint8_t * x, const int8_t * y, int nrc) {
    const int nb = n / 128;
    for (int row = 0; row < nrc; row++) {
        const uint8_t * x_row = x + row * bx / 4;
        int32_t sum = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t * px = x_row + b * 32;
            const int8_t  * py = y + b * 128;
            for (int j = 0; j < 32; j++) {
                const uint8_t v = px[j];
                sum += ((v >> 6) & 3) * py[j];
                sum += ((v >> 4) & 3) * py[j + 32];
                sum += ((v >> 2) & 3) * py[j + 64];
                sum += ( v       & 3) * py[j + 96];
            }
        }
        s[row] = (float)sum;
    }
}

__attribute__((noinline))
static void neon_dot(int n, float * s, size_t bx, const uint8_t * x, const int8_t * y, int nrc) {
    const int nb = n / 128;
    const uint8x16_t m3 = vdupq_n_u8(3);
    for (int row = 0; row < nrc; row++) {
        const uint8_t * x_row = x + row * bx / 4;
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        for (int b = 0; b < nb; b++) {
            const uint8_t * px = x_row + b * 32;
            const int8_t  * py = y + b * 128;
            const uint8x16_t w0 = vld1q_u8(px);
            const uint8x16_t w1 = vld1q_u8(px + 16);
            acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vshrq_n_u8(w0, 6)), vld1q_s8(py));
            acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0, 4), m3)), vld1q_s8(py + 32));
            acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0, 2), m3)), vld1q_s8(py + 64));
            acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vandq_u8(w0, m3)), vld1q_s8(py + 96));
            acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vshrq_n_u8(w1, 6)), vld1q_s8(py + 16));
            acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w1, 4), m3)), vld1q_s8(py + 48));
            acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w1, 2), m3)), vld1q_s8(py + 80));
            acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vandq_u8(w1, m3)), vld1q_s8(py + 112));
        }
        s[row] = (float)vaddvq_s32(vaddq_s32(acc0, acc1));
    }
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(void) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    printf("MATMUL_INT8 macro: yes\n");
#else
    printf("MATMUL_INT8 macro: no\n");
#endif
    const int ns[] = {128, 256, 640, 2560, 6912};
    long cases = 0;
    for (int iter = 0; iter < 6000 + 4; iter++) {
        int n = ns[rnd() % 5];
        int nrc = 1 + rnd() % 8;
        uint8_t * x = malloc((size_t)nrc * n / 4);
        int8_t * y = malloc(n);
        for (size_t i = 0; i < (size_t)nrc * n / 4; i++) x[i] = (uint8_t)rnd();
        for (int i = 0; i < n; i++) y[i] = (int8_t)rnd();
        if (iter >= 6000) {
            for (size_t i = 0; i < (size_t)nrc * n / 4; i++) x[i] = 0xFF;
            for (int i = 0; i < n; i++) y[i] = (iter % 2) ? -128 : 127;
        }
        float a[8], b[8];
        ref_dot(n, a, n, x, y, nrc);
        neon_dot(n, b, n, x, y, nrc);
        for (int r = 0; r < nrc; r++) {
            if (a[r] != b[r]) {
                printf("MISMATCH n=%d nrc=%d row=%d ref=%f neon=%f\n", n, nrc, r, a[r], b[r]);
                return 1;
            }
        }
        free(x);
        free(y);
        cases++;
    }
    printf("OK: %ld cases identical\n", cases);

    const int n = 2560, rows = 6912, reps = 100;
    uint8_t * x = malloc((size_t)rows * n / 4);
    int8_t * y = malloc(n);
    float * out = malloc(rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * n / 4; i++) x[i] = (uint8_t)rnd();
    for (int i = 0; i < n; i++) y[i] = (int8_t)rnd();
    double chk = 0;
    double t0 = now();
    for (int k = 0; k < reps; k++) {
        for (int r = 0; r < rows; r++) ref_dot(n, &out[r], n, x + (size_t)r * n / 4, y, 1);
        chk += out[rows / 2];
    }
    double t_ref = (now() - t0) / reps;
    t0 = now();
    for (int k = 0; k < reps; k++) {
        for (int r = 0; r < rows; r++) neon_dot(n, &out[r], n, x + (size_t)r * n / 4, y, 1);
        chk += out[rows / 2];
    }
    double t_neon = (now() - t0) / reps;
    double gb = (double)rows * n / 4 / 1e9;
    printf("scalar: %.3f ms per matrix, %.1f GB/s of weights\n", t_ref * 1e3, gb / t_ref);
    printf("neon:   %.3f ms per matrix, %.1f GB/s of weights (%.1fx)\n", t_neon * 1e3, gb / t_neon, t_ref / t_neon);
    printf("checksum %f\n", chk);
    return 0;
}

// Streaming read bandwidth test. Build:
//   clang -O3 -mcpu=native -pthread scripts/bw_test.c -o /tmp/bw_test
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BYTES_PER_THREAD (256UL << 20)
#define ITERS 20
#define MAX_THREADS 6

typedef struct { const uint8_t *buf; uint64_t sink; } arg_t;

static void *work(void *p) {
    arg_t *a = p;
    uint64x2_t acc = vdupq_n_u64(0);
    for (int it = 0; it < ITERS; it++) {
        const uint8_t *b = a->buf;
        __asm__ volatile("" : "+r"(b));  // stop the compiler hoisting the pass
        for (size_t i = 0; i < BYTES_PER_THREAD; i += 64) {
            uint8x16_t v0 = vld1q_u8(b + i),      v1 = vld1q_u8(b + i + 16);
            uint8x16_t v2 = vld1q_u8(b + i + 32), v3 = vld1q_u8(b + i + 48);
            acc = vaddq_u64(acc, vreinterpretq_u64_u8(
                      veorq_u8(veorq_u8(v0, v1), veorq_u8(v2, v3))));
        }
    }
    a->sink = vgetq_lane_u64(acc, 0) ^ vgetq_lane_u64(acc, 1);
    return NULL;
}

int main(void) {
    uint8_t *mem = malloc(BYTES_PER_THREAD * MAX_THREADS);
    for (size_t i = 0; i < BYTES_PER_THREAD * MAX_THREADS; i++)
        mem[i] = (uint8_t)(i * 2654435761u >> 24);  // touch every page, non-zero data
    int counts[] = {1, 2, 4, 6};
    for (int c = 0; c < 4; c++) {
        int n = counts[c];
        pthread_t th[MAX_THREADS];
        arg_t args[MAX_THREADS];
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < n; i++) {
            args[i].buf = mem + i * BYTES_PER_THREAD;
            pthread_create(&th[i], NULL, work, &args[i]);
        }
        for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        double gb = (double)n * BYTES_PER_THREAD * ITERS / 1e9;
        printf("%d threads: %.1f GB/s  (sink %llu)\n", n, gb / s,
               (unsigned long long)args[0].sink);
    }
    return 0;
}

# BitNet b1.58 2B4T on a MacBook Neo: a NEON kernel for the I2_S format

Summary: on a base MacBook Neo (Apple A18 Pro, 8 GB), this fork runs Microsoft's BitNet b1.58 2B4T model correctly and generates text 2.7x to 3.6x faster than the scalar fallback it started with, with output verified identical. At 6 threads it reaches about 40 tokens/sec.

Hardware: MacBook Neo, Apple A18 Pro (2 performance + 4 efficiency cores, SME and SME2 reported by the OS), 8 GB, macOS 26.5.2. Model: BitNet-b1.58-2B-4T in the I2_S GGUF format. Details: bench/environment.txt.

## What was done

1. Fixed four bugs that broke the model on Apple Silicon (llama.cpp fork, branch fix/i2s-arm): the wrong activation function (3bd2404df), an undeclared variable that stopped the build (66330bcd9), a scalar dot product that read the wrong bit layout (6ac2a133b), and a GEMM fallback that misused the dot-product kernel and produced NaN (80778baae). Two small fixes in the BitNet repo itself: the CMake source list (5961b58) and the default batch size (162c687). Before the fixes the output was garbage at about 1.2 tokens/sec; that figure is from my project notes and there is no saved output file for it.
2. Profiled one generation run (bench/profile_scalar_2026-10-12.txt). One function, the I2_S dot product, took about 63% of each token.
3. Replaced that function's ARM path with a NEON dot-product kernel, about 30 lines (quants.c in the submodule, commit f332d6c74).
4. Added a Q8_0 output layer (the model ties its output layer to the F16 token embedding, 0.66 GB read per token). This saved about 5 ms per token for a perplexity change of +0.04%. scripts/add_q8_output.py builds the model file.

## Results (same session, plugged in, same model and flags, scalar build 075d752 against NEON build f332d6c74)

Generation, tokens/sec (mean +/- spread, 5 repeats):

| Threads | Scalar | NEON | Speedup |
| --- | --- | --- | --- |
| 1 | 6.43 +/- 0.25 | 22.96 +/- 0.95 | 3.57x |
| 2 | 9.01 +/- 0.15 | 30.57 +/- 0.59 | 3.39x |
| 3 | 9.02 +/- 0.00 | 29.99 +/- 0.05 | 3.32x |
| 4 | 11.53 +/- 0.02 | 36.05 +/- 0.17 | 3.13x |
| 6 | 15.43 +/- 0.15 | 40.97 +/- 0.81 | 2.66x |

Prompt processing, 64-token prompt, tokens/sec (3 repeats):

| Threads | Scalar | NEON | Speedup |
| --- | --- | --- | --- |
| 1 | 7.23 +/- 0.14 | 27.39 +/- 0.41 | 3.79x |
| 2 | 10.43 +/- 0.01 | 40.55 +/- 0.16 | 3.89x |
| 4 | 13.69 +/- 0.02 | 53.51 +/- 0.19 | 3.91x |
| 6 | 18.77 +/- 0.12 | 74.28 +/- 0.52 | 3.96x |

The 6-thread NEON generation number was measured three times (40.52, 40.97, 40.45). The generation speedup shrinks with more threads because the NEON kernel approaches the memory limit (about 64% of the measured 55 GB/s); prefill is compute-bound, so its speedup stays near 3.9x. Raw outputs, including 128-token prompts: bench/b2b_*.txt.

## Correctness

- Perplexity on the wikitext-2 test file (20 chunks, 512 context): 16.7885 with both kernels, and all 20 running values identical (bench/q8out_ppl.txt and bench/neon_ppl.txt).
- A standalone test compares the NEON kernel with the scalar one on 6,004 random and extreme-value cases: bit-identical, run twice (bench/neon_unit_test.txt).
- No NaN in the tensor trace (bench/neon_nan.txt); the same greedy sentence in both builds (bench/neon_paris.txt).

## Limits

- The baseline is the scalar fallback. I have not compared against other optimized BitNet implementations, so this does not show the kernel is faster than what already exists.
- One chip, one model, short prompts, CPU only. Perplexity covers 20 chunks, not the full test file.
- Three threads is no faster than two with either kernel; I have not yet explained why.
- The Q8_0 model is derived from the stock one by a script; SHA-256 hashes of the inputs are in bench/input_checksums.txt.
- Not yet upstreamed or reviewed by a maintainer.

## Who did what

I set the direction, chose what to measure, ran every test, and required that nothing count as done until it was committed and verified. Most of the code (the kernel, the helper scripts, and the diagnosis of the bugs) was written with Claude, an AI assistant.

## Reproduce

Every result file in bench/ starts with a header recording the commit hashes, chip, power state and exact command; bench/README.md explains each file, including the older ones. The verified state is the neon-verified tag in both repos. The full setup from a clean machine has not been repeated yet; the steps are in my project notes and I will add them here once tested.

## Next

Thread-split tuning (the 3-thread dip), a feasibility test for ARM's SME instructions on prompt processing, scaling tests on larger Apple chips, a comparison against existing implementations, and a hardening pass (sanitizers, wider tests).

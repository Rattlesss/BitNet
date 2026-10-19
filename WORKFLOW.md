# How this work was done

This is the working method behind RESULTS.md. The point is that every claim can be traced to a saved file and a commit.

## The loop

1. Baseline first. Before changing any code I recorded how the model behaved: generation and prompt speed (bench/baseline_*.txt), perplexity (bench/baseline_ppl.txt), and the hardware facts (bench/environment.txt, bench/gguf_tensors.txt).
2. Profile before optimizing. A sampling profile (bench/profile_scalar_2026-10-12.txt) showed one function, the I2_S dot product, took about 63% of each token, so that is where the effort went.
3. Predict, then measure. Before the NEON kernel was integrated I wrote down an expected range (22 to 37 tokens/sec at 4 threads from the profile numbers). The measured 36.05 landed at the top of it.
4. Test the change in isolation. scripts/test_i2s_neon.c compares the new kernel with the scalar reference on 6,004 random and extreme cases before the model code was touched.
5. Correctness before speed. A change only counts if the output is identical: the same perplexity at every one of 20 chunks, no NaN in the tensor trace, the same greedy sentence.
6. Record every run. scripts/record.sh saves the output with a header: commit hashes of both repos, chip, power source, thermal state and the exact command.
7. Compare back to back. The scalar and NEON builds were measured in one session, plugged in, same model and flags, with repeats and the spread reported (bench/b2b_*.txt).
8. State the limits. RESULTS.md says what was not tested, including that the baseline is the scalar fallback and not other optimized kernels.

## Git habits

- One change per commit, in the form "Verb: file; description".
- Commit before measuring, then rebuild so the binary's build id matches the commit.
- Merge branches with --no-ff so the history shows each feature as its own branch.
- Tag verified states (neon-verified, in both repos) so they are easy to return to.
- Push the submodule before the parent repo, since the parent records a submodule commit.

## Mistakes that were caught

- A profile entry (madvise) looked like a per-token cost. Reading its call stack showed it was model loading, so it was removed from the per-token table and the doc was corrected.
- The first perplexity result files captured only stdout, so the final-estimate line is missing. This was found when checking the saved files, and bench/README.md records it.
- Some early benchmark runs were on battery with no power state recorded. The back-to-back session was run plugged in to remove that doubt.
- A binary was once built before its commit, so its build id would have been stale. The commit-then-rebuild rule came from that.

## Working with an AI assistant

Claude (an AI assistant) wrote much of the code and the helper scripts, and proposed the experiments. I ran every command myself, pasted back the raw output, decided what to do next, and required that nothing count until it was committed and verified. Details are in the "Who did what" section of RESULTS.md.

## Where to look

- RESULTS.md: the results, correctness checks and limits.
- bench/README.md: what each result file is and its known gaps.
- scripts/: the tools (record.sh, the unit test, the model-builder, the bandwidth test).

# Result files

Files made with scripts/record.sh begin with a header: date, parent and submodule commit hashes, chip, power and thermal state, and the exact command. Those are gguf_tensors.txt, neon_unit_test.txt and input_checksums.txt, plus every file from now on.

Older files were made before record.sh and have no header. Their commands and builds:

- baseline_tg.txt: numbers typed by hand from a llama-bench run, not raw output. llama-bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p 0 -n 64 -t 1,2,4 -ngl 0 -r 3; build 80778baae, scalar I2_S kernel.
- baseline_pp.txt: llama-bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p 64,128 -n 0 -t 1,2,4 -ngl 0 -b 64 -ub 16 -r 3; build 80778baae.
- baseline_ppl.txt: llama-perplexity on the same model and wikitext-2 wiki.test.raw, -c 512 -b 64 -ub 16 --chunks 20 -t 4 -ngl 0 --override-kv tokenizer.ggml.pre=str:llama-bpe. Holds the 20 running values; the last, 16.7822, is the final estimate. The final-estimate line (+/- 0.70059) was on stderr and is not in the file.
- q8out_tg.txt: llama-bench on ggml-model-i2_s-q8out.gguf, same flags as baseline_tg. The build line says b6c645184, but the binary also held the graph change that was committed afterward as 075d752.
- q8out_ppl.txt: llama-perplexity on ggml-model-i2_s-q8out.gguf, same flags as baseline_ppl. Same build caveat. Final estimate 16.7885 (+/- 0.70083 was on stderr, not in the file).
- profile_scalar_2026-10-12.txt: the top-of-stack table and the madvise call stack from a macOS sample run, scalar kernel, submodule 075d752. profile_scalar_2026-10-12_raw.txt is the full report. The sample window overlapped the end of model loading, so madvise is load time.

Power state was not recorded for any older file.

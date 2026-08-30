# Long-context company traces

`company_probe.c` is a diagnostic copy of the numerical llama2.c path. It
retains observations at actual attention, residual, FFN, and layer boundaries.
It is not the composed Atkey evaluator and must not be used as evidence that
weights are physically read once.

The probe accepts inline text for small checks:

```sh
make companyprobe
./company_probe test/stories260K.bin test/tok512.bin \
  "Once upon a time." " A girl walked home."
```

For full stories, pass files so newlines and long text do not travel through
shell arguments:

```sh
./company_probe test/stories260K.bin test/tok512.bin \
  --files prompt.txt completion.txt \
  --trace trace.jsonl --checkpoint-every 32
```

The Stories260K checkpoint has a hard 512-token context. The probe accepts all
512 positions. It does not silently slide, truncate, or reset positions for
longer input because that would change the model being measured. Exact token
counts can be checked without inference:

```sh
./company_probe test/stories260K.bin test/tok512.bin \
  --files prompt.txt completion.txt --count-only
```

Every JSONL record is flushed immediately. Record kinds are:

- `meta`: exact prompt, completion, model-context, layer, and width counts.
- `token`: selected token text, log probability, local rank, distribution
  entropy, top token, cumulative log probability, and prefix mean.
- `layer`: per-token measurements at each actual layer boundary, including
  attention allocation, residual/FFN update ratios and alignments, residual
  inertia, and recurrence against prior hidden states.
- `terminal`: probability and rank of token 1, the delimiter used by the
  TinyStories generation path in this llama2.c implementation.
- `affine_prefix`: affine effective dimension, variance, and path efficiency
  at periodic completion-prefix checkpoints. These are computed online in
  `O(tokens * layers * dim^2)`, rather than by the earlier quadratic pairwise
  rescan.

The records are measurements, not a scalar quality score. A later analysis may
group token and layer records into any desired interval; the probe does not
declare fixed token widths to be syllables, words, or phrases.

To collect several long validation examples without assigning quality labels:

```sh
python3 gather_company_traces.py \
  --cases 4 --min-total-tokens 420 --max-total-tokens 512
```

For each selected source story, the collector saves the prompt, observed final
paragraph, textual aggregate summary, and flushed JSONL trace. `manifest.csv`
records source indices and exact tokenizer counts. Character length affects
only scan order; inclusion always uses the C tokenizer.

Inspect fixed-size windows or any explicitly chosen interval without changing
the recorded data:

```sh
python3 inspect_company_trace.py trace.jsonl --window 40 --layer 4
python3 inspect_company_trace.py trace.jsonl \
  --span 96:144 --layer 4 --non-top-one
```

The interval reader reports its decoded text, token likelihood data, and
averages of the retained observations at the selected layer. `--non-top-one`
shows the selected and locally preferred token pieces and both scores within
the inspected interval.

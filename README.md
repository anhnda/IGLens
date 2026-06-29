# IG-Lens

Telescoping layer-wise Integrated Gradients on the residual stream of a
decoder-only transformer. For a predicted token, IG-Lens decomposes its
**final-layer probability** into per-layer contributions whose sum is *exactly*
the change in target probability:

```
sum_over_chosen_layers IG_L  ==  p_i(final) - p_i(baseline)
```

The decomposition is in **probability space** (the softmax is integrated through,
not linearized) and is **additive across layers** (the per-layer terms telescope
to the total). This is the property logit-space methods (DLA), level-reading
lenses (logit lens, Tuned Lens), and per-layer attribution (Layer Conductance)
each miss in a different way.

## How it works

Pick layers `0 = L0 < L1 < ... < Lk = n` (the final layer is always added). Run
one path through the chosen hidden states,

```
h_base -> h_L1 -> h_L2 -> ... -> h_final
```

and credit each *segment* `h_{Lj-1} -> h_{Lj}` to the layer `Lj` it ends at,
using the same readout applied **once** at the end:

```
f(h) = softmax(lm_head(norm(h)))[y_t]
```

`f` contains no attention and no MLP — it is pointwise in `h`. By the gradient
theorem each segment integral equals `f(h_Lj) - f(h_{Lj-1})`, so the sum
telescopes to `f(h_final) - f(h_base)`. That is the whole identity.

**What IG_L means.** The additional probability that `norm+head` can read out of
`h_L` *beyond* the previously chosen layer — a conditional marginal, not the
total causal effect of the layer through the blocks above it. That trade is the
price of an exact telescoping sum; if you want total layer effect through upper
blocks, use a DLA-style component decomposition instead.

## The `--normalize` estimator (default behavior to enable)

A finite-step Riemann estimate of the segment integral only *approaches* the
endpoint difference as `n_steps -> inf`, and credits steps where the gradient is
large but the output does not move ("sensitivity without response"). The
`--normalize` flag instead credits each step its **observed** output change,
transporting the IDGI consistency principle to the segment:

```
IG_Lj = sum_s ( f(h^(s)) - f(h^(s-1)) )        # grid a_0=0 < ... < a_m=1
```

Because `f` is a one-dimensional probability, IDGI's per-dimension redistribution
collapses to exactly this. Two consequences:

1. **Completeness is exact at any step count.** The segment telescopes
   identically for any grid and any `m >= 1`; the only error is float summation,
   not discretization. `n_steps` stops being an accuracy knob.
2. **Spurious-sensitivity steps are filtered out** — a step that does not move
   the output contributes nothing, by construction.

Under `--normalize`, the computation needs **no backward pass**: each segment is
a difference of forward evaluations of `f` on the grid. It is both faster and
exactly complete. The raw-gradient path (without the flag) is kept as a reference
variant and agrees in the `m -> inf` limit.

## Requirements

- Python 3.9+
- PyTorch (CUDA optional; runs on CPU)
- `transformers`

```bash
pip install torch transformers
```

The model loads in float32; the default is `meta-llama/Llama-3.2-1B-Instruct`
(gated — accept the license and `huggingface-cli login` first).

## Usage

```bash
python ig_lens.py "What is the capital of Vietnam?"
```

Interactive loop with explicit layers and the normalize estimator:

```bash
python ig_lens.py --loop --layers 12 13 14 15 16 --n-steps 4 --normalize
```

Pipe a sentence in:

```bash
echo "Translate to French: good morning" | python ig_lens.py --stdin --normalize
```

### Example output

```
Answer: 'The capital of Vietnam is Hanoi.'
baseline=mean  n_steps=4  target=prob  onset_frac=0.5  normalize=True
idx  token             IG@L16   lens@L16    IG@L15  ...    sum    Δprob  onsetL*
  0  'The'             0.2127     0.9356    0.7097  ... 0.9355   0.9355       15
  1  ' capital'        0.0090     1.0000    0.0796  ... 0.9999   0.9999       12
  ...
  6  'anoi'            0.5905     0.9999    0.1899  ... 0.9998   0.9998       16
```

- `IG@L`   — telescoping segment IG credited to layer `L`.
- `lens@L` — logit-lens probability `softmax(head(norm(h_L)))[y_t]`, **reference
  only**, not used for onset.
- `sum`    — sum of `IG@L` over chosen layers; equals `Δprob` by construction.
- `Δprob`  — `p(final) - p(baseline)` for the token.
- `onsetL*`— earliest layer where cumulative `|IG|` reaches `--onset-frac` of the
  total mass. Small = decided early/easy; large = decided late/hard.

Note: displayed columns may omit a layer for width while still including it in
`sum`, so a displayed row may total below `Δprob` when mass sits in the omitted
(usually final) segment.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `sentence` | — | positional prompt (or use `--stdin` / interactive) |
| `--model` | `meta-llama/Llama-3.2-1B-Instruct` | any causal LM with `norm` + `lm_head` |
| `--layers L ...` | — | explicit `hidden_states` indices; final layer auto-added |
| `--k` | `6` | if `--layers` omitted, sample this many layers evenly |
| `--n-steps` | `64` | interpolation steps per segment (irrelevant to accuracy under `--normalize`) |
| `--normalize` | off | IDGI/prediction-aware estimator: exact completeness at any step count, no backward |
| `--baseline` | `mean` | `mean` (per-position sequence mean) or `zero` |
| `--onset-frac` | `0.5` | mass fraction defining the onset layer `L*` |
| `--max-new-tokens` | `80` | greedy generation length |
| `--stdin` | off | read the prompt from stdin |
| `--loop` | off | keep prompting after each run |

## Completeness check

Every run prints `completeness max|sum - Δprob|`. Without `--normalize` this sits
in the Riemann range (~1e-4..1e-2) and tightens as you raise `--n-steps`. With
`--normalize` it drops to float precision (~1e-6..1e-4) and is independent of
`--n-steps`. A large value without `--normalize` means raise `--n-steps`.

## Layer indexing

Indices refer to `output_hidden_states`: `0` is the embedding, `n` (e.g. 16 for a
16-layer model) is the final hidden state. The final index is always included so
the path ends at the real final hidden vector and `f(h_final)` is the true model
probability.

## Files

- `ig_lens.py` — the tool.
- `ig_lens_idgi_entry.bib` — BibTeX entry for IDGI (the estimator's origin), if
  you are citing the accompanying writeup.

## Caveat

IG-Lens measures readout, not routing. It deliberately ignores how `h_L`
propagates through attention and MLP in the blocks above it — that is what buys
the exact telescoping sum. For total layer effect through upper blocks, reach for
a different tool.
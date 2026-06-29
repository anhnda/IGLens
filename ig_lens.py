"""
ig_lens.py — telescoping layer-IG attribution on hidden states.

Goal
----
For target token y_t predicted at position i, attribute its FINAL-LAYER
probability to the hidden states h_L[i] of a chosen set of layers, such that:

    sum_over_chosen_layers  IG_L  ==  prob_i(origin) - prob_i(baseline)

where prob is read by the SAME final head f(h) = softmax(head(norm(h)))[y_t],
applied ONCE.

Why telescoping (and not independent per-layer IG)
--------------------------------------------------
IG completeness says sum over INPUT DIMS of one path = f(x) - f(x0). It does
NOT say sum over LAYERS = anything, if each layer is attributed independently
back to its own baseline. To get a layer-wise sum that equals the total
prob change, we chain the hidden states ON ONE PATH:

    baseline -> h_{L1} -> h_{L2} -> ... -> h_{Lk}=h_final

and define the contribution of layer Lj as the IG of f over the straight
segment h_{Lj-1} -> h_{Lj} (with h_{L0} := baseline). Each segment's IG, by
the fundamental theorem of calculus, equals f(h_{Lj}) - f(h_{Lj-1}); summing
telescopes to f(h_final) - f(baseline) = prob(origin) - prob(baseline). QED.

f uses ONLY final norm + lm_head. So IG_L measures what norm+head can read
out of h_L beyond h_{L-1}. It deliberately ignores the effect of h_L routed
through upper blocks (attention/MLP). That is the price of a clean
telescoping sum; you cannot have both "total effect via full forward" and
"sum == delta-prob". This file picks the latter, as requested.

--normalize (IDGI / PACE-Grad-style prediction-aware reweighting)
-----------------------------------------------------------------
Borrowing the core idea of IDGI [Yang et al. 2023] and the prediction-aware
consistency principle of PACE-Grad: a step on the path should be credited in
proportion to the OBSERVABLE OUTPUT CHANGE Δf it produces, not the raw
gradient magnitude. Standard IG accumulates <diff, g_s>/n uniformly; this
over-credits steps where g_s is large but f does not actually move (spurious
sensitivity in out-of-support regions).

With --normalize ON, the contribution of interior step s of a segment is
replaced by the true local output change

    Δf_s = f(h_{a_s}) - f(h_{a_{s-1}}),

and the segment value becomes  sum_s Δf_s  (a midpoint-free telescoping of f
along the segment's interior grid). Because f is a 1-D readout here, the
IDGI per-dimension redistribution g⊙diff / <g,diff> collapses to a scalar, so
"normalize" reduces exactly to crediting each step its real Δf. This keeps
EXACT per-segment completeness (sum_s Δf_s = f(h_end) - f(h_start)) — and in
fact makes it exact up to float error regardless of n_steps — while killing
sensitivity-without-response steps. The overall telescoping sum == Δprob is
preserved either way.
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Prompt / generation helpers
# ---------------------------------------------------------------------------

def get_stop_token_ids(tokenizer):
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    for s in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop.add(tid)
    return stop


def build_prompt_ids(tokenizer, sentence):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentence}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return ids["input_ids"][0].to(DEVICE)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt_ids, stop_ids, max_new_tokens):
    cur = prompt_ids.clone()
    out = []
    for _ in range(max_new_tokens):
        logits = model(cur.unsqueeze(0)).logits[0, -1]
        nxt = int(logits.argmax().item())
        if nxt in stop_ids:
            break
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt], device=DEVICE)])
    return torch.tensor(out, dtype=torch.long, device=DEVICE)


# ---------------------------------------------------------------------------
# The ONLY readout: f(h) = softmax(lm_head(norm(h)))[y_t].  Head applied once.
# ---------------------------------------------------------------------------

def get_norm_head(model):
    inner = getattr(model, "model", model)
    normf = getattr(inner, "norm", None)
    if normf is None:
        normf = getattr(inner, "final_layernorm")
    head = getattr(model, "lm_head", None)
    if head is None:
        head = getattr(model, "embed_out")
    return normf, head


def f_prob(normf, head, h_vec, target_id):
    """h_vec: [d] (may require grad). Returns scalar prob of target_id."""
    logits = head(normf(h_vec.unsqueeze(0))).squeeze(0)   # [V]
    return F.softmax(logits, dim=-1)[target_id]


@torch.no_grad()
def f_prob_batch(normf, head, h_mat, target_id):
    """h_mat: [b, d]. Returns prob of target_id per row, [b]. No grad."""
    logp = F.log_softmax(head(normf(h_mat)), dim=-1)      # [b, V]
    return logp[:, target_id].exp()                       # [b]


# ---------------------------------------------------------------------------
# Segment IG: integrate f along straight line h_start -> h_end.
# Returns IG_segment ~= f(h_end) - f(h_start) (completeness on this segment).
#
# normalize=False (default): standard IG, <diff, mean_s grad(h_a)>.
# normalize=True (IDGI/PACE-Grad): credit each interior step its true output
#   change Δf_s = f(h_{a_s}) - f(h_{a_{s-1}}). For a 1-D readout this is the
#   exact IDGI redistribution and gives exact segment completeness.
# ---------------------------------------------------------------------------

def segment_ig(normf, head, h_start, h_end, target_id, n_steps,
               normalize=False):
    diff = (h_end - h_start)

    if normalize:
        # endpoints of the n_steps interior intervals: a = 0, 1/n, ..., 1
        # crediting each interval its real Δf is a clean telescoping of f.
        grid = torch.linspace(0.0, 1.0, n_steps + 1, device=h_start.device)
        pts = h_start.unsqueeze(0) + grid.unsqueeze(1) * diff.unsqueeze(0)  # [n+1, d]
        fvals = f_prob_batch(normf, head, pts, target_id)                  # [n+1]
        return float((fvals[1:] - fvals[:-1]).sum().item())

    grad_accum = torch.zeros_like(diff)
    for s in range(n_steps):
        a = (s + 0.5) / n_steps                 # midpoint Riemann
        h_a = (h_start + a * diff).detach().requires_grad_(True)
        y = f_prob(normf, head, h_a, target_id)
        g, = torch.autograd.grad(y, h_a)
        grad_accum += g.detach()
    grad_mean = grad_accum / n_steps
    return torch.dot(diff, grad_mean).item()


# ---------------------------------------------------------------------------
# Batched segment IG: all (token, segment, step) interpolation points in ONE
# backward, chunked to bound VRAM. Equivalent to calling segment_ig for every
# (t, segment); each interpolation point passes through f independently
# (f = norm+head is pointwise on a single hidden vector, no cross-token path),
# so grad(p.sum()) over the flat batch yields the per-point gradient exactly.
# Returns ig[T, K]: ig[t, j] is the IG credited to chosen_lo[j] for token t.
#
# normalize=True: same prediction-aware reweighting as segment_ig, fully
#   batched and grad-free — each segment's value is sum_s Δf_s over its
#   interior grid. Δprob completeness is preserved exactly.
# ---------------------------------------------------------------------------

def segment_ig_batched(normf, head, starts, ends, target_ids, n_steps,
                       max_rows=4096, normalize=False):
    """
    starts, ends: [T, K, d]   target_ids: [T] (long)
    Returns ig: [T, K] on CPU-friendly float (same value as segment_ig).
    max_rows caps how many interpolation points go through head() at once,
    to keep the [rows, vocab] logits tensor within VRAM.
    """
    T, K, d = starts.shape
    device = starts.device
    diff = ends - starts                                  # [T, K, d]

    if normalize:
        # grid endpoints a = 0 .. 1 with n_steps intervals: [n+1]
        grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)        # [n+1]
        npts = n_steps + 1
        # all grid points: [T, K, n+1, d] -> flat [T*K*(n+1), d]
        pts = (starts[:, :, None, :]
               + grid[None, None, :, None] * diff[:, :, None, :])
        flat = pts.reshape(-1, d)                          # [N, d]
        N = flat.size(0)
        tgt_full = (target_ids[:, None, None]
                    .expand(T, K, npts).reshape(-1))        # [N]

        f_flat = torch.empty(N, device=device, dtype=flat.dtype)
        with torch.no_grad():
            for lo in range(0, N, max_rows):
                hi = min(lo + max_rows, N)
                logp = F.log_softmax(head(normf(flat[lo:hi])), dim=-1)
                rows = torch.arange(hi - lo, device=device)
                f_flat[lo:hi] = logp[rows, tgt_full[lo:hi]].exp()
        fvals = f_flat.reshape(T, K, npts)                 # [T, K, n+1]
        ig = (fvals[:, :, 1:] - fvals[:, :, :-1]).sum(-1)  # [T, K]
        return ig

    a = (torch.arange(n_steps, device=device) + 0.5) / n_steps   # [n]

    # all interpolation points: [T, K, n, d] -> flat [T*K*n, d]
    pts = starts[:, :, None, :] + a[None, None, :, None] * diff[:, :, None, :]
    flat = pts.reshape(-1, d)                              # [N, d], N=T*K*n
    N = flat.size(0)

    # per-row target id, aligned with the flattening order (t, j, s)
    tgt_full = (target_ids[:, None, None]
                .expand(T, K, n_steps).reshape(-1))        # [N]

    grad_flat = torch.empty_like(flat)
    for lo in range(0, N, max_rows):
        hi = min(lo + max_rows, N)
        leaf = flat[lo:hi].detach().requires_grad_(True)   # [b, d]
        logp = F.log_softmax(head(normf(leaf)), dim=-1)    # [b, V]
        rows = torch.arange(hi - lo, device=device)
        p = logp[rows, tgt_full[lo:hi]].exp()              # prob, [b]
        g, = torch.autograd.grad(p.sum(), leaf)            # block-diag -> per-row
        grad_flat[lo:hi] = g.detach()

    grad_mean = grad_flat.reshape(T, K, n_steps, d).mean(2)   # [T, K, d]
    ig = (diff * grad_mean).sum(-1)                           # [T, K]
    return ig


# ---------------------------------------------------------------------------
# Onset extraction (unchanged semantics): earliest layer whose contribution
# stays >= threshold to the end.
# ---------------------------------------------------------------------------

def onset_mass(ig_map, layer_order_low_to_high, frac=0.5):
    """
    Earliest (low->high) layer at which cumulative |IG| reaches `frac` of the
    total |IG| mass. Interpretation: 'the layer by which >=frac of the target
    probability has been read out'. Small L* => decided early/easy; large L*
    => decided late/hard. Parameter-free vs the old threshold rule.
    """
    tot = sum(abs(ig_map[li]) for li in layer_order_low_to_high)
    if tot <= 0.0:
        return layer_order_low_to_high[-1]
    c = 0.0
    for li in layer_order_low_to_high:
        c += abs(ig_map[li])
        if c / tot >= frac:
            return li
    return layer_order_low_to_high[-1]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(model, tokenizer, sentence, args):
    normf, head = get_norm_head(model)

    stop_ids = get_stop_token_ids(tokenizer)
    prompt_ids = build_prompt_ids(tokenizer, sentence)
    answer_ids = greedy_generate(
        model, tokenizer, prompt_ids, stop_ids, args.max_new_tokens)
    if answer_ids.numel() == 0:
        print("  (model produced no answer)\n")
        return

    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        hs = model(full, output_hidden_states=True).hidden_states  # len n+1
    n_layers = len(hs) - 1
    Lp = prompt_ids.numel()
    T = answer_ids.numel()

    # chosen hidden_states indices, ALWAYS include final (n_layers) so the
    # chain ends at the real final hidden -> f(h_final) == true prob.
    if args.layers:
        chosen = sorted(set(int(x) for x in args.layers) | {n_layers})
    else:
        raw = torch.linspace(1, n_layers, steps=min(args.k, n_layers))
        chosen = sorted(set(raw.round().long().tolist()) | {n_layers})

    # reference logit-lens (NOT used for onset)
    @torch.no_grad()
    def lens_p(li, position, target_id):
        h = hs[li][0, position]
        logits = head(normf(h.unsqueeze(0))).squeeze(0)
        return F.softmax(logits, dim=-1)[target_id].item()

    def baseline_vec(li, position):
        if args.baseline == "zero":
            return torch.zeros_like(hs[li][0, position])
        # mean over sequence positions at the LOWEST chosen layer's index.
        return hs[li][0].mean(dim=0).detach()

    pos_slots = torch.arange(Lp - 1, Lp - 1 + T)

    chosen_hi = sorted(chosen, reverse=True)
    chosen_lo = sorted(chosen)

    print()
    print(f"Answer: {tokenizer.decode(answer_ids)!r}")
    print(f"baseline={args.baseline}  n_steps={args.n_steps}  "
          f"target=prob  onset_frac={args.onset_frac}  "
          f"normalize={args.normalize}")
    hdr = f"{'idx':>3}  {'token':<14}"
    for li in chosen_hi:
        hdr += f"{'IG@L'+str(li):>12}{'lens@L'+str(li):>12}"
    hdr += f"{'sum':>9}{'Δprob':>9}{'onsetL*':>9}"
    print(hdr)
    print("-" * len(hdr))

    K = len(chosen_lo)

    # --- build the chains for ALL tokens at once -------------------------
    # ends[t, j]   = h_{chosen_lo[j]}[position_t]
    # starts[t, j] = base_t            if j == 0
    #              = ends[t, j-1]      otherwise
    H = torch.stack([hs[li][0] for li in chosen_lo], dim=0)   # [K, S, d]
    positions = pos_slots.to(H.device)
    ends = H[:, positions, :].permute(1, 0, 2).contiguous()   # [T, K, d]

    base_vecs = torch.stack(
        [baseline_vec(chosen_lo[0], int(p.item())) for p in pos_slots],
        dim=0).to(H.device)                                    # [T, d]

    starts = torch.empty_like(ends)
    starts[:, 0, :] = base_vecs
    if K > 1:
        starts[:, 1:, :] = ends[:, :-1, :]

    target_ids = answer_ids.to(H.device).long()                # [T]

    # --- one batched backward for every (token, segment) ----------------
    ig_all = segment_ig_batched(
        normf, head, starts, ends, target_ids, args.n_steps,
        normalize=args.normalize)                              # [T, K]

    # completeness references f(h_final) and f(base), per token, no-grad
    with torch.no_grad():
        f_final_all = F.softmax(head(normf(ends[:, -1, :])), dim=-1)
        f_final_all = f_final_all[torch.arange(T), target_ids]      # [T]
        f_base_all = F.softmax(head(normf(base_vecs)), dim=-1)
        f_base_all = f_base_all[torch.arange(T), target_ids]        # [T]

    max_err = 0.0
    for t in range(T):
        position = int(pos_slots[t].item())
        target_id = int(answer_ids[t].item())
        tok = tokenizer.decode([target_id])

        ig_map = {li: float(ig_all[t, j].item())
                  for j, li in enumerate(chosen_lo)}

        lens_map = {li: lens_p(li, position, target_id) for li in chosen_lo}

        total = sum(ig_map.values())
        delta = float(f_final_all[t].item() - f_base_all[t].item())
        max_err = max(max_err, abs(total - delta))

        Lstar = onset_mass(ig_map, chosen_lo, args.onset_frac)

        row = f"{t:>3}  {repr(tok):<14}"
        for li in chosen_hi:
            row += f"{ig_map[li]:>12.4f}{lens_map[li]:>12.4f}"
        row += f"{total:>9.4f}{delta:>9.4f}"
        row += f"{(str(Lstar) if Lstar is not None else '-'):>9}"
        print(row)

    print()
    print(f"completeness max|sum - Δprob| = {max_err:.2e}  "
          f"(should be ~1e-4..1e-2; if larger, raise --n-steps)")
    print("IG@L  = telescoping segment IG credited to layer L: "
          "f(h_L)-f(h_{prev chosen}); head=norm+lm_head applied ONCE.")
    if args.normalize:
        print("normalize=ON: each path step credited its true output change "
              "Δf (IDGI / PACE-Grad prediction-aware reweighting); "
              "sensitivity-without-response steps are suppressed.")
    print("sum over chosen layers == prob_i(final) - prob_i(baseline)  (Δprob).")
    print("lens@L = softmax(head(norm(h_L[i])))[y_t]  REFERENCE ONLY.")
    print(f"onsetL* = earliest layer where cumulative |IG| reaches "
          f"{args.onset_frac:.0%} of total mass (small=early/easy, "
          f"large=late/hard).")
    print(f"Layers (hidden_states idx, 0=emb {n_layers}=final): "
          + ", ".join('L'+str(li) for li in chosen_lo))
    print()


def get_sentence(args):
    if args.sentence:
        return " ".join(args.sentence)
    if args.stdin:
        return sys.stdin.read().strip()
    try:
        return input("sentence> ").strip()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Telescoping layer-IG onset (sum == Δprob).")
    ap.add_argument("sentence", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--k", type=int, default=6,
                    help="number of layers to sample if --layers not given")
    ap.add_argument("--layers", nargs="*",
                    help="explicit hidden_states indices (final auto-added)")
    ap.add_argument("--n-steps", type=int, default=64,
                    help="IG interpolation steps PER SEGMENT")
    ap.add_argument("--baseline", choices=["zero", "mean"], default="mean")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--onset-frac", type=float, default=0.5,
                    help="onset L* = earliest layer where cumulative |IG| "
                         "reaches this fraction of total mass")
    ap.add_argument("--normalize", action="store_true",
                    help="IDGI/PACE-Grad prediction-aware reweighting: credit "
                         "each path step its true output change Δf instead of "
                         "the raw gradient (suppresses sensitivity-without-"
                         "response steps). Δprob completeness preserved.")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(DEVICE).eval()

    while True:
        sentence = get_sentence(args)
        if not sentence:
            break
        run(model, tokenizer, sentence, args)
        if not args.loop:
            break
        args.sentence = []


if __name__ == "__main__":
    main()
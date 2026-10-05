# Cross-talk with image inputs: audit of the cache-reuse architecture — findings and fix

Date: 2026-10-04. Server: `run_128k_4slots.sh` (4 decode slots, 128K context, prefix cache + GPU-slot cache + CED
prefill). Method: code audit of every state-reuse path, then controlled experiments against the live server with
synthetic images whose content is known (a coloured shape and a unique word), sequentially, concurrently, mixed with
text requests, with long shared system prompts (so every reuse path fires), and a randomized 4-user audit.

## 1. Result in one paragraph

No reuse path leaks one request's *tokens or kv* into another: every prefix/GPU-slot/snapshot lookup requires an
exact token-id prefix match, image requests bypass all of them, and the 4-user randomized audit (120 answers,
secrets in images and text, shared 2.5K-token system prompt, histories with and without re-sent images) produced
**0 answers containing another user's secret**. What *does* happen is that the **CED prefill approximation**
(`DSV41_CED=1`, the default in all run scripts) computes layers 21–39 only over the last 128 tokens of a prompt;
for those tokens the sliding-window attention of layers 21–39 reaches into ring slots that this prefill never wrote
— they still hold the previous request's kv in the slot-0 scratchpad — and the hidden states of the prompt's tail are
computed against another request's context. For text this is usually harmless; for an **image that ends near the end
of the prompt it makes the image unreadable** (the model answers with words from the surrounding text), which from
the outside looks exactly like cross-talk, and it is history-dependent and flaky. With `DSV41_CED=0` every failing
case reads the image correctly (9/9). The fix is to run exact prefill (`DSV41_CED=0`, now set in
`run_128k_4slots.sh`) at a prefill-speed cost of about 1.7× on 2–4K prompts (§5).

## 2. Reuse paths audited (`dsv41/engine.py`)

| path | key | image requests | risk found |
|---|---|---|---|
| GPU-slot continuation (`_find_best_gpu_slot` → `_prefill_with_gpu_slot_reuse`) | longest common token prefix with any slot's record (`_slot_tokens`, prompt + generated), min 64 tokens, continuation from a multiple of 16 ≤ lcp, needs the slot's window log | skipped for image requests (`best_gpu_slot = (0,0) if req.images`); a text request can match an image request's slot only up to the first image placeholder, i.e. identical text | none; window log protects slot 0 after a failed prefill (`_gpu_slot_reuse_pos` → `None` → SKIP) |
| exact-prefix continuation of slot 0 (`_can_reuse_prefix`) | the whole previous prompt is a prefix of the new one | image prompts contain expanded placeholder spans a text prompt cannot contain | none |
| host / tmpfs prefix snapshots (`_find_prefix_cache_entry`, LCP-HIT, BASE, anchors, block replay) | exact prefix match of `base_ids`; the snapshot is the state *at the base*, replay from the base | image branch returns before any lookup and never stores | none |
| vision inputs (`format_chat` → `prepare_vl_inputs` → `merge_image_embeddings`) | per-request `ImageInput` objects; ViT features cached on the request's object only; no module-level cache; image bytes decoded per request (base64 / data URL / file / http) | – | none |
| `copy_seq(0 → slot)` | copies rings, compressor state, compressed kv, index keys (all mirrors), Engram history | – | none |
| CUDA-graph recapture on cache growth (`_prepare_decode_cache`) | dry pass writes position 0 of every slot; compress/index backed up, window rings / compressor rings / Engram not | only on growth past the initial 32K-token caches | latent (rare; not the reported symptom) |
| failed prefill (worker `except`) | slot 0 may be partially overwritten | `_slot_tokens[0]`, `_prefix_prompt_ids` were left stale | fixed: both invalidated together with the window log (this commit) |
| **CED prefill** (`model._forward_sequential`, `forward_pipelined`: `DSV41_CED`) | layers ≤ `DSV41_CED_ENCODER_LAST_LAYER` (20) over the whole prompt, layers 21–39 over the last 128 tokens only (sequential) / only over the last chunk (pipelined) | all prompts ≥ 2048 tokens | **the symptom** (§3, §4) |

Production log evidence (post-restart window): 948 image prefills and 12,168 text prefills; 9,022 GPU-slot
continuations, one of them sourced from a slot last filled by an image request — at lcp 20,890 of a 26,949-token
prompt whose images came after the shared history (legitimate). The 150+ historical `prefill error ... shape of the
mask` failures were the pre-09-22 chunked-image bug (fixed in 65a96b4); the 11 post-restart errors are the ViT OOMs
of this audit's 1920×1080 frames (6 × 4.2 GiB attention scratch on a GPU with 1.8 GiB free).

## 3. Experiments (live server, greedy, synthetic images with a known word)

| test | result |
|---|---|
| sequential A,B,A,B (same prompt text, 206-token images) | all 4 correct |
| concurrent A,B,C,D / A,A,B,B / 2 images + 2 text | all correct; text-only "what word is in this image" hallucinates (DOG, OK) but never another request's word |
| 8 frames × 1280×720 per request (4,702 tokens, pipelined prefill), sequential ×2, concurrent ×4, concurrent ×4 again on reused slots | 10/10 requests list exactly their own 8 words |
| 4-user randomized audit, seed 1, CED=1 (shared 2.5K system prompt, 30 turns each, 48 images) | 0/120 leaks of another user's secret; **45/120 answers missing the user's own secret, almost all image questions answered with a filler word ("COMMITTEE", "REGISTER")** |
| image last in the prompt, system prompt length swept | correct at 597–2,535 and 2,655–3,855 tokens; wrong at 2,055, 2,295, 3,958–3,979, 4,039, 4,087, 4,095 ("committee", "ANNUAL", "REGISTER", "OK"); image *first* then 2.5K text: always correct; a text needle in the last 30 tokens: always correct |
| same failing prompt after different predecessors (5.9K text / 151-token text / itself / 3K text; a unique marker word in the predecessor's tail) | wrong every time, the answer is a word of the shared filler, never the marker |
| **`DSV41_CED=0`**, the nine failing prompts | **9/9 correct** |
| 4-user randomized audit, seed 2, `DSV41_CED=0` | 0/120 leaks; 22/120 strict mismatches, all OCR-level near-misses of the user's own word ("VIOLINOO" for VIOLIN00, "KOALA" for KOALA30), none a filler word |

## 4. Mechanism

`_forward_sequential` (prompts of 2,048–4,095 tokens): stage 1 runs layers 0–20 over all S tokens (this writes the
compressed kv of the kv-source layers 2/8/14/20, the index keys — also owned by those layers — and the candidates of
layer 20, so the long-range part is complete); stage 2 runs layers 21–39 over `h[:, -128:]` with
`start_pos = S − 128`. For a tail position S−128+j the sliding window is [S−255+j, S−128+j]: 127−j of those
positions were never processed by layers 21–39 for this prompt, so their window kv is whatever the last prefill left
in the slot-0 rings (the previous request's tail). The tail's hidden states at layers 21–39 are therefore computed
against the previous request's context; the decode step then copies them (slot 0 → decode slot) as the prompt's
state. `forward_pipelined` (≥ 4,096 tokens) does the same per chunk: layers 21–39 skip every chunk but the last, so
the first 128 positions of the last chunk see stale windows and any image straddling the last chunk boundary is split
between "encoder-only" and "contaminated" positions — the production image prompts (6.4–6.9K tokens, 6–7 frames)
are in this regime.

Why images and not text: text in the tail is still mostly recoverable from the correct compressed kv and index
selection; an image's word has to be read from ~110 image tokens whose layer-21–39 states are the corrupted ones,
and the question attends to them through the window. The wrong answers are always words from the request's own
text — not the predecessor's — because the kv *values* of the compressed caches are the request's own; only the
tail's processing is wrong.

## 5. Cost of the exact prefill

Vision prefill (image requests), from the server log (`[vision-prefill]` lines), same machine, same slots:

| prompt | CED=1 (approximate, before) | CED=0 (exact, now) | ratio |
|---|---|---|---|
| 2,055–2,295 tokens | – | 5.4–6.5 s (350–380 tok/s) | |
| 3,975 tokens (sweep prompts, 8 runs) | 5.0–5.6 s (710–795 tok/s) | 8.7–9.6 s (416–460 tok/s) | 1.7× |
| 4,690 tokens (8 frames) | ≈ 10.4–10.9 s under 4-way contention | 15.1 s idle (312 tok/s) | |
| 6,439 tokens (11 frames) vs production 6.4–6.9K prompts (median of the last 20) | 8.8 s (729 tok/s) | 19.8 s (326 tok/s) | 2.2× |

Decode speed is unchanged (CED only affects prefill). Text prefills pay the same factor on prompts ≥ 2,048 tokens;
shorter prompts never used CED. So the exact fix costs roughly +4 s on a 4K prompt and +11 s on a 6.5K image
prompt per request. The approximate path can be re-enabled per server with `DSV41_CED=1`, with the behaviour
documented in §3–4.

## 6. Changes

* `run_128k_4slots.sh`: `DSV41_CED=0` (exact prefill). `run_mtp4_4gpu.sh` / `run_mtp_4gpu.sh` set `DSV41_CED=1`
  as well — change them the same way if they are used.
* `dsv41/engine.py`: after a failed prefill, forget slot 0's prompt record, window log and exact-prefix continuation
  state (the batched worker only dropped the target slot's record).
* Options if the CED prefill speed is needed back: raise `DSV41_CED_ENCODER_LAST_LAYER` (fewer skipped layers; still
  approximate), or make the stage-2 window mask refuse positions before the tail (removes the cross-request
  contamination but keeps the approximation that already fails on tail images). Neither is exact; not implemented.

## 7. Follow-up (same day): keep CED for text, never for images

Per the user's decision the run scripts are back on `DSV41_CED=1`, with two code changes in `dsv41/model.py`:
image prompts (`images is not None`) always take the exact prefill in both the sequential and the pipelined path,
and before a CED tail / last-chunk pass the decoder-side (layers > `DSV41_CED_ENCODER_LAST_LAYER`) window rings of
the prefill row are zeroed, so the tail attends to zero keys/values at the positions it never processed instead of to
the previous request's kv. CED remains an approximation for the last 128 tokens of a text prompt (documented in the
README); it no longer carries another request's data.

Verification after the restart (`DSV41_CED=1`, new code): the six previously failing image-last prompts (2,055 /
2,295 / 3,975–4,095 tokens) read the image correctly 6/6 at exact-prefill speed (440–470 tok/s at 4K); text needles
in the tail and questions about the far start of a 1.5K / 3.4K / 5.3K prompt all correct under CED; randomized
4-user audit seed 3: 0/120 leaks, 5/120 strict mismatches (OCR near-misses), 0 errors.

## 8. Follow-up (2026-10-05): a concrete misdescribed image

The user reported a 1810×2466 anime image (short blonde bob, torn black top, blue denim skirt, torn black leggings,
brown boots) described as "long blonde hair past her shoulders, light-blue button-down blouse". Findings:

* **ViT OOM on large images.** The vision encoder called `scaled_dot_product_attention` with 3-D inputs, which takes
  the math kernel and materializes the n×n scores: 9.3 GiB peak for this image's 8,268 patches (measured on GPU 6),
  → `CUDA out of memory` on the shared GPU 2 (≈ 2.6 GiB free). With 4-D inputs the flash kernel runs in 0.02 GiB,
  outputs equal within bf16 rounding (max |Δ| 4.9e-4). Fixed (`dsv41/vision.py`).
* **Engram gate not shut for image tokens.** The reference model passes `token_mask` into the Engram layer so image
  positions get no Engram contribution; our port masked the n-gram hashes but left the gate open, so every image
  token in every Engram layer received the pad-n-gram embedding. Fixed at all four prefill call sites
  (`dsv41/engram.py`, `dsv41/model.py`).
* **ViT rotary cos/sin in bf16** (reference: fp32). Now fp32. Encoder output vs the reference implementation on this
  image: cosine 0.99983 (max |Δ| 0.029 at RMS 0.046) after the fixes, 0.99975 before; the reference is bit-exact
  against itself.
* Image preprocessing (`plan_image_grid`, `load_image`, token types) is identical to the reference: this image is
  resized to 1092×1484 → a 36×26 token grid (974 tokens), the model's maximum; the server does not need to shrink
  images further for memory once the flash kernel is used.

Result on the user's image (`Describe this image in detail ...`, greedy), after the fixes:

| input | before (flash fix only, Engram gate open) | after (Engram mask + fp32 rope) |
|---|---|---|
| full 1810×2466 (974 image tokens) | "long, voluminous blonde hair ... high ponytail", "white sleeveless crop top", "denim shorts" | "short, chin-length blonde hair styled in a bob", "fitted black t-shirt with short sleeves that have jagged, torn edges", "brown belt with a large gold buckle" |
| 905×1233 (720 tokens) | – | same, correct |
| 724×986 (486 tokens) | "short chin-length bob" but "white top" | "short bob", "black short-sleeved t-shirt", correct |

The misdescription was therefore not cross-talk and not the image size: with the Engram gate open on image tokens
the LLM saw image embeddings perturbed by the pad-n-gram Engram vector (974 perturbed positions for a large image,
fewer for a small one), and described a generic character instead. Pre-shrinking images below the model's own cap
is not needed and would discard detail the model is trained on.

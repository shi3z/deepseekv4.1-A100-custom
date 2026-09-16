# DeepSeek-V4.1-Flash on A100 (sm80)

DeepSeek-V4.1-Flash のチェックポイントを A100 上で動かす独自推論ランタイムです。FP8 の dense weights と FP4 の MoE experts を扱い、Triton / CUDA C のカーネルで Ampere の BF16 tensor cores を利用します。Engram の大きなテーブルはホスト RAM に置きます。

**更新日: 2026-09-16。** 5 GPU EP、prefix snapshot/block replay、DSpark MTP を同一サーバで測定した結果を記載します。数値は `results/prefix-audit-20260916/` の実サーバログに対応します。

## 現在の到達点

- 40 層・384 experts を 5 GPU に分散。共有ログでは GPU 順序が `2,0,1,3,4`、expert shards が `77,77,77,77,76`、dense layers が各 GPU 8 層。
- 論理コンテキスト上限は `--max-seq-len 1048576` に設定可能。compressed KV / index cache は小さく初期確保し、必要に応じて拡張する。**1M tokens の実処理完走は未確認。** 現行5 GPU構成では約82K〜118K tokenのcold prefillでVRAM不足になる。
- prefix の共通部分を snapshot から復元し、残りを複数 token の block で replay できる。128 block では生成・tool call・HTTP 200 まで確認済み。
- block=512 の replay は長い区間で **8,320 tokens / 21.869 s = 380.4 tok/s**。短い134-token tailは137.9 tok/s。scalar replayの約45–50 tok/sより高速。
- rolling / multi-anchor snapshot が実装済み。512 block と併用し、複数anchorのSAVE / STOREを確認済み。
- DSpark/MTPのdraftとverificationは実装済み。`--mtp 5 --mtp-device 4`で短中長文のHTTP 200とdraft acceptanceを確認した。VRAM余裕がない長大cold promptではDSparkを解放してplain decodeへfallbackする。

## 起動

### 必要な環境

- NVIDIA A100 80GB と GPU 間 P2P。現在の長コンテキスト構成は 5 枚。必要 VRAM はキャッシュ容量、MTP、prefill の一時領域にも依存する。
- `float8_e8m0fnu` を持つ PyTorch、Triton、`sm_80` 向けコンパイルができる nvcc、transformers / tokenizers / sympy。従来の動作環境は PyTorch 2.13+cu130、Triton 3.5 以降。nvcc は `DSV41_NVCC` で指定可能。
- チェックポイント。既定パスは `/mnt/ssd/models/DeepSeek-V4.1-Flash`。異なる場合は `--ckpt` を指定する。
- Engram テーブルだけで約 189 GiB のホスト RAM。prefix snapshot と tmpfs、ロード時の一時領域を加えた余裕が必要。

### 5 GPU / EP / 長コンテキスト

以下は、実測済みの block size 256 と、保存負荷を抑えるための anchor 最大 2 本を組み合わせた起動例です。最新共有ログの測定は anchor 最大 4 本であり、この例と完全に同じ条件ではありません。GPU 番号は環境に合わせて変更してください。

```bash
# nvcc の場所が異なる場合は変更する
export DSV41_NVCC=/usr/local/cuda-12.8/bin/nvcc

# cold prefill の一時メモリを抑える（現行の既定値を明示）
export DSV41_MOE_PREFILL_CHUNK=256
export DSV41_ENGRAM_PREFILL_CHUNK=256
export DSV41_HC_PREFILL_CHUNK=2048
export DSV41_EP_COMPACT_XQ=1

export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=65536
export DSV41_EXACT_CACHE_GROW=1

export DSV41_DISABLE_PREFIX_CACHE=0
export DSV41_PREFIX_CACHE_DIR=/dev/shm/dsv41-prefix-cache
# RAM と /dev/shm の空き容量に合わせて調整する（GiB）
export DSV41_PREFIX_CACHE_ENTRIES=16
export DSV41_PREFIX_CACHE_GB=32
export DSV41_PREFIX_TMPFS_ENTRIES=16
export DSV41_PREFIX_TMPFS_GB=32

export DSV41_PREFIX_BLOCK_REPLAY=1
export DSV41_PREFIX_BLOCK_SIZE=512
export DSV41_PREFIX_BLOCK_MIN=16
export DSV41_PREFIX_ANCHOR_STRIDE=1024
export DSV41_PREFIX_ANCHOR_MAX=2
export DSV41_DEBUG_PREFIX_ANCHORS=1
export DSV41_DEBUG_PREFIX_SYNC=0
export DSV41_MTP_LONG_PROMPT_LIMIT=65536
export DSV41_MTP_UNLOAD_ON_LONG=1
unset CUDA_LAUNCH_BLOCKING

python -m dsv41.serve \
  --ckpt /mnt/ssd/models/DeepSeek-V4.1-Flash \
  --devices 2,0,1,3,4 \
  --ep --ep-shards 77,77,77,77,76 \
  --max-seq-len 1048576 \
  --host 127.0.0.1 --port 8000 \
  --mtp 5 --mtp-device 4
```

上記の cache 上限は運用例として明示しています。RAM 内と tmpfs は別枠であり、一時コピーや Engram のメモリは含みません。`df -h /dev/shm` と `free -h` で空きを確認してください。tmpfs 保存が不要なら `export DSV41_PREFIX_CACHE_DIR=""` にします。GPU 番号は PyTorch から見える番号で、`CUDA_VISIBLE_DEVICES` を設定した場合は再番号付けされます。

この例ではMTPを5 draftで有効にしています。`DSV41_MTP_LONG_PROMPT_LIMIT`を超えるcold promptでは、MTPを一時停止してDSparkを解放します。prefix guardは既定の256 tokensです。

環境変数の変更はサーバ再起動で反映します。現行コードには block replay と multi-anchor の併用修正が入っており、起動時の再パッチは不要です。

### API / 対話 CLI

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "東京タワーの高さは？"}],
    "max_tokens": 256, "temperature": 0.6, "stream": true
  }'
```

`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/completions`、`GET /health` を提供します。OpenAI クライアントでは `base_url="http://127.0.0.1:8000/v1"` を指定できます。

chat は system / user / assistant / tool messages、`max_tokens`（または `max_completion_tokens`）、`temperature`、`top_p`、`stop`、`seed` を扱います。`thinking: true` または `reasoning_effort` を指定すると thinking mode を選択し、reasoning を `reasoning_content` に分離します。生成された tool call は OpenAI 形式に変換します。画像入力は未対応です。

**chat の `stream: true` は現在、生成と completion の解析が終わってから SSE を送信します。token 生成と同時の逐次配信ではありません。** 長い prefill 中はレスポンス開始まで待ちます。raw `/v1/completions` は別の逐次出力経路です。生成は engine lock で直列化され、サーバの continuous batching は未実装です。

```bash
# 対話 CLI（短いコンテキストの動作確認）
python -m dsv41.chat --devices 2,0,1,3,4 --ep

# 単発生成・プロファイリング
python -m dsv41.run --devices 2,0,1,3,4 --ep --decode graph --chat \
  --prompt "日本で一番高い山と、その標高を教えてください。"
```

サーバの `--max-seq-len` の既定値は 8192 です。長コンテキストでは明示指定してください。

## キャッシュの仕組みとログの読み方

### 初期確保と EP preallocation は別段階

モデルロード開始時の次の表示は、loader による初期確保です。

```text
dynamic cache: initial logical capacity 32,768/1,048,576 tokens
compressed cache allocated at startup: 0.49 GiB (1M full capacity would be 15.63 GiB)
```

`DSV41_EP_PREALLOC_TOKENS=65536` は、40層のweight load後、`EPRuntime`初期化時・CUDA graph capture前に適用されます。`DSV41_EXACT_CACHE_GROW=1`では、長いprefill時に必要行数まで拡張します。5 GPUの上記構成で期待される行数は次のとおりです。

| cache owner | 圧縮比 | loader 初期行数 | 131,072 tokens の事前確保行数 |
|---|---:|---:|---:|
| 2, 8, 14 | 2 | 16,385 | 65,536 |
| 20 | 1 | 32,769 | 131,072 |

```text
[ep-prealloc-exact] compress_kv owner=2 rows=16,385->65,536 mirrors=5
[ep-prealloc-exact] index_k owner=2 rows=16,385->65,536 mirrors=5
[ep-prealloc] owner=2 ratio=2 tokens=131,072 rows=65,536
```

owner 8 / 14 / 20 についても確認します。`0.49 GiB` は prealloc 前の compressed cache の数字であり、総 VRAM 使用量でも prealloc 後の量でもありません。事前確保範囲内の 59K prompt で `[cache-grow]` が出る場合は、対象 cache と設定を調査する必要があります。1M の論理上限と、131K の事前確保範囲は同じではありません。

### Prefix reuse / block replay / multi-anchor

1. 入力 token 列と保存済み prefix を比較し、利用可能な snapshot を探す。
2. snapshot の cache state を復元し、base から入力末尾までを replay する。
3. block replay が有効なら continuation prefill を複数 token ずつ実行する。短い端数などは scalar 経路を使う。
4. 更新条件を満たす場合は途中の anchor で snapshot を作り、新しい base を保存する。次回の一致範囲に応じて再利用する。

```text
[prefix-cache] LCP-HIT old=66,265 new=68,844 lcp=66,264 base=64,244 replay=4,600
[prefix-snapshot] RESTORE slots=74
[prefix-anchor] PLAN ... anchors=...
[prefix-replay-block] ... block=256 ... tok_s=...
[prefix-anchor] SAVE base=...
[prefix-anchor] STORE base=...
[prefill-bench] mode=LCP-HIT new=... reused=... total=... time=...
```

block を大きくするだけでは、base が古いままなら replay 量が増え続けます。multi-anchor は base を前進させつつ、末尾の書き換えにも対応するための仕組みです。次回の prompt が anchor まで一致しなければ、その anchor は再利用できません。

以前の `TypeError: '<' not supported between instances of 'int' and 'list'` は、block側が単一の`snapshot_at`を想定していたためです。現行`engine.py`は`_capture_prefix_anchor()`と`_snapshot_positions`を共用し、未保存anchorをblockが飛び越えないよう境界を切っています。

### 主な環境変数

既定値は現行コードの値です。起動例の設定値とは区別してください。

| 変数 | 既定値 | 用途 |
|---|---:|---|
| `DSV41_NVCC` | `/usr/local/cuda-12.8/bin/nvcc` | CUDA kernel のコンパイラ |
| `DSV41_MOE_PREFILL_CHUNK` | `256` | MoE prefill の chunk サイズ |
| `DSV41_ENGRAM_PREFILL_CHUNK` | `256` | Engram prefill の chunk サイズ |
| `DSV41_HC_PREFILL_CHUNK` | `2048` | hyper-connection prefill の chunk サイズ |
| `DSV41_EP_COMPACT_XQ` | `1` | EP で必要な activation rows のみ転送 |
| `DSV41_CACHE_INIT_TOKENS` | `32768` | loader の初期論理 cache 容量 |
| `DSV41_EP_PREALLOC_TOKENS` | `40000` | EP graph capture前の事前確保範囲 |
| `DSV41_EXACT_CACHE_GROW` | `0` | `1`でcacheを幾何倍ではなく必要行数まで拡張 |
| `DSV41_PREFIX_BLOCK_REPLAY` | `0` | `1` で block replay 有効 |
| `DSV41_PREFIX_BLOCK_SIZE` | `128` | replay block の token 数 |
| `DSV41_PREFIX_BLOCK_MIN` | `16` | block 経路を使う最小 token 数 |
| `DSV41_MTP_LONG_PROMPT_LIMIT` | `65536` | 超過時にMTPを停止してplain decodeへfallback |
| `DSV41_MTP_UNLOAD_ON_LONG` | `1` | 長大prompt fallback時にDSparkをGPUから解放 |
| `DSV41_PREFIX_GUARD` | `256` | prompt 末尾から snapshot base までの余裕（最小 32） |
| `DSV41_PREFIX_REFRESH_TOKENS` | `2048` | base 更新を検討する replay 量 |
| `DSV41_PREFIX_ANCHOR_STRIDE` | `1024` | 更新時の anchor 間隔 |
| `DSV41_PREFIX_ANCHOR_MAX` | `4` | 1 回の更新で作る anchor 最大数 |
| `DSV41_PREFIX_CACHE_ENTRIES` | `64` | RAM 内 cache の最大エントリ数（STORE 時） |
| `DSV41_PREFIX_CACHE_GB` | `128` | RAM 内 cache 容量の上限（GiB、STORE 時） |
| `DSV41_PREFIX_CACHE_DIR` | `/dev/shm/dsv41-prefix-cache` | 再起動時の再利用用保存先。空文字で無効 |
| `DSV41_PREFIX_TMPFS_ENTRIES` | `512`* | 保存先の最大エントリ数 |
| `DSV41_PREFIX_TMPFS_GB` | `512` | 保存先容量の上限（GiB） |

RAM cache の再起動時 LOAD は、未設定時に別の既定値（512 entries / 256 GiB）を使います。上記の起動例では両変数を明示し、LOAD / STORE の上限を揃えています。

\* `DSV41_PREFIX_CACHE_ENTRIES` が明示設定されている場合、tmpfs の既定エントリ数もその値を使います。

`DSV41_DISABLE_PREFIX_CACHE=1` で prefix reuse を無効にできます。chat request 単位の切り分けには `X-DSV41-Prefix-Cache: off` ヘッダもあります。詳細確認は `DSV41_DEBUG_PREFIX_ANCHORS=1`、同期デバッグは `DSV41_DEBUG_PREFIX_SYNC=1`。速度測定時は同期デバッグを無効にします。

## 最新の長コンテキスト測定

以下は 2026-09-16 に稼働中の5 GPU EPサーバ（`--mtp 5 --mtp-device 4`、block=512）へ固定形式のpromptを送り、HTTP requestとサーバログから採った値です。**replay / prefill throughput と生成 decode throughput は異なる指標です。**

| replay 方式 | replay tokens | 時間 | replay tok/s |
|---|---:|---:|---:|
| scalar | — | — | 約 45–50 |
| block 128 | 1,270 | 16.196 s | 78.4 |
| block 128 | 2,021 | 22.935 s | 88.1 |
| block 256 + multi-anchor | 6,883 | 48.906 s | 140.7 |
| block 512 + MTP + multi-anchor | 8,320 | 21.869 s | **380.4** |
| block 512 + MTP (prefix HIT tail) | 134 | 0.972 s | **137.9** |

256 block の途中経過は約 144–145 tok/s。block=512 の長い replay では、アンカー保存を含む区間で 380.4 tok/s、短い prefix HIT tail では 137.9 tok/s でした。短い区間は固定オーバーヘッドの影響が大きいため、長い replay の値と直接比較しないでください。

最新の request 全体の prefill は以下でした。

```text
[prefill-bench] mode=LCP-HIT new=8,320 reused=8,072 total=16,392 time=24.805s new_tok_s=335.4 effective_tok_s=660.8
[prefill-bench] mode=LCP-HIT new=134 reused=16,264 total=16,398 time=0.999s new_tok_s=134.1 effective_tok_s=16,410.5
```

`new_tok_s` は replay 対象 tokens / prefill 全体時間、`effective_tok_s` は再利用分を含む総 prompt tokens / prefill 全体時間です。後者を新規 token の計算速度として扱わないでください。

### 現在のボトルネック

- snapshot は使用済み行だけではなく、対象 tensor の確保済み領域を丸ごと CPU にコピーする。今回のログでは 1 本 **1,605.1 MiB**、74 slots。
- tmpfs への SAVE は 1 本約 1.05–1.11 s。4 本で約 4.3 s かかり、replay 計測の 48.906 s と prefill 全体の 53.319 s の差約 4.4 s に近い。replay 内での snapshot 作成時間もあるため、差分を snapshot 全処理の時間とはみなさない。
- RAM cache は 19 entries で **29.78 GiB**。tmpfs は別途 RAM を消費するため、RAM cache の表示だけで総消費量を判断しない。上限設定は実マシンの空き RAM / `/dev/shm` 容量に合わせる。

今回の測定では `DSV41_PREFIX_BLOCK_SIZE=512`、`DSV41_EP_PREALLOC_TOKENS=65536`、`DSV41_EXACT_CACHE_GROW=1`を使用しました。アンカー保存を含むreplayではsnapshot書き込み時間も含まれます。anchor最大数を4→2にする案は新規snapshotの本数を減らしますが、既存エントリの総量を直ちに半減させる設定ではありません。

117,896 tokenのcold promptでは、MTPを自動fallbackしてDSparkを解放しても、compressed/index cacheがGPU容量を使い切りHTTP 507（CUDA OOM）になりました。これはprefix HIT後の短いreplayの速度とは別の制約です。現行構成で安定して完走を確認できた最大系列は約16K tokenです。

中長期の改善項目は、snapshot を logical used rows のみ保存する方式への変更です。block 拡大による一時 VRAM 増加、長コンテキストでの attention / indexer のコストも引き続き測定対象です。

## 過去の測定: decode / batch / MTP

この節は 9 月 11–12 日のベンチログ・Claude Code 作業記録に基づく参考値です。短い cache 長、GPU 数、バッチ数が異なり、上記の 5 GPU・75K prompt の性能を示すものではありません。

| 構成 | throughput | 備考 |
|---|---:|---|
| 8 GPU layer pipeline、単一 stream | 約 52 tok/s | CUDA graphs / FP8・FP4 tensor-core kernels |
| 7 GPU EP、単一 stream | 約 62–66 tok/s | 過去の decode 測定 |
| 1 A100 + CPU experts / hot experts | 約 32–35 tok/s | CPU・prompt・routing profile に依存 |
| 4 GPU EP、S=32、MTP 無効 | 約 600 tok/s | aggregate、cache 長 2048 |
| 4 GPU EP、S=256、MTP 無効 | 1,972 tok/s | aggregate、cache 長 2048 |
| 4 GPU EP、S=512、MTP 無効 | 2,331 tok/s | aggregate、cache 長 1024 |
| 4 GPU EP、S=1024、MTP 無効 | 2,415 tok/s | aggregate、cache 長 512 |

大バッチでは cache 長を縮めており、S=1024 の単一 sequence あたりは約 2.4 tok/s です。複数 replica への単純な倍算は実測値とは区別します。

`dsv41/mtp_run.py` は DSpark draft を batched verification するベンチ経路です。過去の 4 GPU・32 mixed prompts・K=3 の測定では aggregate 527 tok/s（同測定系列の MTP 無効 466 tok/s）。後続の最適化で測定値が変わるため、条件とログを併せて参照してください。旧 README の「MTP verification 未実装」という記述は現在には当てはまりません。

単一 GPU の `--offload-experts cpu` は experts をホスト RAM に置いて CPU で計算し、`--hot-experts` は一部を GPU に保持します。`--offload-experts gpu` は PCIe 転送を伴う別経路です。CPU offload 時は Engram に加えて experts 約 269 GiB 分の RAM が必要です。

## 実装の配置

| ファイル | 役割 |
|---|---|
| `dsv41/load.py`, `stio.py`, `quant.py` | チェックポイント読込、GPU 配置、量子化形式、初期 cache 確保 |
| `dsv41/model.py` | 40 層 backbone、compressed sparse attention、window cache、MoE、continuation prefill |
| `dsv41/ep.py` | expert parallel runtime、P2P 同期、exact preallocation |
| `dsv41/decode.py` | static-shape decode / batched rows / CUDA graphs |
| `dsv41/engine.py` | 生成、prefix reuse、block replay、multi-anchor、RAM / tmpfs snapshot、MTP verification |
| `dsv41/dspark.py`, `mtp_run.py` | DSpark draft、batched speculative decoding ベンチ |
| `dsv41/serve.py`, `chat.py`, `run.py` | HTTP API、対話 CLI、単発実行 |
| `dsv41/engram.py` | n-gram hash とホスト常駐 Engram テーブル |
| `dsv41/moe_kernels.py`, `fused.py`, `fused2.py`, `kernels.py` | Triton GEMM / attention / 融合演算 |
| `dsv41/cuda/`, `cukern.py` | FP4 / FP8 CUDA kernels と起動処理 |
| `dsv41/cpu/`, `hotcache.py` | CPU experts と GPU hot-expert cache |
| `results/` | 過去のベンチ・プロファイルログ |

FP8 dense weights は対応経路で packed のまま保持し、BF16 tensor-core operands に変換します。FP4 experts も packed のまま保持し、decode / prefill に応じたカーネルを使います。主要な重み容量の目安は experts 約 269 GiB、dense を BF16 化した場合約 13.6 GiB、embedding / output head 約 2.5 GiB、Engram テーブル約 189 GiB。cache、mirror、MTP、一時領域は別途必要です。

## 既知の制約・切り分け

- 1M context は設定上限。実用上限は空き VRAM、cache / snapshot 容量、prefill 時間を含めて検証する。
- multi-anchor と block replay の併用は上記の実行例で確認済み。幅広い入力での出力一致・長時間運用まで保証するものではない。
- `deepseek_v41` と空の model type の不一致警告、および `torch.frombuffer` の non-writable buffer 警告は共有ログに出ている。これだけで prealloc 失敗とは判断せず、その後の weight load・EP prealloc・生成結果を確認する。
- サーバの continuous batching と vision encoder は未実装。chat SSE の送信開始は生成完了後。
- 最新の長コンテキスト測定は本 README に転記した共有ログを根拠とする。`results/` 内の過去の throughput と混同しない。

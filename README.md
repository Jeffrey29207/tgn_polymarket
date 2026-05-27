# Temporal Graph Learning for Polymarket Arbitrage Research

This repository adapts the Temporal Graph Network (TGN) codebase for a research project on
discovering logical and price-based arbitrage candidates in Polymarket prediction markets.

The project treats Polymarket as a temporal heterogeneous graph. Markets, outcome tokens, and
optional topics are nodes; OHLCV observations, complementary outcomes, topic membership, verified
logical dependencies, and price co-movement candidates are timestamped edges. A TGN model is then
trained to produce temporal embeddings and link scores that can help rank candidate relationships
for downstream symbolic verification.

The original TGN implementation is from:

> Temporal Graph Networks for Deep Learning on Dynamic Graphs  
> Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, Michael Bronstein

## Research Goal

The intended research workflow is:

1. Convert raw Polymarket data into a temporal graph.
2. Learn node and edge dynamics from market price histories.
3. Use symbolic verifier outputs as trusted labels for logical relationships where available.
4. Rank unseen market-market or token-token pairs as arbitrage candidates.
5. Inspect high-scoring candidates with domain logic and symbolic checks.

The model is not itself the source of truth for logical arbitrage. It is a candidate discovery and
ranking component. Symbolic verification should remain the authority for whether a relationship is
logically valid.

## Graph Design

The Polymarket-specific preprocessing code builds graph objects such as:

- `market:<condition_id>`: one market node per Polymarket condition id.
- `token:<clob_token_id>`: one outcome token node per tradeable CLOB token.
- `topic:<topic_key>`: optional coarse event/topic nodes inferred from question text.

The main edge types are:

- `market -> token`: OHLCV observation edges. These carry open, high, low, close, volume, returns,
  market liquidity, time-to-end, and optional orderbook features.
- `token:YES -> token:NO`: optional complement edges for binary markets.
- `topic -> market`: optional topic membership edges.
- `market -> market`: optional dependency edges from a symbolic verifier CSV.
- `token -> token`: optional verifier dependency edges or correlation/co-movement candidate edges.

Important detail: OHLCV features are stored on temporal edges, not on token nodes. Token nodes carry
static identity/type features, while each edge records what happened at a particular timestamp.

## Raw Data Expected

The preprocessor expects a Polymarket data root similar to:

```text
C:/Users/User/Downloads/poly_data/
  markets/*.parquet
  data/freqtrade_pair_mapping.csv
  data/data/*.feather
  polymarket_orderbooks-002.jsonl        optional
  polymarket_markets_1y.jsonl            optional fallback
```

The key files are:

- `markets/*.parquet`: market metadata, including `condition_id`, `question`, `outcomes`,
  `clob_token_ids`, `volume`, `liquidity`, `created_at`, and `end_date`.
- `data/freqtrade_pair_mapping.csv`: maps each sanitized OHLCV filename back to the original
  `condition_id`.
- `data/data/*.feather`: OHLCV time series for individual market outcome tokens.

The OHLCV-to-token assignment is deterministic: filename -> condition id -> market metadata ->
outcome label -> CLOB token id. The code strips date/version/USDC/timeframe suffixes before matching
the filename to the real outcome labels in market metadata.

## Setup

From the parent folder:

```powershell
cd C:\Users\User\Downloads\tgn\tgn_polymarket
..\.venv\Scripts\python.exe --version
```

Core dependencies include:

```text
numpy
pandas
pyarrow
scikit-learn
torch
```

## Preprocessing

Small smoke-test graph:

```powershell
..\.venv\Scripts\python.exe utils\preprocess_polymarket.py `
  --data-root C:/Users/User/Downloads/poly_data `
  --output-dir ./data `
  --output-name polymarket_sample `
  --timeframe 4h `
  --min-market-volume 10000 `
  --max-feather-files 50 `
  --progress-every 10 `
  --include-topic-edges `
  --include-complement-edges
```

Practical capped graph:

```powershell
..\.venv\Scripts\python.exe utils\preprocess_polymarket.py `
  --data-root C:/Users/User/Downloads/poly_data `
  --output-dir ./data `
  --output-name polymarket_4h `
  --timeframe 4h `
  --min-market-volume 10000 `
  --max-edge-rows 1000000 `
  --progress-every 100
```

Richer graph with topic and complement edges:

```powershell
..\.venv\Scripts\python.exe utils\preprocess_polymarket.py `
  --data-root C:/Users/User/Downloads/poly_data `
  --output-dir ./data `
  --output-name polymarket_4h_rich `
  --timeframe 4h `
  --min-market-volume 10000 `
  --max-edge-rows 1000000 `
  --progress-every 100 `
  --include-topic-edges `
  --include-complement-edges
```

With symbolic verifier labels:

```powershell
..\.venv\Scripts\python.exe utils\preprocess_polymarket.py `
  --data-root C:/Users/User/Downloads/poly_data `
  --output-dir ./data `
  --output-name polymarket_4h_verified `
  --timeframe 4h `
  --min-market-volume 10000 `
  --max-edge-rows 1000000 `
  --progress-every 100 `
  --include-topic-edges `
  --include-complement-edges `
  --verified-pairs-csv C:/path/to/verified_pairs.csv
```

If you do not have a verifier CSV yet, omit `--verified-pairs-csv`.

The output files follow the TGN format:

```text
data/ml_<output-name>.csv
data/ml_<output-name>.npy
data/ml_<output-name>_node.npy
data/ml_<output-name>_metadata.json
```

## Verifier CSV Format

Market-level dependency labels may use:

```csv
source_condition_id,target_condition_id,label,relation_type,timestamp
```

Token-level dependency labels may use:

```csv
source_token_id,target_token_id,label,relation_type,timestamp
```

You can also identify token rows by condition/outcome pairs:

```csv
source_condition_id,source_outcome,target_condition_id,target_outcome,label,relation_type,timestamp
```

Supported relation labels are encoded into edge features, including implication, contradiction,
equivalence, mutual exclusion, and other.

## Training

Self-supervised temporal link prediction:

```powershell
..\.venv\Scripts\python.exe train_self_supervised.py `
  -d polymarket_4h `
  --use_memory `
  --prefix poly-4h-small `
  --negative_sampler typed_temporal `
  --n_epoch 2 `
  --bs 100 `
  --n_degree 5 `
  --progress_every_batches 25
```

Useful options:

- `--negative_sampler typed_temporal`: samples type-compatible temporal negatives.
- `--progress_every_batches`: prints training progress during long CPU runs.
- `--memory_dim`: can be omitted; the code uses the node feature dimension automatically.
- `--strict_memory_check`: re-enables strict TGN memory drift assertions for debugging.

The self-supervised ground truth is edge existence: observed temporal edges are positive examples,
and sampled non-edges are negative examples. This is useful for learning temporal graph structure,
but it is not the same as training directly on verified arbitrage labels.

## Interpreting Results

High AP/AUC in self-supervised training means the model can distinguish observed edges from sampled
negative edges. On the current market-token graph, this is a pipeline sanity check rather than proof
of arbitrage discovery.

For the research objective, the more meaningful evaluation is candidate ranking against verifier
labels:

- Does the model rank verifier-positive market/token pairs above verifier-negative or unknown pairs?
- Are high-scoring token-token candidates logically plausible?
- Are high-scoring pairs economically meaningful after fees, spreads, and liquidity constraints?

The helper functions in `evaluation/evaluation.py` include Polymarket candidate scoring utilities
for this later verifier-centered evaluation.

## Files Added or Adapted for Polymarket

- `utils/preprocess_polymarket.py`: converts raw Polymarket metadata, OHLCV, optional orderbooks,
  and optional verifier labels into TGN input files.
- `utils/utils.py`: includes a typed temporal negative sampler for heterogeneous Polymarket nodes.
- `evaluation/evaluation.py`: includes helpers for scoring/ranking Polymarket candidate edges.
- `train_self_supervised.py`: adds Polymarket-friendly progress logging, typed negatives, and
  automatic memory dimension handling.
- `train_supervised.py`: aligns memory dimension and memory checking behavior with the
  self-supervised script.
- `model/tgn.py`: adds optional relaxed memory drift handling for long Polymarket runs.

## Suggested Research Path

1. Build a small smoke-test graph and inspect `ml_<name>_metadata.json`.
2. Train for 1-2 epochs to confirm the pipeline runs.
3. Build the rich graph with topic and complement edges.
4. Add verifier-labeled dependency edges.
5. Evaluate ranking quality on held-out verified pairs.
6. Use high-ranking unverified token-token or market-market pairs as symbolic-verifier candidates.

## Citation

If citing the underlying architecture, cite the original TGN paper:

```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico
    Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```

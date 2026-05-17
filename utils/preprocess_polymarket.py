import argparse
import ast
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EDGE_FEATURE_COLUMNS = [
  "edge_type_market_token",
  "edge_type_complement",
  "edge_type_cross_market",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "log_return",
  "abs_return",
  "high_low_spread",
  "time_to_end_days",
  "market_log_volume",
  "market_log_liquidity",
  "best_bid",
  "best_ask",
  "book_spread",
  "book_mid",
  "bid_depth_top5",
  "ask_depth_top5",
  "book_imbalance_top5",
  "complement_sum_minus_one",
  "cross_return_correlation",
]

# The first ten node features are deliberately fixed and low-dimensional. Any text-derived
# question features are appended after these, so training code can reliably infer node type from
# columns 0 and 1 when it needs type-aware negative sampling.
NODE_BASE_COLUMNS = [
  "node_type_market",
  "node_type_token",
  "side_yes",
  "side_no",
  "market_log_volume",
  "market_log_liquidity",
  "market_active",
  "market_closed",
  "market_lifetime_days",
  "num_outcomes",
]


def safe_literal_list(value):
  """Parse Polymarket fields that are stored as JSON/list-looking strings."""
  if value is None or (isinstance(value, float) and np.isnan(value)):
    return []
  if isinstance(value, list):
    return value
  try:
    return json.loads(value)
  except Exception:
    try:
      return ast.literal_eval(value)
    except Exception:
      return []


def to_unix_seconds(series):
  dt = pd.to_datetime(series, utc=True, errors="coerce")
  return (dt.astype("int64") // 10 ** 9).astype("float64")


def scalar_unix_seconds(value):
  if value is None or pd.isna(value):
    return np.nan
  timestamp = pd.to_datetime(value, utc=True, errors="coerce")
  if pd.isna(timestamp):
    return np.nan
  return timestamp.timestamp()


def log1p_float(value):
  try:
    if value is None or pd.isna(value):
      return 0.0
    return float(np.log1p(max(float(value), 0.0)))
  except Exception:
    return 0.0


def hashed_text_features(text, dimension):
  """Cheap, dependency-free text features for market questions.

  This is not a replacement for semantic embeddings. It gives the TGN a weak lexical signal
  without requiring an external embedding model or network access.
  """
  features = np.zeros(dimension, dtype=np.float32)
  if dimension <= 0 or not isinstance(text, str):
    return features

  tokens = [token.lower() for token in text.replace("-", " ").replace("_", " ").split()]
  for token in tokens:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "little") % dimension
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    features[bucket] += sign

  norm = np.linalg.norm(features)
  return features / norm if norm > 0 else features


class NodeIndexer:
  """Maps Polymarket string identifiers to contiguous TGN node ids.

  TGN reserves node 0 for padding, so real market/token nodes start at 1.
  """
  def __init__(self):
    self.raw_to_idx = {}
    self.idx_to_raw = {0: "__padding__"}

  def get(self, raw_id):
    raw_id = str(raw_id)
    if raw_id not in self.raw_to_idx:
      idx = len(self.raw_to_idx) + 1
      self.raw_to_idx[raw_id] = idx
      self.idx_to_raw[idx] = raw_id
    return self.raw_to_idx[raw_id]


def load_markets(data_root, max_markets=None):
  """Load Polymarket market metadata from parquet shards, falling back to JSONL if needed."""
  markets_dir = data_root / "markets"
  parquet_paths = sorted(p for p in markets_dir.glob("*.parquet") if not p.name.startswith("._"))

  frames = []
  if parquet_paths:
    for path in parquet_paths:
      frames.append(pd.read_parquet(path))
      if max_markets is not None and sum(len(frame) for frame in frames) >= max_markets:
        break
    markets = pd.concat(frames, ignore_index=True)
    if max_markets is not None:
      markets = markets.head(max_markets)
  else:
    jsonl_path = data_root / "polymarket_markets_1y.jsonl"
    rows = []
    with jsonl_path.open(encoding="utf-8") as fh:
      for line_idx, line in enumerate(fh):
        if max_markets is not None and line_idx >= max_markets:
          break
        rows.append(json.loads(line))
    markets = pd.DataFrame(rows)
    markets = markets.rename(columns={
      "conditionId": "condition_id",
      "clobTokenIds": "clob_token_ids",
      "outcomePrices": "outcome_prices",
      "createdAt": "created_at",
      "endDate": "end_date",
      "marketMakerAddress": "market_maker_address",
    })

  rename_map = {
    "conditionId": "condition_id",
    "clobTokenIds": "clob_token_ids",
    "createdAt": "created_at",
    "endDate": "end_date",
  }
  markets = markets.rename(columns={k: v for k, v in rename_map.items() if k in markets.columns})
  markets["condition_id"] = markets["condition_id"].astype(str)
  markets = markets.drop_duplicates("condition_id", keep="last")
  return markets


def build_market_lookup(markets):
  """Normalize market metadata and expose condition/token lookup tables."""
  lookup = {}
  token_to_condition = {}

  for row in markets.to_dict("records"):
    condition_id = str(row.get("condition_id"))
    outcomes = safe_literal_list(row.get("outcomes"))
    token_ids = safe_literal_list(row.get("clob_token_ids"))
    side_to_token = {}

    for outcome, token_id in zip(outcomes, token_ids):
      side_to_token[str(outcome).strip().lower()] = str(token_id)
      token_to_condition[str(token_id)] = condition_id

    if "yes" not in side_to_token and len(token_ids) >= 1:
      side_to_token["yes"] = str(token_ids[0])
    if "no" not in side_to_token and len(token_ids) >= 2:
      side_to_token["no"] = str(token_ids[1])

    created_at = scalar_unix_seconds(row.get("created_at"))
    end_date = scalar_unix_seconds(row.get("end_date"))
    lifetime_days = 0.0
    if not np.isnan(created_at) and not np.isnan(end_date):
      lifetime_days = max((end_date - created_at) / 86400.0, 0.0)

    lookup[condition_id] = {
      "condition_id": condition_id,
      "question": str(row.get("question", "")),
      "side_to_token": side_to_token,
      "token_ids": [str(token_id) for token_id in token_ids],
      "volume": float(row.get("volume", 0.0) or 0.0),
      "liquidity": float(row.get("liquidity", 0.0) or 0.0),
      "log_volume": log1p_float(row.get("volume", 0.0)),
      "log_liquidity": log1p_float(row.get("liquidity", 0.0)),
      "active": float(bool(row.get("active", False))),
      "closed": float(bool(row.get("closed", False))),
      "end_date": end_date,
      "lifetime_days": lifetime_days,
      "num_outcomes": float(len(outcomes)),
    }

  return lookup, token_to_condition


def infer_side_from_filename(filename):
  upper = filename.upper()
  if "YES" in upper:
    return "yes"
  if "NO" in upper:
    return "no"
  return "unknown"


def load_pair_mapping(data_root, timeframe):
  mapping_path = data_root / "data" / "freqtrade_pair_mapping.csv"
  mapping = pd.read_csv(mapping_path)
  mapping = mapping[mapping["Timeframe"].astype(str) == timeframe]
  mapping["Original_Condition_ID"] = mapping["Original_Condition_ID"].astype(str)
  return {
    row["New_Filename"]: row["Original_Condition_ID"]
    for _, row in mapping.iterrows()
  }


def orderbook_features(snapshot, depth_levels=5):
  bids = sorted(snapshot.get("bids", []), key=lambda x: float(x.get("price", 0.0)), reverse=True)
  asks = sorted(snapshot.get("asks", []), key=lambda x: float(x.get("price", 0.0)))

  best_bid = float(bids[0]["price"]) if bids else 0.0
  best_ask = float(asks[0]["price"]) if asks else 0.0
  mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
  spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0

  bid_depth = sum(float(level.get("size", 0.0)) for level in bids[:depth_levels])
  ask_depth = sum(float(level.get("size", 0.0)) for level in asks[:depth_levels])
  denom = bid_depth + ask_depth
  imbalance = (bid_depth - ask_depth) / denom if denom > 0 else 0.0

  return {
    "best_bid": best_bid,
    "best_ask": best_ask,
    "book_spread": spread,
    "book_mid": mid,
    "bid_depth_top5": np.log1p(max(bid_depth, 0.0)),
    "ask_depth_top5": np.log1p(max(ask_depth, 0.0)),
    "book_imbalance_top5": imbalance,
  }


def load_orderbook_hourly(data_root, token_ids, max_rows=None):
  """Read order book snapshots and keep the latest snapshot for each token-hour.

  The raw order book JSONL can be very large, so this function streams line by line and only keeps
  compact features for tokens that appear in the selected OHLCV files.
  """
  orderbook_path = data_root / "polymarket_orderbooks-002.jsonl"
  if not orderbook_path.exists():
    return {}

  token_ids = set(str(token_id) for token_id in token_ids)
  latest = {}
  with orderbook_path.open(encoding="utf-8") as fh:
    for row_idx, line in enumerate(fh):
      if max_rows is not None and row_idx >= max_rows:
        break
      snapshot = json.loads(line)
      token_id = str(snapshot.get("token_id") or snapshot.get("assetId"))
      if token_id not in token_ids:
        continue
      timestamp_ms = int(snapshot.get("timestamp") or snapshot.get("indexedAt") or 0)
      hour = (timestamp_ms // 1000) // 3600 * 3600
      latest[(token_id, float(hour))] = orderbook_features(snapshot)

  return latest


def make_edge_features(edge_type, ohlcv=None, market=None, orderbook=None,
                       complement_sum_minus_one=0.0, cross_return_correlation=0.0):
  """Create one edge-feature vector in the exact order expected by EDGE_FEATURE_COLUMNS."""
  features = dict.fromkeys(EDGE_FEATURE_COLUMNS, 0.0)
  features[f"edge_type_{edge_type}"] = 1.0

  if ohlcv is not None:
    for key in ["open", "high", "low", "close", "volume", "log_return", "abs_return",
                "high_low_spread"]:
      features[key] = float(ohlcv.get(key, 0.0))

  if market is not None:
    features["market_log_volume"] = float(market.get("log_volume", 0.0))
    features["market_log_liquidity"] = float(market.get("log_liquidity", 0.0))
    if ohlcv is not None and not np.isnan(market.get("end_date", np.nan)):
      features["time_to_end_days"] = max((market["end_date"] - float(ohlcv["ts"])) / 86400.0, 0.0)

  if orderbook is not None:
    for key in ["best_bid", "best_ask", "book_spread", "book_mid", "bid_depth_top5",
                "ask_depth_top5", "book_imbalance_top5"]:
      features[key] = float(orderbook.get(key, 0.0))

  features["complement_sum_minus_one"] = float(complement_sum_minus_one)
  features["cross_return_correlation"] = float(cross_return_correlation)
  return [features[column] for column in EDGE_FEATURE_COLUMNS]


def add_edge(edges, edge_features, src, dst, ts, label, feature_vector):
  # Edge index 0 is reserved for padding, so the next feature row is also the next real edge id.
  edge_idx = len(edge_features)
  edges.append({
    "u": int(src),
    "i": int(dst),
    "ts": float(ts),
    "label": float(label),
    "idx": edge_idx,
  })
  edge_features.append(feature_vector)


def build_node_features(node_indexer, node_info, text_hash_dim):
  feature_columns = NODE_BASE_COLUMNS + [f"question_hash_{i}" for i in range(text_hash_dim)]
  node_features = np.zeros((len(node_indexer.raw_to_idx) + 1, len(feature_columns)), dtype=np.float32)

  for raw_id, idx in node_indexer.raw_to_idx.items():
    info = node_info.get(raw_id, {})
    base = [
      float(info.get("node_type_market", 0.0)),
      float(info.get("node_type_token", 0.0)),
      float(info.get("side_yes", 0.0)),
      float(info.get("side_no", 0.0)),
      float(info.get("market_log_volume", 0.0)),
      float(info.get("market_log_liquidity", 0.0)),
      float(info.get("market_active", 0.0)),
      float(info.get("market_closed", 0.0)),
      float(info.get("market_lifetime_days", 0.0)),
      float(info.get("num_outcomes", 0.0)),
    ]
    text_features = hashed_text_features(info.get("question", ""), text_hash_dim)
    node_features[idx] = np.concatenate([np.array(base, dtype=np.float32), text_features])

  return node_features, feature_columns


def add_correlation_edges(edges, edge_features, token_series, token_nodes, token_markets,
                          threshold, min_overlap, max_tokens, max_edges_per_token):
  """Add weak positive cross-market edges from high return correlation.

  These edges are intended as candidate-retrieval supervision, not final arbitrage truth. The
  symbolic/logical verification stage should still decide whether a ranked pair is actionable.
  """
  token_ids = sorted(token_series.keys())[:max_tokens]
  added_per_token = defaultdict(int)

  for left_idx, left_token in enumerate(token_ids):
    left_series = token_series[left_token]
    for right_token in token_ids[left_idx + 1:]:
      if token_markets.get(left_token) == token_markets.get(right_token):
        continue
      if added_per_token[left_token] >= max_edges_per_token:
        break
      if added_per_token[right_token] >= max_edges_per_token:
        continue

      joined = left_series.join(token_series[right_token], how="inner", lsuffix="_left",
                                rsuffix="_right").dropna()
      if len(joined) < min_overlap:
        continue

      corr = joined["log_return_left"].corr(joined["log_return_right"])
      if corr is None or np.isnan(corr) or abs(corr) < threshold:
        continue

      ts = float(joined.index.max())
      feature_vector = make_edge_features("cross_market", cross_return_correlation=corr)
      add_edge(edges, edge_features, token_nodes[left_token], token_nodes[right_token], ts, 1.0,
               feature_vector)
      added_per_token[left_token] += 1
      added_per_token[right_token] += 1


def run(args):
  data_root = Path(args.data_root)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  markets = load_markets(data_root, max_markets=args.max_markets)
  market_lookup, _ = build_market_lookup(markets)
  filename_to_condition = load_pair_mapping(data_root, args.timeframe)

  node_indexer = NodeIndexer()
  node_info = {}
  token_nodes = {}
  token_markets = {}
  edges = []
  edge_features = [np.zeros(len(EDGE_FEATURE_COLUMNS), dtype=np.float32).tolist()]
  token_series = {}
  close_by_condition_side_ts = defaultdict(dict)

  feather_dir = data_root / "data" / "data"
  feather_paths = sorted(feather_dir.glob(f"*-{args.timeframe}.feather"))
  if args.max_feather_files is not None:
    feather_paths = feather_paths[:args.max_feather_files]

  needed_token_ids = set()
  for path in feather_paths:
    condition_id = filename_to_condition.get(path.name)
    market = market_lookup.get(str(condition_id))
    side = infer_side_from_filename(path.name)
    if market is None or side not in market["side_to_token"]:
      continue
    needed_token_ids.add(market["side_to_token"][side])

  orderbooks = {}
  if args.include_orderbook:
    orderbooks = load_orderbook_hourly(data_root, needed_token_ids, max_rows=args.max_orderbook_rows)

  for path in feather_paths:
    condition_id = filename_to_condition.get(path.name)
    market = market_lookup.get(str(condition_id))
    side = infer_side_from_filename(path.name)
    if market is None or side not in market["side_to_token"]:
      continue

    token_id = market["side_to_token"][side]
    market_raw = f"market:{condition_id}"
    token_raw = f"token:{token_id}"
    market_node = node_indexer.get(market_raw)
    token_node = node_indexer.get(token_raw)
    token_nodes[token_id] = token_node
    token_markets[token_id] = condition_id

    node_info[market_raw] = {
      "node_type_market": 1.0,
      "market_log_volume": market["log_volume"],
      "market_log_liquidity": market["log_liquidity"],
      "market_active": market["active"],
      "market_closed": market["closed"],
      "market_lifetime_days": market["lifetime_days"],
      "num_outcomes": market["num_outcomes"],
      "question": market["question"],
    }
    node_info[token_raw] = {
      "node_type_token": 1.0,
      "side_yes": 1.0 if side == "yes" else 0.0,
      "side_no": 1.0 if side == "no" else 0.0,
      "market_log_volume": market["log_volume"],
      "market_log_liquidity": market["log_liquidity"],
      "market_active": market["active"],
      "market_closed": market["closed"],
      "market_lifetime_days": market["lifetime_days"],
      "num_outcomes": market["num_outcomes"],
      "question": market["question"],
    }

    frame = pd.read_feather(path).sort_values("date")
    # Convert OHLCV rows into repeated market-token events. This lets TGN observe how each
    # outcome token evolves over time while keeping the graph format compatible with ml_*.csv.
    frame["ts"] = to_unix_seconds(frame["date"])
    frame["log_return"] = np.log(frame["close"].replace(0, np.nan)).diff().replace(
      [np.inf, -np.inf], np.nan).fillna(0.0)
    frame["abs_return"] = frame["log_return"].abs()
    frame["high_low_spread"] = (frame["high"] - frame["low"]).fillna(0.0)
    frame["volume"] = np.log1p(frame["volume"].clip(lower=0))

    series = frame[["ts", "close", "log_return"]].dropna().set_index("ts")
    token_series[token_id] = series

    for row in frame.to_dict("records"):
      ts = float(row["ts"])
      if math.isnan(ts):
        continue
      hour = ts // 3600 * 3600
      orderbook = orderbooks.get((token_id, float(hour)))
      feature_vector = make_edge_features("market_token", row, market, orderbook)
      add_edge(edges, edge_features, market_node, token_node, ts, 0.0, feature_vector)
      close_by_condition_side_ts[(condition_id, ts)][side] = float(row.get("close", 0.0))

  for (condition_id, ts), closes in close_by_condition_side_ts.items():
    # YES/NO complement edges encode the within-market logical relationship. The feature
    # yes_close + no_close - 1 is a direct signal of temporary complement mispricing.
    market = market_lookup.get(str(condition_id))
    if market is None or "yes" not in closes or "no" not in closes:
      continue
    yes_token = market["side_to_token"].get("yes")
    no_token = market["side_to_token"].get("no")
    if yes_token not in token_nodes or no_token not in token_nodes:
      continue
    complement_sum_minus_one = closes["yes"] + closes["no"] - 1.0
    feature_vector = make_edge_features("complement", market=market,
                                        complement_sum_minus_one=complement_sum_minus_one)
    add_edge(edges, edge_features, token_nodes[yes_token], token_nodes[no_token], ts, 1.0,
             feature_vector)

  if args.add_correlation_edges:
    add_correlation_edges(edges, edge_features, token_series, token_nodes, token_markets,
                          args.correlation_threshold, args.min_correlation_overlap,
                          args.max_correlation_tokens, args.max_correlation_edges_per_token)

  if len(edges) == 0:
    raise ValueError(
      "No Polymarket edges were produced. Check that freqtrade_pair_mapping.csv, "
      "market metadata, and the selected timeframe refer to overlapping condition IDs."
    )

  graph_df = pd.DataFrame(edges).sort_values(["ts", "idx"]).reset_index(drop=True)
  # After sorting by time, reassign contiguous edge ids and reorder the feature matrix to match.
  # model/tgn.py indexes edge_raw_features by this idx column during every forward pass.
  old_edge_idxs = graph_df["idx"].astype(int).values
  graph_df["idx"] = np.arange(1, len(graph_df) + 1)
  edge_features = [edge_features[0]] + [edge_features[int(old_idx)] for old_idx in old_edge_idxs]

  node_features, node_feature_columns = build_node_features(node_indexer, node_info,
                                                            args.text_hash_dim)

  graph_path = output_dir / f"ml_{args.output_name}.csv"
  edge_feature_path = output_dir / f"ml_{args.output_name}.npy"
  node_feature_path = output_dir / f"ml_{args.output_name}_node.npy"
  metadata_path = output_dir / f"ml_{args.output_name}_metadata.json"

  graph_df.to_csv(graph_path, index=False)
  np.save(edge_feature_path, np.array(edge_features, dtype=np.float32))
  np.save(node_feature_path, node_features)

  metadata = {
    "node_map": node_indexer.raw_to_idx,
    "idx_to_raw_node": node_indexer.idx_to_raw,
    "edge_feature_columns": EDGE_FEATURE_COLUMNS,
    "node_feature_columns": node_feature_columns,
    "timeframe": args.timeframe,
    "num_edges": int(len(graph_df)),
    "num_nodes": int(node_features.shape[0] - 1),
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

  print(f"Wrote {graph_path}")
  print(f"Wrote {edge_feature_path}")
  print(f"Wrote {node_feature_path}")
  print(f"Wrote {metadata_path}")
  print(f"Nodes: {metadata['num_nodes']}, edges: {metadata['num_edges']}")


def parse_args():
  parser = argparse.ArgumentParser("Preprocess Polymarket data into the TGN ml_* format")
  parser.add_argument("--data-root", default="C:/Users/User/Downloads/poly_data")
  parser.add_argument("--output-dir", default="./data")
  parser.add_argument("--output-name", default="polymarket")
  parser.add_argument("--timeframe", default="1h", choices=["1h", "4h", "1d"])
  parser.add_argument("--text-hash-dim", type=int, default=64)
  parser.add_argument("--max-markets", type=int, default=None)
  parser.add_argument("--max-feather-files", type=int, default=None)
  parser.add_argument("--include-orderbook", action="store_true")
  parser.add_argument("--max-orderbook-rows", type=int, default=None)
  parser.add_argument("--add-correlation-edges", action="store_true")
  parser.add_argument("--correlation-threshold", type=float, default=0.85)
  parser.add_argument("--min-correlation-overlap", type=int, default=24)
  parser.add_argument("--max-correlation-tokens", type=int, default=1000)
  parser.add_argument("--max-correlation-edges-per-token", type=int, default=10)
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())

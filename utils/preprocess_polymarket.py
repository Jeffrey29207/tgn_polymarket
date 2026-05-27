import argparse
import ast
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EDGE_FEATURE_COLUMNS = [
  "edge_type_market_token",
  "edge_type_complement",
  "edge_type_cross_market",
  "edge_type_topic_market",
  "edge_type_market_dependency",
  "edge_type_token_dependency",
  "relation_implication",
  "relation_contradiction",
  "relation_equivalence",
  "relation_mutual_exclusion",
  "relation_other",
  "verified_label",
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
  "node_type_topic",
  "side_yes",
  "side_no",
  "is_binary_market",
  "outcome_index",
  "market_log_volume",
  "market_log_liquidity",
  "market_active",
  "market_closed",
  "market_lifetime_days",
  "num_outcomes",
]


def safe_literal_list(value):
  """Parse Polymarket fields that are stored as JSON/list-looking strings.

  Market metadata stores fields such as outcomes and clob_token_ids as strings like
  '["Yes", "No"]'. Depending on the source file, those strings may be valid JSON or Python-like
  literals. This helper accepts both formats and returns an empty list when the value is missing
  or malformed so one bad row does not stop preprocessing.
  """
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
  """Convert a pandas datetime-like Series into float Unix seconds.

  TGN expects timestamps as numeric values that can be sorted and subtracted. Polymarket OHLCV
  files store timestamps as pandas datetimes, so this normalizes them to UTC seconds.
  """
  dt = pd.to_datetime(series, utc=True, errors="coerce")
  return (dt.astype("int64") // 10 ** 9).astype("float64")


def scalar_unix_seconds(value):
  """Convert one datetime-like value into Unix seconds, or NaN if unavailable."""
  if value is None or pd.isna(value):
    return np.nan
  timestamp = pd.to_datetime(value, utc=True, errors="coerce")
  if pd.isna(timestamp):
    return np.nan
  return timestamp.timestamp()


def log1p_float(value):
  """Safely convert a non-negative market quantity into log1p scale.

  Volume, liquidity, and order book depths can have very large magnitudes. Log scaling keeps those
  features numerically tame while preserving the difference between small and large markets.
  """
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


def normalize_key(text):
  """Normalize market/outcome text for filename and verifier CSV matching."""
  return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def infer_topic_key(question):
  """Create a coarse event/topic id from a market question.

  This is intentionally lightweight: it groups obvious domains and otherwise falls back to a
  content-word signature. It is a retrieval prior, not a final semantic clustering model.
  """
  text = str(question).lower()
  keyword_topics = [
    ("topic:crypto", ["bitcoin", "ethereum", "solana", "crypto", "btc", "eth"]),
    ("topic:us-politics", ["trump", "biden", "harris", "election", "senate", "congress"]),
    ("topic:finance", ["s&p", "spx", "nasdaq", "dow", "fed", "rate", "inflation", "stocks"]),
    ("topic:sports-nba", ["nba", "lakers", "celtics", "knicks", "pistons"]),
    ("topic:sports-nfl", ["nfl", "super bowl", "chiefs", "eagles", "cowboys"]),
    ("topic:sports-nhl", ["nhl", "capitals", "canadiens", "lightning"]),
    ("topic:sports-soccer", ["premier league", "champions league", "fc ", "world cup"]),
  ]
  for topic, keywords in keyword_topics:
    if any(keyword in text for keyword in keywords):
      return topic

  stopwords = {
    "will", "the", "a", "an", "on", "in", "of", "to", "by", "before", "after", "or", "and",
    "who", "what", "which", "be", "is", "are", "for", "with",
  }
  words = [normalize_key(word) for word in str(question).replace("?", " ").split()]
  words = [word for word in words if len(word) > 2 and word not in stopwords]
  signature = "-".join(words[:3]) if words else hashlib.md5(str(question).encode("utf-8")).hexdigest()[:8]
  return f"topic:{signature}"


class NodeIndexer:
  """Maps Polymarket string identifiers to contiguous TGN node ids.

  TGN reserves node 0 for padding, so real market/token nodes start at 1.
  """
  def __init__(self):
    self.raw_to_idx = {}
    self.idx_to_raw = {0: "__padding__"}

  def get(self, raw_id):
    """Return the integer TGN node id for a raw Polymarket id, creating it if needed."""
    raw_id = str(raw_id)
    if raw_id not in self.raw_to_idx:
      idx = len(self.raw_to_idx) + 1
      self.raw_to_idx[raw_id] = idx
      self.idx_to_raw[idx] = raw_id
    return self.raw_to_idx[raw_id]


class EdgeFeatureWriter:
  """Append edge feature rows to a temporary binary file instead of keeping them in RAM.

  The full hourly Polymarket graph can contain enough edges that a Python list of feature vectors
  becomes much larger than the final float32 array. This writer stores only compact float32 rows on
  disk while preprocessing runs. At the end, run() reorders those rows into the final .npy file
  after the edge CSV has been sorted by timestamp.
  """
  def __init__(self, path, feature_dimension):
    self.path = Path(path)
    self.feature_dimension = feature_dimension
    self.count = 0
    self._fh = self.path.open("wb")

  def append_feature(self, feature_vector):
    """Write one real edge feature vector and return its 1-based TGN edge id."""
    feature_array = np.asarray(feature_vector, dtype=np.float32)
    if feature_array.shape != (self.feature_dimension,):
      raise ValueError(
        f"Expected edge feature shape {(self.feature_dimension,)}, got {feature_array.shape}"
      )
    feature_array.tofile(self._fh)
    self.count += 1
    return self.count

  def append_features(self, feature_matrix):
    """Write many edge feature rows and return the first 1-based edge id."""
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    if feature_array.ndim != 2 or feature_array.shape[1] != self.feature_dimension:
      raise ValueError(
        f"Expected edge feature matrix with {self.feature_dimension} columns, "
        f"got {feature_array.shape}"
      )
    first_idx = self.count + 1
    feature_array.tofile(self._fh)
    self.count += feature_array.shape[0]
    return first_idx

  def close(self):
    if not self._fh.closed:
      self._fh.close()

  def memmap(self):
    """Open the written feature rows as a read-only memmap.

    The temporary file contains only real edge rows. The padding row 0 is added later when writing
    the final ml_<name>.npy file.
    """
    self.close()
    return np.memmap(self.path, dtype=np.float32, mode="r",
                     shape=(self.count, self.feature_dimension))

  def cleanup(self):
    self.close()
    if self.path.exists():
      self.path.unlink()


def load_markets(data_root, max_markets=None):
  """Load Polymarket market metadata from parquet shards, falling back to JSONL if needed.

  The parquet directory is preferred because it is already columnar and much faster to scan. The
  JSONL fallback supports the one-year market dump. Returned columns are normalized so later code
  can use snake_case names no matter which source was used.
  """
  markets_dir = data_root / "markets"
  parquet_paths = sorted(p for p in markets_dir.glob("*.parquet") if not p.name.startswith("._"))

  frames = []
  if parquet_paths:
    market_columns = [
      "condition_id", "question", "outcomes", "clob_token_ids", "volume", "liquidity",
      "active", "closed", "created_at", "end_date",
    ]
    for path in parquet_paths:
      frames.append(pd.read_parquet(path, columns=market_columns))
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
  """Normalize market metadata and expose condition/token lookup tables.

  Returns:
    lookup: condition_id -> compact market dictionary used while creating nodes and edges.
    token_to_condition: clob token id -> parent condition id.

  The lookup stores both raw identifiers and numeric/static features such as log volume,
  liquidity, market lifetime, active/closed flags, and YES/NO token ids.
  """
  lookup = {}
  token_to_condition = {}

  for row in markets.to_dict("records"):
    condition_id = str(row.get("condition_id"))
    outcomes = safe_literal_list(row.get("outcomes"))
    token_ids = safe_literal_list(row.get("clob_token_ids"))
    side_to_token = {}
    outcome_to_token = {}
    token_to_outcome = {}

    for outcome_idx, (outcome, token_id) in enumerate(zip(outcomes, token_ids)):
      outcome_key = normalize_key(outcome)
      token_id = str(token_id)
      side_to_token[str(outcome).strip().lower()] = token_id
      outcome_to_token[outcome_key] = token_id
      token_to_outcome[token_id] = {
        "outcome": str(outcome),
        "outcome_key": outcome_key,
        "outcome_index": float(outcome_idx),
      }
      token_to_condition[token_id] = condition_id

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
      "outcome_to_token": outcome_to_token,
      "token_to_outcome": token_to_outcome,
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
      "is_binary_market": float(len(outcomes) == 2),
      "topic_key": infer_topic_key(str(row.get("question", ""))),
    }

  return lookup, token_to_condition


def infer_outcome_key_from_filename(filename, market):
  """Infer the outcome represented by an OHLCV filename using real market outcome labels.

  OHLCV filenames are sanitized pair names, usually shaped like:
      <market-question><outcome><date>_USDC-4h.feather

  The condition id tells us which market the file belongs to, but the filename tells us which
  outcome token within that market is being priced. We therefore first strip mechanical suffixes
  such as timeframe, quote currency, and date, then look for an outcome label at the end of the
  remaining name. This avoids substring mistakes such as reading "November" or "Latino" as "NO".
  If a filename does not follow the expected suffix shape, the fallback chooses the outcome label
  whose last occurrence appears furthest to the right in the filename.
  """
  stem = Path(filename).stem
  stem = re.sub(r"-\d+[a-zA-Z]+$", "", stem)
  stem = re.sub(r"_?USDC$", "", stem, flags=re.IGNORECASE)
  stem = re.sub(r"[vV]\d+$", "", stem)
  stem = re.sub(r"\d{8}$", "", stem)
  normalized_stem = normalize_key(stem)
  outcomes = sorted(market.get("outcome_to_token", {}).keys(), key=len, reverse=True)

  for outcome_key in outcomes:
    if outcome_key and normalized_stem.endswith(outcome_key):
      return outcome_key

  normalized_filename = normalize_key(filename)
  rightmost_match = None
  for outcome_key in outcomes:
    position = normalized_filename.rfind(outcome_key)
    if outcome_key and position >= 0:
      if rightmost_match is None or position > rightmost_match[0]:
        rightmost_match = (position, outcome_key)
  if rightmost_match is not None:
    return rightmost_match[1]

  return "unknown"


def load_pair_mapping(data_root, timeframe):
  """Load the mapping from OHLCV Feather filenames back to Polymarket condition ids.

  freqtrade filenames are sanitized trading-pair names, not the original condition ids. This CSV
  restores that link so each time series can be attached to the correct market metadata row.
  """
  mapping_path = data_root / "data" / "freqtrade_pair_mapping.csv"
  mapping = pd.read_csv(mapping_path)
  mapping = mapping[mapping["Timeframe"].astype(str) == timeframe]
  mapping["Original_Condition_ID"] = mapping["Original_Condition_ID"].astype(str)
  return {
    row["New_Filename"]: row["Original_Condition_ID"]
    for _, row in mapping.iterrows()
  }


def orderbook_features(snapshot, depth_levels=5):
  """Summarize one raw order book snapshot into fixed numeric features.

  The raw snapshot contains full bid/ask ladders. TGN needs a fixed-length edge feature vector, so
  this extracts best bid, best ask, spread, mid price, top-N bid/ask depth, and a depth imbalance
  score in [-1, 1].
  """
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

  The returned dictionary is keyed by (token_id, hour_timestamp). During OHLCV edge creation, that
  key is used to attach microstructure features to the matching market-token event.
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
                       complement_sum_minus_one=0.0, cross_return_correlation=0.0,
                       relation_type=None, verified_label=0.0):
  """Create one edge-feature vector in the exact order expected by EDGE_FEATURE_COLUMNS.

  The edge type is encoded as a one-hot feature because the current TGN implementation is
  homogeneous. This lets one model consume market-token observation edges, within-market
  complement edges, and optional cross-market candidate edges without changing model/tgn.py.
  Missing feature groups are filled with zeroes.
  """
  features = dict.fromkeys(EDGE_FEATURE_COLUMNS, 0.0)
  features[f"edge_type_{edge_type}"] = 1.0
  relation_key = normalize_key(relation_type or "")
  if relation_key in ("implies", "implication", "imply"):
    features["relation_implication"] = 1.0
  elif relation_key in ("contradicts", "contradiction", "contradict"):
    features["relation_contradiction"] = 1.0
  elif relation_key in ("equivalent", "equivalence", "same"):
    features["relation_equivalence"] = 1.0
  elif relation_key in ("mutualexclusion", "mutuallyexclusive", "exclusive"):
    features["relation_mutual_exclusion"] = 1.0
  elif relation_key:
    features["relation_other"] = 1.0
  features["verified_label"] = float(verified_label)

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
  """Append one temporal interaction and its aligned feature vector.

  edges becomes the ml_*.csv table, while edge_features becomes ml_*.npy. The idx field points
  from each CSV row to the corresponding row in the edge feature matrix.
  """
  # Edge index 0 is reserved for padding. For full Polymarket runs, edge_features is an
  # EdgeFeatureWriter; for tiny tests it can still be a regular list.
  if hasattr(edge_features, "append_feature"):
    edge_idx = edge_features.append_feature(feature_vector)
  else:
    edge_idx = len(edge_features)
    edge_features.append(feature_vector)

  edges.append({
    "u": int(src),
    "i": int(dst),
    "ts": float(ts),
    "label": float(label),
    "idx": edge_idx,
  })


def add_market_token_edges(edge_tables, edge_features, market_node, token_node, frame, market,
                           orderbooks, token_id):
  """Append all OHLCV observations for one token as a vectorized edge batch."""
  n_rows = len(frame)
  if n_rows == 0:
    return

  feature_matrix = np.zeros((n_rows, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)
  column_to_idx = {column: idx for idx, column in enumerate(EDGE_FEATURE_COLUMNS)}
  feature_matrix[:, column_to_idx["edge_type_market_token"]] = 1.0

  for column in ["open", "high", "low", "close", "volume", "log_return", "abs_return",
                 "high_low_spread"]:
    values = frame[column].to_numpy(dtype=np.float32)
    feature_matrix[:, column_to_idx[column]] = np.nan_to_num(values, nan=0.0, posinf=0.0,
                                                             neginf=0.0)

  feature_matrix[:, column_to_idx["market_log_volume"]] = float(market.get("log_volume", 0.0))
  feature_matrix[:, column_to_idx["market_log_liquidity"]] = float(
    market.get("log_liquidity", 0.0))
  if not np.isnan(market.get("end_date", np.nan)):
    time_to_end = (market["end_date"] - frame["ts"].to_numpy(dtype=np.float64)) / 86400.0
    feature_matrix[:, column_to_idx["time_to_end_days"]] = np.maximum(time_to_end, 0.0)

  if orderbooks:
    for row_idx, ts in enumerate(frame["ts"].to_numpy(dtype=np.float64)):
      hour = ts // 3600 * 3600
      orderbook = orderbooks.get((token_id, float(hour)))
      if orderbook is None:
        continue
      for column in ["best_bid", "best_ask", "book_spread", "book_mid", "bid_depth_top5",
                     "ask_depth_top5", "book_imbalance_top5"]:
        feature_matrix[row_idx, column_to_idx[column]] = float(orderbook.get(column, 0.0))

  first_idx = edge_features.append_features(feature_matrix)
  edge_tables.append(pd.DataFrame({
    "u": np.full(n_rows, int(market_node), dtype=np.int32),
    "i": np.full(n_rows, int(token_node), dtype=np.int32),
    "ts": frame["ts"].to_numpy(dtype=np.float64),
    "label": np.zeros(n_rows, dtype=np.float32),
    "idx": np.arange(first_idx, first_idx + n_rows, dtype=np.int64),
  }))


def add_topic_market_edge(edges, edge_features, topic_node, market_node, ts, market):
  """Add a topic/event-to-market edge used as a semantic retrieval prior."""
  feature_vector = make_edge_features("topic_market", market=market)
  add_edge(edges, edge_features, topic_node, market_node, ts, 0.0, feature_vector)


def build_node_features(node_indexer, node_info, text_hash_dim):
  """Build the ml_*_node.npy matrix and return its column names.

  Node ids are already assigned by NodeIndexer. This function creates one static feature row for
  every market and token node, with row 0 left as all-zero padding. It combines fixed structural
  features with lightweight hashed question text features.
  """
  feature_columns = NODE_BASE_COLUMNS + [f"question_hash_{i}" for i in range(text_hash_dim)]
  node_features = np.zeros((len(node_indexer.raw_to_idx) + 1, len(feature_columns)), dtype=np.float32)

  for raw_id, idx in node_indexer.raw_to_idx.items():
    info = node_info.get(raw_id, {})
    base = [
      float(info.get("node_type_market", 0.0)),
      float(info.get("node_type_token", 0.0)),
      float(info.get("node_type_topic", 0.0)),
      float(info.get("side_yes", 0.0)),
      float(info.get("side_no", 0.0)),
      float(info.get("is_binary_market", 0.0)),
      float(info.get("outcome_index", 0.0)),
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


def resolve_token_from_verifier_row(row, prefix, market_lookup):
  """Resolve a verifier CSV row into a token id if token/outcome fields are present."""
  token_field = f"{prefix}_token_id"
  if token_field in row and not pd.isna(row[token_field]):
    return str(row[token_field])

  condition_field = f"{prefix}_condition_id"
  outcome_field = f"{prefix}_outcome"
  if condition_field not in row or outcome_field not in row:
    return None
  condition_id = str(row[condition_field])
  market = market_lookup.get(condition_id)
  if market is None or pd.isna(row[outcome_field]):
    return None
  outcome_key = normalize_key(row[outcome_field])
  return market.get("outcome_to_token", {}).get(outcome_key)


def add_verified_dependency_edges(edges, edge_features, verified_pairs_csv, market_lookup,
                                  market_nodes, token_nodes, default_ts):
  """Add symbolic-verifier labels as market-market and token-token temporal edges.

  Expected CSV columns can be either:
    source_condition_id,target_condition_id,label,relation_type,timestamp
  or token-level:
    source_token_id,target_token_id,label,relation_type,timestamp
  Token rows may also use source_condition_id/source_outcome and target_condition_id/target_outcome.
  """
  if verified_pairs_csv is None:
    return {"market_dependency_edges": 0, "token_dependency_edges": 0}

  frame = pd.read_csv(verified_pairs_csv)
  counts = {"market_dependency_edges": 0, "token_dependency_edges": 0}
  for row in frame.to_dict("records"):
    label = float(row.get("label", row.get("verified_label", 1.0)) or 0.0)
    relation_type = row.get("relation_type", row.get("relation", "other"))
    ts = scalar_unix_seconds(row.get("timestamp")) if "timestamp" in row else np.nan
    if np.isnan(ts):
      ts = default_ts

    source_token = resolve_token_from_verifier_row(row, "source", market_lookup)
    target_token = resolve_token_from_verifier_row(row, "target", market_lookup)
    if source_token in token_nodes and target_token in token_nodes:
      feature_vector = make_edge_features("token_dependency", relation_type=relation_type,
                                          verified_label=label)
      add_edge(edges, edge_features, token_nodes[source_token], token_nodes[target_token], ts, label,
               feature_vector)
      counts["token_dependency_edges"] += 1
      continue

    source_condition = str(row.get("source_condition_id", ""))
    target_condition = str(row.get("target_condition_id", ""))
    if source_condition in market_nodes and target_condition in market_nodes:
      feature_vector = make_edge_features("market_dependency", relation_type=relation_type,
                                          verified_label=label)
      add_edge(edges, edge_features, market_nodes[source_condition], market_nodes[target_condition],
               ts, label, feature_vector)
      counts["market_dependency_edges"] += 1

  return counts


def write_sorted_edge_features(edge_features, old_edge_idxs, edge_feature_path, chunk_size=100000):
  """Write the final edge-feature .npy in timestamp-sorted CSV order.

  graph_df is sorted by timestamp near the end of preprocessing, which changes edge ids. The
  temporary feature file is still in creation order, so this function copies rows into a final
  numpy file in chunks. Row 0 is all zeros for TGN padding; sorted real edges start at row 1.
  """
  feature_dimension = len(EDGE_FEATURE_COLUMNS)
  num_edges = len(old_edge_idxs)

  if hasattr(edge_features, "memmap"):
    raw_features = edge_features.memmap()
    output = np.lib.format.open_memmap(edge_feature_path, mode="w+", dtype=np.float32,
                                       shape=(num_edges + 1, feature_dimension))
    output[0] = 0.0
    for start in range(0, num_edges, chunk_size):
      end = min(num_edges, start + chunk_size)
      output[start + 1:end + 1] = np.nan_to_num(raw_features[old_edge_idxs[start:end] - 1],
                                                nan=0.0, posinf=0.0, neginf=0.0)
    output.flush()
    del output
    del raw_features
    edge_features.cleanup()
    return

  ordered_features = [edge_features[0]] + [edge_features[int(old_idx)] for old_idx in old_edge_idxs]
  np.save(edge_feature_path, np.nan_to_num(np.asarray(ordered_features, dtype=np.float32),
                                           nan=0.0, posinf=0.0, neginf=0.0))


def run(args):
  """Main preprocessing entry point.

  This orchestrates the complete raw-data-to-TGN conversion:
    1. load market metadata and filename mappings;
    2. assign integer ids to market/token nodes;
    3. turn OHLCV rows into temporal market-token edges;
    4. add YES/NO complement edges for logical within-market structure;
    5. optionally add high-correlation cross-market edges;
    6. write ml_<name>.csv, ml_<name>.npy, ml_<name>_node.npy, and metadata JSON.
  """
  data_root = Path(args.data_root)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if args.verified_pairs_csv is not None:
    verified_pairs_path = Path(args.verified_pairs_csv)
    if not verified_pairs_path.exists():
      raise FileNotFoundError(
        f"--verified-pairs-csv was set to '{args.verified_pairs_csv}', but that file does not "
        "exist. Pass the real CSV path produced by your symbolic verifier, or omit "
        "--verified-pairs-csv to build the rich graph without verifier-labeled edges."
      )
    args.verified_pairs_csv = verified_pairs_path

  markets = load_markets(data_root, max_markets=args.max_markets)
  market_lookup, _ = build_market_lookup(markets)
  filename_to_condition = load_pair_mapping(data_root, args.timeframe)

  node_indexer = NodeIndexer()
  node_info = {}
  token_nodes = {}
  token_markets = {}
  market_nodes = {}
  topic_nodes = {}
  market_first_ts = {}
  edge_tables = []
  extra_edges = []
  temp_feature_path = output_dir / f".ml_{args.output_name}_edge_features.tmp"
  if temp_feature_path.exists():
    temp_feature_path.unlink()
  edge_features = EdgeFeatureWriter(temp_feature_path, len(EDGE_FEATURE_COLUMNS))
  token_series = {}
  close_tables = []

  feather_dir = data_root / "data" / "data"
  feather_paths = sorted(feather_dir.glob(f"*-{args.timeframe}.feather"))
  if args.max_feather_files is not None:
    feather_paths = feather_paths[:args.max_feather_files]

  start_ts = scalar_unix_seconds(args.start_date) if args.start_date else None
  end_ts = scalar_unix_seconds(args.end_date) if args.end_date else None

  needed_token_ids = set()
  processed_files = 0
  skipped_files = 0
  for file_idx, path in enumerate(feather_paths, start=1):
    condition_id = filename_to_condition.get(path.name)
    market = market_lookup.get(str(condition_id))
    if market is None:
      skipped_files += 1
      continue
    outcome_key = infer_outcome_key_from_filename(path.name, market)
    if outcome_key not in market["outcome_to_token"]:
      skipped_files += 1
      continue
    if market["volume"] < args.min_market_volume:
      skipped_files += 1
      continue
    needed_token_ids.add(market["outcome_to_token"][outcome_key])

  orderbooks = {}
  if args.include_orderbook:
    orderbooks = load_orderbook_hourly(data_root, needed_token_ids, max_rows=args.max_orderbook_rows)

  for file_idx, path in enumerate(feather_paths, start=1):
    condition_id = filename_to_condition.get(path.name)
    market = market_lookup.get(str(condition_id))
    if market is None:
      skipped_files += 1
      continue
    outcome_key = infer_outcome_key_from_filename(path.name, market)
    if outcome_key not in market["outcome_to_token"]:
      skipped_files += 1
      continue
    if market["volume"] < args.min_market_volume:
      skipped_files += 1
      continue

    token_id = market["outcome_to_token"][outcome_key]
    outcome_info = market["token_to_outcome"].get(token_id, {})
    side = outcome_key
    market_raw = f"market:{condition_id}"
    token_raw = f"token:{token_id}"
    market_node = node_indexer.get(market_raw)
    token_node = node_indexer.get(token_raw)
    market_nodes[condition_id] = market_node
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
      "is_binary_market": market["is_binary_market"],
      "question": market["question"],
    }
    node_info[token_raw] = {
      "node_type_token": 1.0,
      "side_yes": 1.0 if outcome_key == "yes" else 0.0,
      "side_no": 1.0 if outcome_key == "no" else 0.0,
      "is_binary_market": market["is_binary_market"],
      "outcome_index": outcome_info.get("outcome_index", 0.0),
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
    if start_ts is not None:
      frame = frame[frame["ts"] >= start_ts]
    if end_ts is not None:
      frame = frame[frame["ts"] <= end_ts]
    if len(frame) == 0:
      skipped_files += 1
      continue
    market_first_ts[condition_id] = min(market_first_ts.get(condition_id, float("inf")),
                                       float(frame["ts"].min()))

    frame["log_return"] = np.log(frame["close"].replace(0, np.nan)).diff().replace(
      [np.inf, -np.inf], np.nan).fillna(0.0)
    frame["abs_return"] = frame["log_return"].abs()
    frame["high_low_spread"] = (frame["high"] - frame["low"]).fillna(0.0)
    frame["volume"] = np.log1p(frame["volume"].clip(lower=0))

    series = frame[["ts", "close", "log_return"]].dropna().set_index("ts")
    token_series[token_id] = series

    add_market_token_edges(edge_tables, edge_features, market_node, token_node, frame, market,
                           orderbooks, token_id)
    close_tables.append(pd.DataFrame({
      "condition_id": condition_id,
      "ts": frame["ts"].to_numpy(dtype=np.float64),
      "side": outcome_key,
      "close": frame["close"].to_numpy(dtype=np.float32),
    }))

    processed_files += 1
    if args.max_edge_rows is not None and edge_features.count >= args.max_edge_rows:
      print(f"Reached --max-edge-rows={args.max_edge_rows}; stopping early.")
      break
    if args.progress_every > 0 and file_idx % args.progress_every == 0:
      print(f"Processed {file_idx}/{len(feather_paths)} files; "
            f"matched={processed_files}, skipped={skipped_files}, edges={edge_features.count}")

  if args.include_topic_edges:
    for condition_id, market_node in market_nodes.items():
      market = market_lookup[condition_id]
      topic_raw = market["topic_key"]
      topic_node = node_indexer.get(topic_raw)
      topic_nodes[topic_raw] = topic_node
      node_info[topic_raw] = {
        "node_type_topic": 1.0,
        "question": topic_raw,
      }
      add_topic_market_edge(extra_edges, edge_features, topic_node, market_node,
                            market_first_ts.get(condition_id, 0.0), market)

  if args.include_complement_edges and close_tables:
    close_df = pd.concat(close_tables, ignore_index=True)
    close_df = close_df.pivot_table(index=["condition_id", "ts"], columns="side", values="close",
                                    aggfunc="last").reset_index()
  else:
    close_df = pd.DataFrame()

  for row in close_df.itertuples(index=False):
    # YES/NO complement edges encode the within-market logical relationship. The feature
    # yes_close + no_close - 1 is a direct signal of temporary complement mispricing.
    condition_id = str(row.condition_id)
    if not hasattr(row, "yes") or not hasattr(row, "no"):
      continue
    if pd.isna(row.yes) or pd.isna(row.no):
      continue
    market = market_lookup.get(condition_id)
    if market is None:
      continue
    yes_token = market["side_to_token"].get("yes")
    no_token = market["side_to_token"].get("no")
    if yes_token not in token_nodes or no_token not in token_nodes:
      continue
    complement_sum_minus_one = float(row.yes) + float(row.no) - 1.0
    feature_vector = make_edge_features("complement", market=market,
                                        complement_sum_minus_one=complement_sum_minus_one)
    add_edge(extra_edges, edge_features, token_nodes[yes_token], token_nodes[no_token], row.ts, 1.0,
             feature_vector)

  if args.add_correlation_edges:
    add_correlation_edges(extra_edges, edge_features, token_series, token_nodes, token_markets,
                          args.correlation_threshold, args.min_correlation_overlap,
                          args.max_correlation_tokens, args.max_correlation_edges_per_token)

  verifier_counts = add_verified_dependency_edges(extra_edges, edge_features,
                                                  args.verified_pairs_csv,
                                                  market_lookup, market_nodes, token_nodes,
                                                  default_ts=max(market_first_ts.values())
                                                  if market_first_ts else 0.0)

  if len(edge_tables) == 0 and len(extra_edges) == 0:
    raise ValueError(
      "No Polymarket edges were produced. Check that freqtrade_pair_mapping.csv, "
      "market metadata, and the selected timeframe refer to overlapping condition IDs."
    )

  graph_parts = edge_tables
  if extra_edges:
    graph_parts.append(pd.DataFrame(extra_edges))
  graph_df = pd.concat(graph_parts, ignore_index=True).sort_values(["ts", "idx"]).reset_index(drop=True)
  # After sorting by time, reassign contiguous edge ids and reorder the feature matrix to match.
  # model/tgn.py indexes edge_raw_features by this idx column during every forward pass.
  old_edge_idxs = graph_df["idx"].astype(int).values
  graph_df["idx"] = np.arange(1, len(graph_df) + 1)

  node_features, node_feature_columns = build_node_features(node_indexer, node_info,
                                                            args.text_hash_dim)

  graph_path = output_dir / f"ml_{args.output_name}.csv"
  edge_feature_path = output_dir / f"ml_{args.output_name}.npy"
  node_feature_path = output_dir / f"ml_{args.output_name}_node.npy"
  metadata_path = output_dir / f"ml_{args.output_name}_metadata.json"

  graph_df.to_csv(graph_path, index=False)
  write_sorted_edge_features(edge_features, old_edge_idxs, edge_feature_path,
                             chunk_size=args.feature_write_chunk_size)
  np.save(node_feature_path, node_features)

  metadata = {
    "node_map": node_indexer.raw_to_idx,
    "idx_to_raw_node": node_indexer.idx_to_raw,
    "edge_feature_columns": EDGE_FEATURE_COLUMNS,
    "node_feature_columns": node_feature_columns,
    "timeframe": args.timeframe,
    "num_edges": int(len(graph_df)),
    "num_nodes": int(node_features.shape[0] - 1),
    "num_market_nodes": len(market_nodes),
    "num_token_nodes": len(token_nodes),
    "num_topic_nodes": len(topic_nodes),
    "verifier_edges": verifier_counts,
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

  print(f"Wrote {graph_path}")
  print(f"Wrote {edge_feature_path}")
  print(f"Wrote {node_feature_path}")
  print(f"Wrote {metadata_path}")
  print(f"Nodes: {metadata['num_nodes']}, edges: {metadata['num_edges']}")


def parse_args():
  """Define command-line options for building a Polymarket TGN dataset."""
  parser = argparse.ArgumentParser("Preprocess Polymarket data into the TGN ml_* format")
  parser.add_argument("--data-root", default="C:/Users/User/Downloads/poly_data")
  parser.add_argument("--output-dir", default="./data")
  parser.add_argument("--output-name", default="polymarket")
  parser.add_argument("--timeframe", default="1h", choices=["1h", "4h", "1d"])
  parser.add_argument("--text-hash-dim", type=int, default=64)
  parser.add_argument("--max-markets", type=int, default=None)
  parser.add_argument("--max-feather-files", type=int, default=None)
  parser.add_argument("--start-date", default=None,
                      help="Optional UTC date/datetime lower bound for OHLCV rows, e.g. 2025-01-01.")
  parser.add_argument("--end-date", default=None,
                      help="Optional UTC date/datetime upper bound for OHLCV rows, e.g. 2025-03-01.")
  parser.add_argument("--min-market-volume", type=float, default=0.0,
                      help="Skip markets whose metadata volume is below this threshold.")
  parser.add_argument("--max-edge-rows", type=int, default=None,
                      help="Stop after writing roughly this many market-token edges.")
  parser.add_argument("--progress-every", type=int, default=250,
                      help="Print preprocessing progress every N Feather files. Use 0 to disable.")
  parser.add_argument("--include-complement-edges", action="store_true",
                      help="Add YES/NO complement edges. Useful, but can add millions of rows.")
  parser.add_argument("--include-topic-edges", action="store_true",
                      help="Add coarse event/topic nodes with topic -> market edges.")
  parser.add_argument("--verified-pairs-csv", default=None,
                      help="Optional symbolic-verifier labels CSV. Supports market-market rows "
                           "with source_condition_id,target_condition_id,label,relation_type, "
                           "or token-token rows with source_token_id,target_token_id.")
  parser.add_argument("--include-orderbook", action="store_true")
  parser.add_argument("--max-orderbook-rows", type=int, default=None)
  parser.add_argument("--add-correlation-edges", action="store_true")
  parser.add_argument("--correlation-threshold", type=float, default=0.85)
  parser.add_argument("--min-correlation-overlap", type=int, default=24)
  parser.add_argument("--max-correlation-tokens", type=int, default=1000)
  parser.add_argument("--max-correlation-edges-per-token", type=int, default=10)
  parser.add_argument("--feature-write-chunk-size", type=int, default=100000,
                      help="Number of edge feature rows to copy at a time when building the final "
                           ".npy file. Lower this if memory is still tight.")
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args())

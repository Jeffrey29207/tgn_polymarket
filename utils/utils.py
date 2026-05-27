import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


class TemporalTypedNegativeEdgeSampler(object):
  """
  Negative sampler for temporal heterogeneous graphs.

  The original sampler draws destinations uniformly from all destination nodes. That is fine for
  simple bipartite benchmarks, but it can be too easy for Polymarket graphs: replacing a token
  node with a market node, for example, creates an obviously invalid negative. This sampler can
  keep negatives type-compatible and avoid edges that were already observed before the query time.
  """
  def __init__(self, src_list, dst_list, timestamps=None, node_types=None, seed=None,
               avoid_historical_edges=True):
    """Create lookup tables used for fast negative sampling.

    src_list, dst_list, and timestamps describe the observed positive temporal edges. node_types
    is an optional array indexed by node id; when present, each negative destination is sampled
    from the same node type as the positive destination.
    """
    self.seed = seed
    self.random_state = np.random.RandomState(seed) if seed is not None else np.random
    self.src_list = np.asarray(src_list)
    self.dst_list = np.asarray(dst_list)
    self.timestamps = np.asarray(timestamps) if timestamps is not None else None
    self.node_types = np.asarray(node_types) if node_types is not None else None
    self.avoid_historical_edges = avoid_historical_edges

    self.unique_src = np.unique(self.src_list)
    self.unique_dst = np.unique(self.dst_list)
    self.type_to_dst = {}

    # Group destination candidates by node type. For Polymarket this prevents easy negatives like
    # replacing an outcome token destination with a market node destination.
    if self.node_types is not None:
      for dst in self.unique_dst:
        node_type = self.node_types[dst]
        self.type_to_dst.setdefault(node_type, []).append(dst)
      self.type_to_dst = {
        node_type: np.asarray(nodes, dtype=self.unique_dst.dtype)
        for node_type, nodes in self.type_to_dst.items()
      }

    self.edge_timestamps = {}
    if timestamps is not None:
      # Store historical edge times so batch sampling can reject negatives that were already
      # positive before the current query time.
      for src, dst, ts in zip(self.src_list, self.dst_list, self.timestamps):
        self.edge_timestamps.setdefault((src, dst), []).append(ts)
      for edge, edge_times in self.edge_timestamps.items():
        self.edge_timestamps[edge] = np.sort(np.asarray(edge_times))

  def _candidate_destinations(self, positive_dst):
    """Return the destination pool that matches the positive destination's node type."""
    if self.node_types is None:
      return self.unique_dst
    node_type = self.node_types[positive_dst]
    return self.type_to_dst.get(node_type, self.unique_dst)

  def _edge_seen_before(self, src, dst, timestamp):
    """Check whether (src, dst) was already a positive edge before timestamp."""
    if not self.avoid_historical_edges or self.timestamps is None:
      return False
    edge_times = self.edge_timestamps.get((src, dst))
    if edge_times is None:
      return False
    return np.searchsorted(edge_times, timestamp, side="left") > 0

  def _sample_one_destination(self, src, positive_dst=None, timestamp=None, max_attempts=100):
    """Sample one plausible negative destination for a positive edge.

    The sampler first tries rejection sampling: same destination type, not the real destination,
    and not an already-seen positive historical edge. If the graph is too dense for that to work
    quickly, it falls back to a type-compatible destination so training can continue.
    """
    candidates = self._candidate_destinations(positive_dst) if positive_dst is not None else self.unique_dst
    if len(candidates) == 0:
      candidates = self.unique_dst

    for _ in range(max_attempts):
      sampled_dst = candidates[self.random_state.randint(0, len(candidates))]
      if positive_dst is not None and sampled_dst == positive_dst:
        continue
      if src is not None and timestamp is not None and self._edge_seen_before(src, sampled_dst, timestamp):
        continue
      return sampled_dst

    # Dense subgraphs can make rejection sampling difficult; return any type-compatible non-self
    # destination as a pragmatic fallback.
    fallback = candidates[self.random_state.randint(0, len(candidates))]
    if positive_dst is not None and len(candidates) > 1:
      while fallback == positive_dst:
        fallback = candidates[self.random_state.randint(0, len(candidates))]
    return fallback

  def sample(self, size, sources=None, destinations=None, timestamps=None):
    """Sample a batch of negative destinations.

    When only size is supplied, this behaves like the original RandEdgeSampler and ignores
    temporal/type context. When sources, destinations, and timestamps are supplied, it returns
    context-aware negatives aligned with the positive batch.
    """
    # Keep the original RandEdgeSampler-compatible mode for callers that only pass a size.
    if sources is None or destinations is None or timestamps is None:
      src_index = self.random_state.randint(0, len(self.unique_src), size)
      sampled_src = self.unique_src[src_index]
      sampled_dst = np.asarray([
        self._sample_one_destination(None)
        for _ in range(size)
      ])
      return sampled_src, sampled_dst

    sampled_dst = np.asarray([
      self._sample_one_destination(src, dst, ts)
      for src, dst, ts in zip(sources, destinations, timestamps)
    ])
    return np.asarray(sources), sampled_dst

  def sample_for_batch(self, sources, destinations, timestamps):
    """Convenience wrapper used by train/evaluation loops for positive-edge batches."""
    # Training/evaluation code calls this path when it wants type-compatible, time-aware negatives.
    return self.sample(len(sources), sources=sources, destinations=destinations,
                       timestamps=timestamps)

  def reset_random_state(self):
    """Reset deterministic sampling for validation/test reproducibility."""
    if self.seed is not None:
      self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

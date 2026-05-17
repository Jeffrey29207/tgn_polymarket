import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      # Polymarket can use a sampler that needs the current positive edge and timestamp in order
      # to draw type-compatible negatives. Existing datasets still use the old size-only sampler.
      if hasattr(negative_edge_sampler, "sample_for_batch"):
        _, negative_samples = negative_edge_sampler.sample_for_batch(sources_batch, destinations_batch,
                                                                     timestamps_batch)
      else:
        _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc


def score_temporal_edges(model, sources, destinations, timestamps, edge_idxs, n_neighbors,
                         batch_size=200, mutate_memory=False):
  """
  Score arbitrary temporal edges without requiring negative samples.

  This is useful for Polymarket candidate retrieval, where a candidate edge means "these two
  markets/outcomes may be dependent" rather than "this observed edge should update node memory".
  By default the function restores model memory after every batch so scoring hypothetical
  candidates does not contaminate subsequent scores.
  """
  scores = np.zeros(len(sources), dtype=np.float32)
  num_instance = len(sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    model.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = sources[s_idx:e_idx]
      destinations_batch = destinations[s_idx:e_idx]
      timestamps_batch = timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx:e_idx]

      memory_backup = None
      if getattr(model, "use_memory", False) and not mutate_memory:
        memory_backup = model.memory.backup_memory()

      # Reuse the positive-edge branch. The negative destination is irrelevant here.
      # Memory is restored afterward by default because candidate edges are hypothetical retrieval
      # queries, not observed events that should update the temporal graph state.
      pos_prob, _ = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                     destinations_batch, timestamps_batch,
                                                     edge_idxs_batch, n_neighbors)
      scores[s_idx:e_idx] = pos_prob.cpu().numpy()

      if memory_backup is not None:
        model.memory.restore_memory(memory_backup)

  return scores


def ranking_metrics(labels, scores, k_values=(10, 50, 100)):
  """Ranking metrics for dependency candidate retrieval."""
  labels = np.asarray(labels).astype(np.float32)
  scores = np.asarray(scores).astype(np.float32)
  order = np.argsort(-scores)
  sorted_labels = labels[order]

  metrics = {}
  if len(np.unique(labels)) > 1:
    metrics["ap"] = average_precision_score(labels, scores)
    metrics["auc"] = roc_auc_score(labels, scores)
  else:
    metrics["ap"] = float("nan")
    metrics["auc"] = float("nan")

  total_positive = max(float(labels.sum()), 1.0)
  positive_ranks = np.where(sorted_labels > 0)[0]
  metrics["mrr"] = 1.0 / float(positive_ranks[0] + 1) if len(positive_ranks) > 0 else 0.0

  for k in k_values:
    k = min(int(k), len(sorted_labels))
    if k <= 0:
      continue
    hits = float(sorted_labels[:k].sum())
    metrics[f"precision@{k}"] = hits / k
    metrics[f"recall@{k}"] = hits / total_positive

  return metrics


def eval_polymarket_candidate_ranking(model, candidate_data, n_neighbors, batch_size=200,
                                      k_values=(10, 50, 100), mutate_memory=False):
  """
  Evaluate candidate dependency edges produced for Polymarket.

  candidate_data should expose the same attributes as utils.data_processing.Data:
  sources, destinations, timestamps, edge_idxs, and labels. Labels are interpreted as
  dependency positives/negatives, not market resolution labels.
  """
  scores = score_temporal_edges(model=model,
                                sources=candidate_data.sources,
                                destinations=candidate_data.destinations,
                                timestamps=candidate_data.timestamps,
                                edge_idxs=candidate_data.edge_idxs,
                                n_neighbors=n_neighbors,
                                batch_size=batch_size,
                                mutate_memory=mutate_memory)
  metrics = ranking_metrics(candidate_data.labels, scores, k_values=k_values)
  metrics["scores"] = scores
  return metrics


def rank_polymarket_candidates(model, sources, destinations, timestamps, edge_idxs, n_neighbors,
                               batch_size=200, mutate_memory=False, top_k=100):
  """
  Return top-k scored candidate edges for manual inspection or symbolic verification.
  """
  scores = score_temporal_edges(model=model,
                                sources=np.asarray(sources),
                                destinations=np.asarray(destinations),
                                timestamps=np.asarray(timestamps),
                                edge_idxs=np.asarray(edge_idxs),
                                n_neighbors=n_neighbors,
                                batch_size=batch_size,
                                mutate_memory=mutate_memory)
  order = np.argsort(-scores)[:top_k]
  return [{
    "source": int(sources[idx]),
    "destination": int(destinations[idx]),
    "timestamp": float(timestamps[idx]),
    "edge_idx": int(edge_idxs[idx]),
    "score": float(scores[idx]),
  } for idx in order]

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, TemporalTypedNegativeEdgeSampler, \
  get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=None, help='Dimensions of the memory for '
                                                                'each node. Defaults to the node '
                                                                'feature dimension.')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--negative_sampler', type=str, default="random",
                    choices=["random", "typed_temporal"],
                    help='Negative sampler to use. typed_temporal keeps destination node types '
                         'compatible and avoids historical positives when possible.')
parser.add_argument('--progress_every_batches', type=int, default=50,
                    help='Print training progress every N optimizer steps. Use 0 to disable.')
parser.add_argument('--strict_memory_check', action='store_true',
                    help='Fail if the temporary and persisted memory update paths differ. '
                         'By default, training logs the drift and aligns persisted memory.')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
print(f"[setup] Arguments parsed for dataset={args.data}, prefix={args.prefix}", flush=True)

### Extract data for training, validation and testing
print(f"[data] Loading dataset {DATA} from ./data/ml_{DATA}.*", flush=True)
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
print("[data] Loaded "
      f"nodes={node_features.shape[0]}, node_dim={node_features.shape[1]}, "
      f"edge_features={edge_features.shape}, interactions={full_data.n_interactions}",
      flush=True)
print("[data] Split sizes "
      f"train={train_data.n_interactions}, val={val_data.n_interactions}, "
      f"test={test_data.n_interactions}, new_val={new_node_val_data.n_interactions}, "
      f"new_test={new_node_test_data.n_interactions}",
      flush=True)

if MEMORY_DIM is None:
  MEMORY_DIM = node_features.shape[1]
  print(f"[setup] --memory_dim not provided; using node feature dimension {MEMORY_DIM}",
        flush=True)
elif USE_MEMORY and MEMORY_DIM != node_features.shape[1]:
  raise ValueError(
    "This TGN implementation adds node memory to raw node features, so --memory_dim must match "
    f"the node feature dimension. Got memory_dim={MEMORY_DIM}, "
    f"node_dim={node_features.shape[1]}. Rerun with --memory_dim {node_features.shape[1]}."
  )

# Initialize training neighbor finder to retrieve temporal graph
print("[neighbors] Building training neighbor finder", flush=True)
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
print("[neighbors] Building full neighbor finder", flush=True)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

node_types = None
if args.negative_sampler == "typed_temporal":
  # Polymarket preprocessing stores one-hot node type columns first: market, token, topic.
  # For older datasets this still produces a harmless coarse type id.
  node_types = np.argmax(node_features[:, :3], axis=1) if node_features.shape[1] >= 3 else None

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
if args.negative_sampler == "typed_temporal":
  # The typed sampler is useful for heterogeneous Polymarket graphs: it samples negatives that
  # look structurally plausible instead of mixing market-node and token-node destinations.
  train_rand_sampler = TemporalTypedNegativeEdgeSampler(train_data.sources, train_data.destinations,
                                                       train_data.timestamps, node_types=node_types)
  val_rand_sampler = TemporalTypedNegativeEdgeSampler(full_data.sources, full_data.destinations,
                                                     full_data.timestamps, node_types=node_types,
                                                     seed=0)
  nn_val_rand_sampler = TemporalTypedNegativeEdgeSampler(new_node_val_data.sources,
                                                        new_node_val_data.destinations,
                                                        new_node_val_data.timestamps,
                                                        node_types=node_types, seed=1)
  test_rand_sampler = TemporalTypedNegativeEdgeSampler(full_data.sources, full_data.destinations,
                                                      full_data.timestamps, node_types=node_types,
                                                      seed=2)
  nn_test_rand_sampler = TemporalTypedNegativeEdgeSampler(new_node_test_data.sources,
                                                         new_node_test_data.destinations,
                                                         new_node_test_data.timestamps,
                                                         node_types=node_types, seed=3)
else:
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                        seed=1)
  test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
  nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                         new_node_test_data.destinations,
                                         seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
print(f"[device] Using {device}", flush=True)

# Compute time statistics
print("[time] Computing time-delta normalization statistics", flush=True)
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
print("[time] Stats "
      f"src_mean={mean_time_shift_src:.4f}, src_std={std_time_shift_src:.4f}, "
      f"dst_mean={mean_time_shift_dst:.4f}, dst_std={std_time_shift_dst:.4f}",
      flush=True)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)
  print(f"[run {i + 1}/{args.n_runs}] Initializing TGN model", flush=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            strict_memory_check=args.strict_memory_check)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  print(f"[run {i + 1}/{args.n_runs}] Training instances={num_instance}, "
        f"batches_per_epoch={num_batch}, batch_size={BATCH_SIZE}", flush=True)
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training
    print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Starting training", flush=True)

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      progress_step = (k // args.backprop_every) + 1
      progress_total = math.ceil(num_batch / args.backprop_every)
      progress_start_time = time.time()
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        # typed_temporal needs the positive destination and timestamp to avoid historical positives.
        # The default sampler ignores those details and preserves the original TGN behavior.
        if hasattr(train_rand_sampler, "sample_for_batch"):
          _, negatives_batch = train_rand_sampler.sample_for_batch(sources_batch, destinations_batch,
                                                                   timestamps_batch)
        else:
          _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())
      if args.progress_every_batches > 0 and (
          progress_step == 1 or progress_step % args.progress_every_batches == 0 or
          k + args.backprop_every >= num_batch):
        seen_instances = min(num_instance, (k + args.backprop_every) * BATCH_SIZE)
        elapsed_epoch = time.time() - start_epoch
        print(f"[epoch {epoch + 1}/{NUM_EPOCH}] "
              f"batch_step={progress_step}/{progress_total}, "
              f"seen={seen_instances}/{num_instance}, "
              f"loss={loss.item():.6f}, "
              f"step_time={time.time() - progress_start_time:.2f}s, "
              f"epoch_elapsed={elapsed_epoch:.1f}s",
              flush=True)

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Training done in {epoch_time:.1f}s; "
          "starting validation", flush=True)
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Validation old-node "
          f"AP={val_ap:.6f}, AUC={val_auc:.6f}", flush=True)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)
    print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Validation new-node "
          f"AP={nn_val_ap:.6f}, AUC={nn_val_auc:.6f}", flush=True)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Early stopping triggered; "
            f"loading epoch {early_stopper.best_epoch}", flush=True)
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
      print(f"[epoch {epoch + 1}/{NUM_EPOCH}] Saved checkpoint {get_checkpoint_path(epoch)}",
            flush=True)

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  print(f"[run {i + 1}/{args.n_runs}] Starting test evaluation", flush=True)
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  print(f"[run {i + 1}/{args.n_runs}] Test old-node AP={test_ap:.6f}, AUC={test_auc:.6f}; "
        f"new-node AP={nn_test_ap:.6f}, AUC={nn_test_auc:.6f}", flush=True)
  print(f"[run {i + 1}/{args.n_runs}] Saving model to {MODEL_SAVE_PATH}", flush=True)
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')

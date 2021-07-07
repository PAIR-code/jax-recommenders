"""Evaluation metrics for recommender systems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from typing import Union

from jax import numpy as jnp
import numpy as np
from scipy import sparse


def recall_binary_at_k(predictions: Union[jnp.ndarray, np.ndarray],
                       labels: Union[jnp.ndarray, np.ndarray, sparse.spmatrix],
                       k: int = 10) -> Union[jnp.ndarray, np.ndarray]:
  """Compute Recall@K for binary labels.

  Args:
    predictions: A matrix (numpy.array or jax.numpy.array) of shape [M,N]
      where M the number of users in the batch to be evaluated
      and N the total number of items such that the value at
      `predictions[m,n]` holds a float containing the predicted score for user
      index `m` on item index `n`. The score is used to rank the items
      that the model predicts the user will interact with.

    labels: A matrix (numpy.array, jax.numpy.array or scipy.sparse) of shape
      [M,N] where M the is are the number of users in the batch to be evaluated
      and N the total number of items such that the value at `labels[m,n]`
      (either 0 or 1) denotes if an item is relevant to the user, value 1
      (user had an interaction) or not value 0 (no interaction)
      This is the groundtruth over which the predictions are evaluated.

    k: Recall will be computed for top-k entities.
  Returns:
    recall@k scores over the batch of M users [M,].
  """
  batch_size = predictions.shape[0]
  # Corner case, return 1.0 if k is set k>= prediction.shape[1] and there
  # are relevant items in the labels.
  if k >= predictions.shape[1] and not any(labels.sum(axis=1) == 0):
    return np.ones((batch_size,)).astype(np.float32)

  top_k_indices = np.argpartition(-1.0*predictions, k, axis=1)
  predictions_binary = np.zeros_like(predictions, dtype=bool)
  predictions_binary[np.arange(batch_size)[:, np.newaxis],
                     top_k_indices[:, :k]] = True

  if  sparse.isspmatrix(labels):
    labels_binary = (labels > 0).toarray()
  else:
    labels_binary = np.asarray(labels > 0)
  recall = (np.logical_and(
      labels_binary, predictions_binary).sum(axis=1)).astype(np.float32)

  return recall / np.minimum(k, labels_binary.sum(axis=1))


def ndcg_binary_metric_at_k(predictions: Union[jnp.ndarray, np.ndarray],
                            labels: Union[jnp.ndarray, np.ndarray,
                                          sparse.spmatrix],
                            k: int = 10) -> Union[jnp.ndarray, np.ndarray]:
  """Compute NDCG@K for binary labels.

  Args:
    predictions: A matrix (numpy.array or jax.numpy.array) of shape [M,N]
      where M the number of users in the batch to be evaluated
      and N the total number of items such that the value at
      `predictions[m,n]` holds a float containing the predicted score for user
      index `m` on item index `n`. The score is used to rank the items
      that the model predicts the user will interact with.

    labels: A matrix (numpy.array, jax.numpy.array or scipy.sparse) of shape
      [M,N] where M the number of users in the batch to be evaluated
      and N the total number of items such that the value at `labels[m,n]`
      (either 0 or 1) denotes if an item is relevant to the user, value 1
      (user had an interaction) or not value 0 (no interaction)
      This is the groundtruth over which the predictions are evaluated.

    k: NDCG will be computed for top-k entities.
  Returns:
    ndcg@k score for each of the M users in the batch [M,].
  """
  batch_size = predictions.shape[0]
  if k > predictions.shape[1]:
    k = predictions.shape[1]

  if k < predictions.shape[1]:
    top_k_indices = np.argpartition(-1.0*predictions, k, axis=1)
  else:
    top_k_indices = np.arange(predictions.shape)
  top_k_scores = predictions[np.arange(batch_size)[:, np.newaxis],
                             top_k_indices[:, :k]]
  top_k_sorted_indices = np.argsort(-top_k_scores, axis=1)

  idx_topk = top_k_indices[np.arange(batch_size)[:, np.newaxis],
                           top_k_sorted_indices]
  # build the discount template
  tp = 1. / np.log2(np.arange(2, k + 2))

  if sparse.isspmatrix(labels):
    sums = labels.getnnz(axis=1)
    labels = labels.toarray()
  else:
    sums = labels.sum(axis=1, dtype=np.int32)
  dcg = (labels[np.arange(batch_size)[:, np.newaxis], idx_topk] *
         tp).sum(axis=1)
  idcg = np.array([(tp[:min(n, k)]).sum() for n in sums])
  return dcg/idcg

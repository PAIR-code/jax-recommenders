# Lint as: python3
"""Tests for google3.third_party.rudders.jax_recommenders.evaluation_metrics."""
from typing import List

import numpy as np
from scipy import sparse

from absl.testing import absltest
from absl.testing import parameterized
from lib.evaluation import evaluation_metrics


class EvaluationMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(('k_1', 1, [1, 1]), ('k_2', 2, [1, 0.5]),
                                  ('k_4', 4, [1, 0.5]), ('k_8', 8, [1, 1]),
                                  ('k_10', 10, [1, 1]))
  def test_recall_at_k(self, k_test: int, expected: List[float]):
    predictions = np.asarray([[0.9, 0, 0.8, 0.7, 0.2, 0, 0, 0],
                              [0.8, 0, 0, 0, 0.7, 0.5, 0.6, 0.2]])
    true_labels = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 1, 0]])
    true_labels_spr = sparse.csr_matrix(true_labels)
    recall = evaluation_metrics.recall_binary_at_k(
        predictions, true_labels, k=k_test)
    recall_spr = evaluation_metrics.recall_binary_at_k(
        predictions, true_labels_spr, k=k_test)
    self.assertSequenceEqual(expected, list(recall))
    self.assertSequenceEqual(expected, list(recall_spr))

  @parameterized.named_parameters(('k_1', 1, [1, 1]), ('k_2', 2, [1, 0.61]),
                                  ('k_4', 4, [1, 0.59]))
  def test_ndcg_metric_at_k(self, k_test: int, expected: List[float]):
    predictions = np.asarray([[0.9, 0, 0.8, 0.7, 0.2, 0, 0, 0],
                              [0.8, 0, 0, 0, 0.7, 0.5, 0.6, 0.2]])
    true_labels = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 1, 0]])
    true_labels_spr = sparse.csr_matrix(true_labels)
    ndcg = list(evaluation_metrics.ndcg_binary_metric_at_k(
        predictions, true_labels, k=k_test))
    ndcg_spr = list(evaluation_metrics.ndcg_binary_metric_at_k(
        predictions, true_labels_spr, k=k_test))
    ndcg = list(ndcg)
    self.assertEqual(expected[0], ndcg[0])
    self.assertAlmostEqual(expected[1], ndcg[1], places=2)
    ndcg_spr = list(ndcg_spr)
    self.assertEqual(expected[0], ndcg_spr[0])
    self.assertAlmostEqual(expected[1], ndcg_spr[1], places=2)

if __name__ == '__main__':
  googletest.main()


# Lint as: python3

# Copyright 2017 The Rudders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Tests for dataset_utils.

Note that Pandas DataFrame lists have been manually aligned so that you can read
a "row" from the named-column representation by simply reading "down". This
makes it much easier to verify and write the intended behaviour.

blaze test --test_output=streamed \
  third_party/py/jax_recommenders/datasets:dataset_utils_test
"""

from jax import random
from jax_recommenders.datasets.dataset_utils import DatasetsBuilder

import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest

# TODO(b/193004530): when/if JAX team supports a mock random generator, use that
# to avoid potential breakages if/when JAX team changes default random
# number generator.


class DatasetUtilsTest(googletest.TestCase):

  def test_add_per_user_time_deltas(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    input_df = pd.DataFrame({
        'user_id':    [0, 1, 0, 0, 3, 2],
        'item_id':    [0, 1, 2, 0, 1, 3],
        'timestamp':  [1, 2, 2, 2, 5, 9],
    })
    expected_df = pd.DataFrame({
        'user_id':    [0, 1, 0, 0, 3, 2],
        'item_id':    [0, 1, 2, 0, 1, 3],
        'timestamp':  [1, 2, 2, 2, 5, 9],
        # 'is_first' has 1 for the lowest timestamp of each user.
        'is_first':   [1, 1, 0, 0, 1, 1],
        # 'time_delta' has the timestamp difference from the previous review.
        # by the same user.
        'time_delta': [0, 0, 1, 1, 0, 0],
    })
    # pylint: enable=bad-whitespace
    # pyformat: enable
    actual_df = DatasetsBuilder.add_per_user_time_deltas(input_df)
    pd.util.testing.assert_frame_equal(expected_df, actual_df)

  def test_add_per_user_time_deltas_missing_column(self):
    # pylint: disable=bad-whitespace
    input_df = pd.DataFrame({
        'user_id': [0, 1, 0, 3, 2],
        'item_id': [0, 1, 2, 1, 3],
    })
    # pylint: enable=bad-whitespace
    with self.assertRaises(KeyError):
      DatasetsBuilder.add_per_user_time_deltas(input_df)

  def test_leave_last_one_out_split(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    input_df = pd.DataFrame({
        'user_id':   [0, 1, 2, 0, 1,  2,  0,  0,  3,  1 ],
        'timestamp': [1, 2, 2, 8, 15, 20, 21, 21, 25, 30],
        'item_id':   [0, 1, 2, 2, 2,  1,  0,  1,  2,  0 ],
    })
    # Expected Train is all but the last entry for each user.
    expected_train_df = pd.DataFrame({
        'user_id':   [0, 1, 2, 0, 1,  0 ],
        'timestamp': [1, 2, 2, 8, 15, 21],
        'item_id':   [0, 1, 2, 2, 2,  1 ],
    })
    # Test is the last entry for each user.
    expected_test_df = pd.DataFrame({
        'user_id':   [2,  0,  3,  1 ],
        'timestamp': [20, 21, 25, 30],
        'item_id':   [1,  0,  2,  0 ],
    })
    # pyformat: enable
    # pylint: enable=bad-whitespace

    train_df, test_df = DatasetsBuilder.leave_last_one_out_split(
        input_df, [0, 1, 2, 3])
    pd.util.testing.assert_frame_equal(expected_train_df,
                                       train_df.reset_index(drop=True))
    pd.util.testing.assert_frame_equal(expected_test_df,
                                       test_df.reset_index(drop=True))

  def test_leave_last_one_out_split_with_a_user_not_in_the_test(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    input_df = pd.DataFrame({
        'user_id':   [0, 1, 2, 0, 1,  2,  0,  0,  1 ],
        'timestamp': [1, 2, 2, 8, 15, 20, 21, 21, 30],
        'item_id':   [0, 1, 2, 2, 2,  1,  0,  1,  0 ],
    })
    # Expected Train is all but the last entry for each user.
    expected_train_df = pd.DataFrame({
        'user_id':   [0, 1, 2, 0, 1,  2,  0 ],
        'timestamp': [1, 2, 2, 8, 15, 20, 21],
        'item_id':   [0, 1, 2, 2, 2,  1,  1 ],
    })
    # Test is the last entry for each user.
    expected_test_df = pd.DataFrame({
        'user_id':   [0,  1 ],
        'timestamp': [21, 30],
        'item_id':   [0,  0 ],
    })
    # pyformat: enable
    # pylint: enable=bad-whitespace

    train_df, test_df = DatasetsBuilder.leave_last_one_out_split(
        input_df, [0, 1])
    pd.util.testing.assert_frame_equal(expected_train_df,
                                       train_df.reset_index(drop=True))
    pd.util.testing.assert_frame_equal(expected_test_df,
                                       test_df.reset_index(drop=True))

  def test_make_datasets_from_reviews_dataframe(self):
    """Tests making Datasets from a DataFrame of reviews.

    This is a fairly complex integration test that splits a dataset represented
    as a DataFrame of reviews (contains columns: ['user_id', 'timestamp',
    'user_rating', 'is_first', 'time_delta']) into a time separated train and
    eval set.
    """
    # pylint: disable=bad-whitespace
    # pyformat: disable
    all_items_df = pd.DataFrame({
        'user_id':     [10,11,12,10,11,12,10,10,14,11,13,13,15,15,16,16],
        'timestamp':   [1, 2, 2, 8, 15,20,21,21,25,30,33,34,34,36,38,40],
        'item_id':     [10,11,12,12,12,11,10,11,12,10,12,10,12,11,10,11],
        'user_rating': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 1, 3, 1, 1, 4 ],
    })
    # Note that Pandas' `assert_frame_equal` is sensitive to the order of
    # columns, and our ordering is non-determinitsic, so we sort the DataFrame
    # columns explicitly for the expected-columns and actual-columns to make
    # sure the names and their contents are equal; there isn't a simple default
    # option for this in Pandas testing library.
    expected_columns = sorted(['user_id', 'timestamp', 'item_id', 'user_rating',
                               'is_first', 'time_delta'])
    expected_train_df = pd.DataFrame({
        'user_id':     [0, 1, 2, 0, 1, 0, 4, 1, 3, 3, 5, 6, 6 ],
        'timestamp':   [1, 2, 2, 8, 15,21,25,30,33,34,34,38,40],
        'item_id':     [0, 1, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 1 ],
        'user_rating': [1.,2.,3.,4.,5.,3.,4.,5.,5.,1.,3.,1.,4.],
        'is_first':    [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0 ],
        'time_delta':  [0, 0, 0, 7,13, 13,0, 15,0, 1, 0, 0, 2 ],
    }).reindex(expected_columns, axis=1)
    expected_eval_df = pd.DataFrame({
        'user_id':     [2, 0, 5 ],
        'timestamp':   [20,21,36],
        'item_id':     [1, 0, 1 ],
        'user_rating': [1.,2.,1.],
        'is_first':    [0, 0, 0 ],
        'time_delta':  [18,13, 2 ],
    }).reindex(expected_columns, axis=1)
    # pyformat: enable
    # pylint: enable=bad-whitespace

    rng = random.PRNGKey(7)  # Results in picking test users: 2,5,0.

    with self.assertRaises(ValueError):
      datasets = DatasetsBuilder.make_datasets_from_reviews_dataframe(
          all_items_df, 100, rng)

    test_set_size = 3
    datasets = DatasetsBuilder.make_datasets_from_reviews_dataframe(
        all_items_df, test_set_size, rng)
    self.assertLen(expected_train_df.user_id, datasets.num_train)
    self.assertLen(expected_eval_df.user_id, datasets.num_eval)
    # The user and item IDs lists relate each original ID to the corresponding
    # "Dense" integer id (which is the position in the list).
    self.assertSequenceEqual([10, 11, 12, 13, 14, 15, 16], datasets.users)
    self.assertSequenceEqual([10, 11, 12], datasets.items)
    actual_train = tfds.as_dataframe(datasets.train)
    actual_train = actual_train.reindex(sorted(actual_train.columns), axis=1)
    pd.util.testing.assert_frame_equal(expected_train_df, actual_train)

    actual_eval = tfds.as_dataframe(datasets.eval)
    actual_eval = actual_eval.reindex(sorted(actual_eval.columns), axis=1)
    pd.util.testing.assert_frame_equal(expected_eval_df, actual_eval)

  def test_make_datasets_from_string_id_reviews_dataframe(self):
    """Test on string-ID'd reviews DataFrame into a recommendation Dataset."""
    # pylint: disable=bad-whitespace
    # pyformat: disable
    all_items_df = pd.DataFrame({
        'user_id':     ['b','a','c','a'],
        'timestamp':   [ 1,  2,  2,  8 ],
        'item_id':     ['x','z','y','y'],
        'user_rating': [ 1., 2., 3., 4.],
    })
    expected_columns = ['user_id', 'timestamp', 'item_id', 'user_rating',
                        'is_first', 'time_delta']
    expected_train_df = pd.DataFrame({
        'user_id':     [ 0,  2, ],
        'timestamp':   [ 2,  2, ],
        'item_id':     [ 2,  1, ],
        'user_rating': [ 2., 3. ],
        'is_first':    [ 1,  1, ],
        'time_delta':  [ 0,  0, ],
    }).reindex(sorted(expected_columns), axis=1)
    expected_eval_df = pd.DataFrame({
        'user_id':     [ 1,  0 ],
        'timestamp':   [ 1,  8 ],
        'item_id':     [ 0,  1 ],
        'user_rating': [ 1., 4.],
        'is_first':    [ 1,  0 ],
        'time_delta':  [ 0,  6 ],
    }).reindex(sorted(expected_columns), axis=1)
    # pyformat: enable
    # pylint: enable=bad-whitespace
    rng = random.PRNGKey(7)  # Results in picking test users: 0, 1 = a, b
    test_set_size = 2
    datasets = DatasetsBuilder.make_datasets_from_reviews_dataframe(
        all_items_df, test_set_size, rng)

    self.assertLen(expected_train_df.user_id, datasets.num_train)
    self.assertLen(expected_eval_df.user_id, datasets.num_eval)
    # The user and item IDs lists relate each original ID to the corresponding
    # "Dense" integer id (which is the position in the list).
    self.assertSequenceEqual(['a', 'b', 'c'], datasets.users)
    self.assertSequenceEqual(['x', 'y', 'z'], datasets.items)
    actual_train = tfds.as_dataframe(datasets.train)
    actual_train = actual_train.reindex(sorted(actual_train.columns), axis=1)
    pd.util.testing.assert_frame_equal(expected_train_df, actual_train)

    actual_eval = tfds.as_dataframe(datasets.eval)
    actual_eval = actual_eval.reindex(sorted(actual_eval.columns), axis=1)
    pd.util.testing.assert_frame_equal(expected_eval_df, actual_eval)


if __name__ == '__main__':
  googletest.main()

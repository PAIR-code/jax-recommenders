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

"""Input pipeline utilities."""

import itertools
from typing import Mapping, Iterable, Tuple, Sequence, Union

import dataclasses
from jax import numpy as jnp
from jax import random
import pandas as pd
import tensorflow as tf


@dataclasses.dataclass
class Datasets():
  """A Wrapper for a Recommender dataset.

  Contains:
    - The train and eval datasets with dense IDs.
    - The train and eval set sizes.
    - The relationship between the original IDs and the dense IDs.

  User and Item IDs from the original namespace corresponding to their dense
  indexes according to where they are in users/items list:
    users[i] = original user ID of dense user id i,
    items[j] = original item ID of dense item id j.

  Attributes:
    train: The training dataset.
    eval: The evaluation dataset.
    users: A sequence of all user IDs.
    items: A sequence of all item IDs.
    num_train: The number of examples in `train`.
    num_eval: The number of examples in `eval`.
  """
  # TODO(b/193004520): Rewrite as a single TFDS dataset with splits. See:
  #  https://www.tensorflow.org/datasets/add_dataset
  # Consider inheriting from `tfds.core.GeneratorBasedBuilder`
  # and writing the _split_generators method (using the below implementation)
  train: tf.data.Dataset
  eval: tf.data.Dataset
  users: Sequence[str]
  items: Sequence[str]
  num_train: int
  num_eval: int


def _create_tfds_from(df: pd.DataFrame) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices({
      'item_id': tf.cast(df['item_id'].values, dtype=tf.int32),
      'timestamp': tf.cast(df['timestamp'].values, dtype=tf.int32),
      'user_id': tf.cast(df['user_id'].values, dtype=tf.int32),
      'user_rating': tf.cast(df['user_rating'].values, dtype=tf.float32),
      # Note that is_first has value of 1 for true, and a value of 0 for false.
      'is_first': tf.cast(df['is_first'].values, dtype=tf.int32),
      'time_delta': tf.cast(df['time_delta'].values, dtype=tf.int32),
  })


def create_dense_map(id_set: Iterable[str]) -> Mapping[str, int]:
  """Creates a dense mapping from elements of id_set to integers."""
  return {id_str: i for (i, id_str) in enumerate(id_set)}


def time_diff(s: Iterable[Union[float, int]]) -> Union[float, int]:
  """Tiny helper to compute time diffs for window size 2 series."""
  first, second = itertools.islice(s, 2)
  return second - first


def add_per_user_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
  """Adds time differences between the reviews (rows) in the dataframe.

  Assumes:
    - `df` is sorted by time.
    - The time is specified in the column 'timestamp'.
    - User IDs are in the column 'user_id'.

  When the dataframe contains multiple identical timestamps, the time_delta and
  is_first entry will be the same for all of them.

  Args:
    df: Pandas dataframe. Must have 'user_id' and 'timestamp' columns.

  Returns:
    A new Pandas dataframe based on `df` with the following additional columns:
    - 'time_delta': Contains the time since the user's previous entry, and is 0
      for the user's first entry.
    - 'is_first': Contains 1 for the user's first review (w.r.t. timestamp), and
      is 0 otherwise.

  Raises:
    KeyError: If df is missing the 'user_id' or 'timestamp' column.
  """
  # Add the 'is_first' column.
  min_timestamps_df = df.groupby('user_id').agg(
      min_timestamp=('timestamp', 'min'))
  with_min_timestamp_df = df.join(min_timestamps_df, on='user_id')
  returned_df = df.copy()
  returned_df['is_first'] = (with_min_timestamp_df['min_timestamp'] ==
                             with_min_timestamp_df['timestamp']).apply(int)

  # Add 'time_delta' column.
  user_timestamp_groups = returned_df[['user_id',
                                       'timestamp']].drop_duplicates()
  # Every initial user's 'time_delta' will be NaN, fillna replaces that with 0.
  time_delta_df = (
      user_timestamp_groups.groupby('user_id')['timestamp'].rolling(
          2, min_periods=2).apply(time_diff).fillna(0).to_frame('time_delta'))
  user_timestamp_groups['time_delta'] = time_delta_df.droplevel(
      'user_id').sort_index()
  user_timestamp_groups = user_timestamp_groups.set_index(
      ['user_id', 'timestamp'])
  returned_df = returned_df.join(
      user_timestamp_groups, on=['user_id', 'timestamp'])
  # Correct for Pandas bug/oddidty where types get lost and operations often
  # turn integers into floats.
  return returned_df.astype({'is_first': 'int64', 'time_delta': 'int64'})


def leave_last_one_out_split(
    reviews_df: pd.DataFrame,
    eval_user_ids: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Splits reviews_df into train and eval splits.

  The eval split is the chronologically last review for each of eval_user_ids.
  When there are two last reviews, the fist one becomes the test review.

  Args:
    reviews_df: Each row is a review (a user and item interaction); it must
      contain a 'user_id' and a 'timestamp' column.
    eval_user_ids: Iterable of user ids that should be in the eval set. These
      must be formatted in the same way as they are in `df` (the '==' operation
      will be applied to filter for entries from a user).

  Returns:
    A pair `(train_df, eval_df)`: `train_df` is a dataframe containing training
    data (all but the last review for eval users, and all reviews for other
    users); `eval_df` is the dataframe containing the last review for every eval
    user.
  """
  eval_users_set = set(eval_user_ids)
  eval_user_selector = reviews_df['user_id'].apply(
      lambda uid: uid in eval_users_set)
  eval_users_df = reviews_df[eval_user_selector]
  # For each user_id, pick the review with the largest timestamp. Results in a
  # DataFrame where each index corresponds to an index in reviews_df with the
  # largest timestamp for that user.
  user_grouped_timestamps = eval_users_df.groupby(['user_id'])['timestamp']
  user_last_review_idx = (
      user_grouped_timestamps.apply(
          lambda group: group.nlargest(1)).reset_index(level=0, drop=False))
  eval_review_selector = reviews_df.index.isin(user_last_review_idx.index)
  eval_df = reviews_df[eval_review_selector]
  train_df = reviews_df[~eval_review_selector]
  return train_df, eval_df


def make_datasets_from_reviews_dataframe(reviews_df: pd.DataFrame,
                                         eval_users_count: int,
                                         rng: jnp.ndarray) -> Datasets:
  """Creates a Datasets object for a DataFrame of reviews.

  Note: the train and eval datasets have a shared dense ID space for users
  and items.

  Properties:
    - The 'user_id' and 'item_id' columns can contain integers or strings.
    - 'user_id' entries get reindexed to a dense mapping, from 0 to the number
      of user ids.
    - 'item_id' entries get reindexed to a dense mapping, from 0 to the number
      of item ids.
    - 'time_delta' entries get computed and added.
    - 'user_rating' entries get converted to floats.
    - The last time-stamped review for randomly picked set of `user_id`s (the
      size of which is specified by `eval_users_count`) gets turned into the
      eval dataset.

  Args:
    reviews_df: Pandas DataFrame of reviews. Assumes it has columns:
      - 'timestamp': int,
      - 'user_id': int or string,
      - 'item_id': int or string,
      - 'user_rating': int or float.
    eval_users_count: number of users to include in the eval set.
    rng: jax random number generator to select eval users.

  Returns:
    A `Datasets` object for the train/eval split. The eval set has the last
    review for the eval set of users. Given:
      `ds = make_datasets_from_reviews_dataframe(...)`
    Then the following invariants hold:
      - `len(ds.eval) == eval_users_count == ds.num_eval`.
      - `len(ds.eval) + len(ds.train) == len(reviews_df)`.
      - `len(ds.train) == ds.num_train`.
      - `ds.users` is the set of users sorted, the user IDs come from the
        user-representation's sort order List[str].sort() or List[int].sort().
      - `ds.items` is the set of items sorted the same way as ds.users.
      - `ds.eval` and `ds.train` have a 'time_delta' column: it has the time
        difference to the user's previous (by timestamp) review.
      - `ds.eval` and `ds.train` have a 'is_first' column: which is 1 for the
        set of minimal timestamp reviews (typically only one per user, but there
        may be more) and it is 0 for all other reviews.
      - The 'user_rating' column of `ds.eval` and `ds.train` is converted to a
        float.
      - Every review in `ds.eval` is a maximal w.r.t. that user's review
        timestamps.
      - The eval set may contain reviews which have `is_first` = 1 (providing
        constraints for this will happen in a followup CL).

  Raises:
    ValueError: When eval_users_count > the number of users in the dataframe.
  """
  # TODO(b/193004884): Provide a parameter to select users, e.g. only select
  # eval users that have more than some number of past reviews.

  # Sort in ascending order of time (entries earlier in time come first).
  reviews_df = reviews_df.sort_values('timestamp')
  # Adds deltas between reviews, named 'time_delta'.
  reviews_df = add_per_user_time_deltas(reviews_df)

  # Densify the user and item IDs.
  #
  # This maps item IDs and user IDs from the original IDs to integer IDs in the
  # range 0 to (num_items - 1) or (num_users - 1 ) respectively. This is so we
  # can access embeddings in a (num_items * embedding_dim) or
  # (num_users * embedding_dim) matrix easily.
  user_list = sorted(set(reviews_df['user_id']))
  user_ids_map = create_dense_map(user_list)
  reviews_df['user_id'] = reviews_df['user_id'].apply(
      lambda user_id: user_ids_map[user_id])

  item_list = sorted(set(reviews_df['item_id']))
  item_ids_map = create_dense_map(item_list)
  reviews_df['item_id'] = reviews_df['item_id'].apply(
      lambda item_id: item_ids_map[item_id])

  dense_user_list = [user_ids_map[user_id] for user_id in user_list]
  # Raises ValueError if eval_users_count > len(dense_user_list).
  eval_users = list(
      random.choice(
          rng,
          jnp.array(dense_user_list),
          shape=(eval_users_count,),
          replace=False))

  # Split train and eval.
  #
  # TODO(b/193004524): Support a dev set separate from train and eval.
  train_df, eval_df = leave_last_one_out_split(reviews_df, eval_users)
  ds_train = _create_tfds_from(train_df)
  ds_eval = _create_tfds_from(eval_df)

  return Datasets(
      train=ds_train,
      eval=ds_eval,
      users=user_list,
      items=item_list,
      num_train=len(train_df),
      num_eval=len(eval_df))

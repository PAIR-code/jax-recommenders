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

"""Provides functions to create a recommendation Datasets object for MovieLens.

For more details about the dataset see:
  https://www.tensorflow.org/datasets/catalog/movielens
Note there is an identical duplicate dataset named 'movie_lens', it does not
matter which is used.
"""
from typing import Mapping, Tuple, Optional

from jax import numpy as jnp
from jax import random
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.datasets import dataset_utils


def make_reviews_dataframe(all_movielens_df: pd.DataFrame) -> pd.DataFrame:
  """Makes a generic reviews Pandas DataFrame from a MovieLens DataFrame."""
  # Make named items and users IDs (items are movies).
  all_movielens_df['item_id'] = all_movielens_df['movie_id'].apply(
      lambda b: f'ml_mid_{b}')
  all_movielens_df['user_id'] = all_movielens_df['user_id'].apply(
      lambda b: f'ml_uid_{b}')
  # Throw away all columns we don't care about.
  return all_movielens_df[['item_id', 'user_id', 'user_rating', 'timestamp']]


def prepare_movielens_datasets(tfds_name: str = 'latest-small-ratings',
                               eval_users_count: int = 500,
                               rng: Optional[jnp.ndarray] = None,
                               random_seed: Optional[int] = None
                               ) -> dataset_utils.Datasets:
  """Prepares MovieLens datasets for Recommender training and eval.

  Loads and splits the dataset into train and eval. The eval subset is based on
  the most recent review of a random sample of users.

  Args:
    tfds_name: The MovieLens dataset name. The default is
      movielens/latest-small-ratings (100k ratings). Other valid names include:
      '25m', 'latest-small', '100k', '1m', '20m', The names are postfixed with
      '-ratings' for the ratings, and '-movies' for just the movie and genre
      information.
    eval_users_count: The number of eval users to include.
    rng: An optional random number generator to use for the split. If not
      provided, uses the `random_seed` argument.
    random_seed: An optional random seed integer to use if rng is not specified.

  Raises:
    ValueError: When eval_users_count > the number of users in the MovieLens
      dataset.
    ValueError: When random_seed and rng are None.

  Returns:
    A Datasets object with the train and eval splits.
  """
  ds_and_info: Tuple[Mapping[str, tf.data.Dataset], tfds.core.DatasetInfo] = (
      tfds.load(f'movielens/{tfds_name}', with_info=True))
  ds, ds_info = ds_and_info

  # We take ds['train'] because the MovieLens TFDS is not well designed, it
  # contains only a single split with all the data and calls that 'train'.
  all_movielens_df: pd.DataFrame = tfds.as_dataframe(ds['train'], ds_info)
  if rng is None and random_seed is None:
    raise ValueError('At least one of rng or random_seed must be defined.')
  if rng is None:
    rng = random.PRNGKey(random_seed)

  # Rename MovieLens-specific fields to the more generic reviews structured
  # DataFrame used in the dataset_utils code.
  reviews_df = make_reviews_dataframe(all_movielens_df)

  return dataset_utils.make_datasets_from_reviews_dataframe(
      reviews_df, eval_users_count, rng)

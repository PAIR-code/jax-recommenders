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
from typing import Mapping, Tuple
from jax_recommenders.datasets import dataset_utils

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


class Movielens(dataset_utils.DatasetsBuilder):
  """Builder class to create a recommendation Datasets for MovieLens."""

  @staticmethod
  def make_reviews_dataframe(all_movielens_df: pd.DataFrame) -> pd.DataFrame:
    """Makes a generic reviews Pandas DataFrame from a MovieLens DataFrame."""
    # Make named items and users IDs (items are movies).
    all_movielens_df['item_id'] = all_movielens_df['movie_id'].apply(
        lambda b: f'ml_mid_{b}')
    all_movielens_df['user_id'] = all_movielens_df['user_id'].apply(
        lambda b: f'ml_uid_{b}')
    # Throw away all columns we don't care about.
    return all_movielens_df[['item_id', 'user_id', 'user_rating', 'timestamp']]

  @classmethod
  def load_data_as_dataframe(cls,
                             tfds_name: str = 'latest-small-ratings',
                             ) -> pd.DataFrame:
    """Load MovieLens data from Tensorflow repositories.

    Args:
      tfds_name: The MovieLens dataset name. The default is
        movielens/latest-small-ratings (100k ratings). Other valid names include
        '25m', 'latest-small', '100k', '1m', '20m', The names are postfixed with
        '-ratings' for the ratings, and '-movies' for just the movie and genre
        information.

    Returns:
      A dataframe with the full movielens data.
    """
    ds_and_info: Tuple[Mapping[str, tf.data.Dataset], tfds.core.DatasetInfo] = (
        tfds.load(f'movielens/{tfds_name}', with_info=True))
    ds, ds_info = ds_and_info

    # We take ds['train'] because the MovieLens TFDS is not well designed, it
    # contains only a single split with all the data and calls that 'train'.
    all_movielens_df: pd.DataFrame = tfds.as_dataframe(ds['train'], ds_info)

    # Rename MovieLens-specific fields to the more generic reviews structured
    # DataFrame used in the dataset_utils code.
    return cls.make_reviews_dataframe(all_movielens_df)

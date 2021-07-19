# Copyright 2021 Google LLC
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

"""A collaborative Filtering Recommender model based on Deep Neural Networks.

This predict a user's rating for an item.
"""
from typing import Sequence

from flax import linen as nn
from jax import numpy as jnp


class CfDnn(nn.Module):
  """A simple DNN model.

  A neural network implementation of a collaborative filtering model.
  When layers = [] (no extra layers), this is equivalent to a standard
  collaborative filtering model (jnp.dot(item_embedding, user_embedding)).

  The model's score for a user/item pair is:
    (dot_product_output * half_range) + self.mean,
  i.e. this is a regression model, not a probabilistic one-hot encoding.

  When there are more layers, the user embedding is fed through the feedforward
  network and the output is dot-producted with the item embedding.

  Attributes:
    num_items: The total number of items in the dataset.
    num_users: The total number of users in the dataset.
    repr_size: The size of the embedding representation for users and items.
    layers: The sizes of the layers in the Deep Neural Network that are stacked
      on the user representation.
    mean: The mean of the ratings i.e. for [1,2,3,4,5] it is 3.0.
    half_range: Half the range of the ratings, e.g. for [1-5] ranged ratings,
      this is (5-1)/2 [i.e. range == mean (+ or -) 2].
  """
  num_items: int
  num_users: int
  repr_size: int
  # TODO(b/193013302): Implement a two towers model.
  layers: Sequence[int]
  mean = 3.0
  half_range = 2.0

  @nn.compact
  def __call__(self, batch_input):
    item_ids = batch_input['item_id']
    user_ids = batch_input['user_id']
    item_embeds = nn.Embed(
        num_embeddings=self.num_items, features=self.repr_size)(
            item_ids)
    user_embeds = nn.Embed(
        num_embeddings=self.num_users, features=self.repr_size)(
            user_ids)

    # Initial input to the DNN is simply the user-embedding.
    batch_x = user_embeds

    if self.layers:
      # TODO(b/193005140): Look into whether people normally use dropout and
      # batch/layer-norm for such recommendation systems.
      for layer_size in self.layers[:-1]:
        batch_x = nn.Dense(features=layer_size)(batch_x)
        batch_x = nn.relu(batch_x)
      batch_x = nn.Dense(features=self.layers[-1])(batch_x)
    # After the (optional) NN layers, the output score (in 0-1 range) is the dot
    # product of users and items, normalized by the (sqrt of the) representation
    # size.
    batch_y = (
        jnp.sum(jnp.multiply(item_embeds, batch_x), axis=1) /
        jnp.sqrt(self.repr_size))

    # Normalize range e.g. [1-5] for MovieLens like datasets.
    return batch_y * self.half_range + self.mean

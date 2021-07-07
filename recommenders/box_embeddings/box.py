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
"""Box model implementation.

   Inspired by https://github.com/iesl/gumbel-box-embeddings

   In box embeddings we assume that a concept can be associated with a
   hyper-cuboid (i.e. a box in vector space).
   In a recommender systems setting, a common set of concepts are users, items
   and their attributes. As an alternative to the various representations used,
   e.g. vector-based models, we use box embeddings for the representation of
   user preferences, items, and attributes.

"""
import jax.numpy as jnp


# Checks if box size is within limits
def _box_shape_ok(t: jnp.ndarray) -> bool:
  return len(t.shape) >= 2 and t.shape[-2] == 2


class BoxTensor(object):
  """A box represented as a tensor.

  It can represent single or multiple boxes stacked together.
  Each box is represented by its lower left, top right and center coordinates.

  Generally a box tensor is of the form: BoxTensor(b,2,d), where
  b = batch size on the experiment
  c = 2 = because of the (lower_left, top_right) coordinates
  d = the space dimensionality
  """

  def __init__(self, data: jnp.ndarray) -> None:
    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(
          f'Shape of data has to be (**,2,num_dims) but is {tuple(data.shape)}')
    super().__init__()

  def __repr__(self):
    return 'box_tensor(' + self.data.__repr__() + ')'

  @property
  def lower_left(self) -> jnp.ndarray:
    """Lower left coordinate of the box."""
    return self.data[..., 0, :]

  @property
  def top_right(self) -> jnp.ndarray:
    """Top right coordinate of the box."""
    return self.data[..., 1, :]

  @property
  def centre(self) -> jnp.ndarray:
    """Centre coordinate of the box."""
    return (self.lower_left + self.top_right) / 2

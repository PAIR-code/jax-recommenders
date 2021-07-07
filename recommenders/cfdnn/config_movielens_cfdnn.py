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

"""The configuration for a Deep Collaborative Filtering Recommender.

Includes hyperparameter sweeps.
"""
import datetime

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""

  datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

  config = ml_collections.ConfigDict()
  # Working directory for logs and related files.
  config.workdir = f'/tmp/jax_recommenders_{datetime_str}'
  # Which fragment of the MovieLens dataset to evaluate on.
  config.ds_name = 'latest-small-ratings'
  # The number of users to take the last review from for evaluation.
  config.eval_users_count = 500

  # Top level random seed which cannot be 0.
  config.seed = 7

  # Model Parameters.
  config.learning_rate = 0.003
  config.features_dim = 64
  config.batch_size = 64
  config.num_epochs = 10
  # Number of units in each layer separated by commas. For example:
  #  - [] = no layers = a simple collaborative filter.
  #  - [64,32] = 2 layers with 64 units in the first layer and 32 in the second.
  config.layers = []

  # Logging parameters.
  config.log_loss_every_steps = 100
  config.eval_every_steps = 100
  config.checkpoint_every_steps = 1000

  # TODO(ldixon): Check this, maybe remove.
  config.trial = 0  # Dummy for repeated runs.
  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  return h.sweep('config.trial', range(1))

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

"""Utilities for configuring JAX recommenders."""

from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

# TODO(ldixon): Once we use python 3.8, we can define the Batch type:
# Batch = Dict[Literal['user_rating', 'user_id', 'movie_id', 'timestamp',
#                      'time_delta'],
#              jnp.ndarray]

_CONFIG_FILE = config_flags.DEFINE_config_file(
    'config',
    None,
    'Training configuration, assumed to have a workdir:str parameter.',
    lock_config=True)

flags.mark_flags_as_required(['config'])

# Flags --jax_backend_target and --jax_xla_backend are available through JAX.
FLAGS = flags.FLAGS


def setup_jax():
  """Borg setup for JAX programs."""
  config = _CONFIG_FILE.value

  if not isinstance(config.workdir, str):
    raise ValueError('--config.workdir must be set to path string')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  if FLAGS.jax_backend_target:
    logging.info('Using JAX backend target %s', FLAGS.jax_backend_target)
    jax_xla_backend = FLAGS.jax_xla_backend or 'None'
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('JAX host: %d / %d', jax.host_id(), jax.host_count())
  logging.info('JAX devices: %r', jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  # Note that the task 0 is not guaranteed to be host 0.
  platform.work_unit().set_task_status(
      f'host_id: {jax.host_id()}, host_count: {jax.host_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       config.workdir, 'workdir')

  tf.io.gfile.makedirs(config.workdir)

  return config

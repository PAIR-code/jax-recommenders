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

"""A Collaborative-Filtering DNN recommender for MovieLens."""

from typing import Dict, Tuple, Union, List, Any

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
import flax
from flax import optim
import flax.linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp

from jax_recommenders.datasets.movielens import Movielens
from jax_recommenders.recommenders.cfdnn import model_cfdnn
from jax_recommenders.utils import progress_mngr
from jax_recommenders.utils import train_utils

import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# X-Manager metadata.
_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'movielens', 'Name of the experiment.', short_name='n')
_TAGS = flags.DEFINE_list(
    'tags', '', 'Comma-separated list of tags to add to default tags.')


@flax.struct.dataclass
class TrainState:
  """The data to checkpoint a model.

  Note: This can't contain the model because this needs to be serializable.

  Attributes:
    step: The current step index of training, i.e. number of previous batches
      trained on so far.
    optimizer: The optimizer being used, which holds its own state.
  """
  step: int
  optimizer: flax.optim.Optimizer


# TODO(b/193009736): Think about the right type abstractions for this function.
def eval_step(
    model: nn.Module, params_dict: Dict[str, Any],
    batch: Dict[str, jnp.ndarray]) -> Tuple[Union[float, List[float]], Any]:
  """Evaluates the given model on the batch.

  Args:
    model: Module to compute predictions.
    params_dict: Replicated Flax/Linnen parameters dict.
    batch: Inputs that should be evaluated.

  Returns:
    Loss and predictions.
  """
  preds = model.apply(params_dict, batch)
  loss = jnp.sum(jnp.mean((preds - batch['user_rating'])**2, axis=-1))
  return loss, preds


# A helper to use progress manager that provides a loss evaluation function for
# logging loss.
def make_loss_eval_fn(
    model: nn.Module,
    batched_eval_ds: tf.data.Dataset) -> progress_mngr.LossEvaluatorFn:
  """Creates a loss function from a state evaluation function."""

  def loss_eval_fn(state: TrainState) -> float:
    # Note: eval_step(...)[0] selects the first of the tuple of (loss,
    # predictions), so that we average just the loss.
    return np.mean([
        eval_step(model, {'params': state.optimizer.target}, batch)[0]
        for batch in tfds.as_numpy(batched_eval_ds)
    ])

  return loss_eval_fn


def train_step(model: nn.Module, state: TrainState,
               batch: Dict[str, jnp.ndarray]) -> Tuple[TrainState, float]:
  """Performs a single training step.

  Args:
    model: A model configured for the batch size.
    state: Current training state. Updated training state will be returned.
    batch: Training inputs for this step.

  Returns:
    Tuple of: the updated state, and the loss (mean squared error).
  """
  grad_fn = jax.value_and_grad(
      lambda params: eval_step(model, {'params': params}, batch), has_aux=True)
  (loss, _), grad = grad_fn(state.optimizer.target)

  optimizer = state.optimizer.apply_gradient(grad)
  # The pytype annotation is needed because
  #   https://github.com/google/jax/issues/2371
  new_state = state.replace(optimizer=optimizer, step=state.step + 1)  # pytype: disable=attribute-error
  return new_state, loss


def train_and_evaluate(config: ml_collections.ConfigDict):
  """Trains and evaluates the model.

  The training will run for config.num_epochs.

  Args:
    config: The configuration to use; specifies training parameters such as the
      number of epochs to train for as well as model parameters like the number
      of dimensions embeddings should have. For an example config, see:
        ./config_movielens_cfdnn.py
  """
  logging.info('Loading MovieLens dataset: %s', config.ds_name)
  # --------------------------------------------------------------------------
  # Load the dataset.
  rng = jax.random.PRNGKey(config.seed)
  rng, model_rng, dataset_rng = jax.random.split(rng, 3)
  datasets = Movielens.prepare_datasets(
      config.ds_name, rng=dataset_rng, random_seed=config.seed,
      eval_users_count=config.eval_users_count)
  batched_train_ds = datasets.train.batch(config.batch_size)
  # We batch the eval set as well so that we can use the same model directly
  # (because the model will be configured in memory to run on that given batch
  # size).
  batched_eval_ds = datasets.eval.batch(config.batch_size)

  num_train_batches = len(batched_train_ds)

  # --------------------------------------------------------------------------
  # Initialize model & optimizer.
  logging.info('Initializing model...')
  model = model_cfdnn.CfDnn(
      num_items=len(datasets.items),
      num_users=len(datasets.users),
      repr_size=config.features_dim,
      layers=config.layers)

  # Initialize variables using batch to provide fake init data to setup correct
  # JAX array sizes.
  variables = model.init(model_rng,
                         next(tfds.as_numpy(batched_train_ds).__iter__()))
  # We disable typechecking because while flax modules have what is needed here,
  # the parameter_overview library hasn't yet specified JAX modules as a valid
  # type.
  parameter_overview.log_parameter_overview(variables)  # pytype: disable=wrong-arg-types
  optimizer = optim.Adam(learning_rate=config.learning_rate).create(
      variables['params'])
  state = TrainState(step=0, optimizer=optimizer)
  # TODO(b/193005145): Consider if this can be wrapped into the progress manager
  # which is already responsible for saving the checkpoints too.
  state = checkpoints.restore_checkpoint(config.workdir, state)

  epochs_done_count, steps_in_epoch_done = divmod(state.step, num_train_batches)
  logging.info('Starting at: epochs_done_count=%d, steps_in_epoch_done=%d',
               epochs_done_count, steps_in_epoch_done)

  # --------------------------------------------------------------------------
  # Start metrics writer/logging for TensorBoard.
  loss_eval_fn = make_loss_eval_fn(model, batched_eval_ds)
  progress_manager = progress_mngr.ProgressManager(config, loss_eval_fn,
                                                   num_train_batches)
  progress_manager.writer.write_hparams(dict(config))

  # --------------------------------------------------------------------------
  # Main Training Loop.
  #
  # TODO(b/193005016): Support early stopping.
  logging.info('Starting training loop at step %d.', state.step)
  with metric_writers.ensure_flushes(progress_manager.writer):
    while epochs_done_count <= config.num_epochs:
      for batch in tfds.as_numpy(batched_train_ds.skip(steps_in_epoch_done)):
        with jax.profiler.StepTraceAnnotation('train', step_num=state.step):
          state, batch_loss = train_step(model, state, batch)
        progress_manager.maybe_report_metrics(state, batch_loss)
        progress_manager.maybe_save_checkpoint(state)
        # At the end of the epoch, start at the beginning of the next epoch.
        steps_in_epoch_done = 0
      epochs_done_count += 1


def main(argv):
  del argv
  config = train_utils.setup_jax()
  train_and_evaluate(config)


if __name__ == '__main__':
  # Provide access to the --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)

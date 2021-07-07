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

"""A helper library to manage reporting progress and saving checkpoints."""

# TODO(ldixon): Think of and add an appropriate test.

from typing import Callable, Any

from clu import metric_writers
from clu import periodic_actions
from flax.training import checkpoints
import jax
import ml_collections
import numpy as np

# Note: The intended type of 'TrainState' is that it has 'step: int' parameter.
# But python's type system can't express this, so it's Any here.
TrainState = Any

# A function that evaluates loss to be reported every evaluation cycle.
# TODO(ldixon): Generalize the returned float loss to a dict of metrics.
LossEvaluatorFn = Callable[[TrainState], float]


class ProgressManager:
  """Manages and reports on progress (metrics and checkpoints).

  Attributes:
    writer: The TensorFlow metrics writer.
  """
  writer: metric_writers.MetricWriter

  def __init__(self, config: ml_collections.ConfigDict,
               eval_fn: LossEvaluatorFn, num_train_batches: int) -> None:
    self._workdir: str = config.workdir

    self._eval_every_steps: int = config.eval_every_steps
    self._log_loss_every_steps: int = config.log_loss_every_steps
    self._checkpoint_every_steps: int = config.checkpoint_every_steps

    self._num_train_batches: int = num_train_batches
    self._num_epochs: int = config.num_epochs
    self._num_train_steps: int = self._num_epochs * self._num_train_batches

    self._eval_fn: LossEvaluatorFn = eval_fn

    self.writer = metric_writers.create_default_writer(
        self._workdir, just_logging=jax.host_id() > 0)
    self._progress_reporter = periodic_actions.ReportProgress(
        num_train_steps=self._num_train_steps, writer=self.writer)

  def is_last_step(self, state: TrainState) -> bool:
    epoch, epoch_step = divmod(state.step, self._num_train_batches)
    return epoch_step == self._num_train_batches and epoch == self._num_epochs

  def maybe_report_metrics(self, state: TrainState, batch_loss: float) -> None:
    """Report metrics when appropriate."""
    is_last_step = self.is_last_step(state)
    if state.step % self._eval_every_steps == 0 or is_last_step:
      with self._progress_reporter.timed('eval'):
        eval_mean = self._eval_fn(state)
        self.writer.write_scalars(state.step, {
            'eval_MSE_loss': eval_mean,
            'eval_RMSE': np.sqrt(eval_mean),
        })
    if state.step % self._log_loss_every_steps == 0 or is_last_step:
      self.writer.write_scalars(
          state.step, {
              'train_batch_MSE_loss': batch_loss,
              'train_batch_RMSE': np.sqrt(batch_loss),
          })

  # TODO(ldixon): Keep the best checkpoint instead of the last 3.
  def maybe_save_checkpoint(self, state: TrainState) -> None:
    is_last_step = self.is_last_step(state)
    if state.step % self._checkpoint_every_steps == 0 or is_last_step:
      checkpoints.save_checkpoint(self._workdir, state, state.step, keep=3)

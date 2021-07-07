"""Tests for box."""

from absl import logging
import jax.numpy as jnp

from lib.recommenders.box_embeddings import box
from absl.testing import absltest


class BoxTest(googletest.TestCase):

  def test_correct_box_construction(self):
    a = jnp.array([1, 2, -1, 5, 0, 2, -2, 3, -3, 3, -2, 4])
    a = a.reshape(3, 2, 2)
    a_box = box.BoxTensor(a)
    self.assertIsInstance(a_box, box.BoxTensor)

  def test_wrong_box_construction(self):
    a = jnp.array([1, 2, -1, 5, 0, 2, -2, 3, -3, 3, -2, 4])
    a = a.reshape(2, 3, 2)
    with self.assertRaisesRegex(ValueError, 'Shape of data has to be'):
      box.BoxTensor(a)

  def create_a_box(self, array, b, d):
    """Create a BoxTensor with dims (b, 2, d) from array."""
    a = jnp.array(array)
    a = a.reshape(b, 2, d)
    a_box = box.BoxTensor(a)
    logging.info('Created a box tensor with elements: %s', a_box.data)
    return a_box

  def test_lower_left(self):
    # Creates a BoxTensor(3, 2, 2)
    a_box = self.create_a_box([1, 2, -1, 5, 0, 2, -2, 3, -3, 3, -2, 4], 3, 2)
    # check lower left coordinate
    # lower_left shape is (3, 2)
    lower_left = jnp.array([[1, 2, 0], [2, -3, 3]])
    self.assertEqual(lower_left.all(), a_box.lower_left.all())

  def test_top_right(self):
    # Creates a BoxTensor(3, 2, 2)
    a_box = self.create_a_box([1, 2, -1, 5, 0, 2, -2, 3, -3, 3, -2, 4], 3, 2)
    # check top right coordinate
    # top_right shape is (3, 2)
    top_right = jnp.array([[-1, 5, -2], [3, -2, 4]])
    self.assertEqual(top_right.all(), a_box.top_right.all())

  def test_centre(self):
    # Creates a BoxTensor(3, 2, 2)
    a_box = self.create_a_box([1, 2, -1, 5, 0, 2, -2, 3, -3, 3, -2, 4], 3, 2)
    # check centre coordinate
    # centre shape is (3, 2)
    centre = jnp.array([[0.0000, 3.5000, -1.0000], [2.5000, -2.5000, 3.5000]])
    self.assertEqual(centre.all(), a_box.centre.all())

if __name__ == '__main__':
  googletest.main()

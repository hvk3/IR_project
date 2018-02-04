# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils

from tensorflow import logging
def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"]):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64)}
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names], 1)

    return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])

import tensorflow as tf

from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import readers

FLAGS = flags.FLAGS

flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
              "The directory to save the model files in.")
flags.DEFINE_string(
  "train_data_pattern", "",
  "File glob for the training dataset. If the files refer to Frame Level "
  "features (i.e. tensorflow.SequenceExample), then set --reader_type "
  "format. The (Sequence)Examples are expected to have 'rgb' byte array "
  "sequence feature as well as a 'labels' int64 context feature.")
flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
              "to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

# Model flags.
flags.DEFINE_bool(
  "frame_features", False,
  "If set, then --train_data_pattern must be frame-level features. "
  "Otherwise, --train_data_pattern must be aggregated video-level "
  "features. The model must also be set appropriately (i.e. to read 3D "
  "batches VS 4D batches.")

# Training flags.
flags.DEFINE_integer("batch_size", 1024,
               "How many examples to process per batch for training.")
flags.DEFINE_string("label_loss", "CrossEntropyLoss",
              "Which loss function to use for training the model.")

# Other flags.
flags.DEFINE_integer("num_readers", 8,
               "How many threads to use for reading input files.")


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  reader = readers.YT8MAggregatedFeatureReader(
    feature_names=feature_names, feature_sizes=feature_sizes)

  return reader

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


if __name__ == "__main__":

    reader = get_reader()

    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            reader,
            FLAGS.train_data_pattern,
            batch_size=FLAGS.batch_size,
            num_readers=FLAGS.num_readers,
            num_epochs=FLAGS.num_epochs))

    tf.summary.histogram("model/input_raw", model_input_raw)
    feature_dim = len(model_input_raw.get_shape()) - 1

    # video_id, model_input_raw, labels_batch, num_frames = (
    #     reader.prepare_serialized_examples(serialized_examples))


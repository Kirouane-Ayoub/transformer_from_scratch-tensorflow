import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, file_path: str, batch_size: int = 64, buffer_size: int = 20000):
        self.file_path = file_path
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def load_data(self):
        # Load the data from the file
        data = np.load(self.file_path)

        # Create a TensorFlow Dataset from the data
        dataset = tf.data.Dataset.from_tensor_slices(data)

        # Shuffle and batch the dataset
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)

        return dataset

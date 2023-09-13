import tensorflow as tf
from transformer import Transformer
from dataset import Dataset
from utils import create_masks

def train_transformer(dataset, transformer, num_epochs=20):
    """
    Trains the transformer model.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(num_epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = transformer(inp, tar_inp, 
                                             True, 
                                             enc_padding_mask, 
                                             combined_mask, 
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)    
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()*100}')

def main():
    # Create a Dataset instance
    dataset = Dataset("data.npy")

    # Load and preprocess the data
    data = dataset.load_data()

    # Create a Transformer instance
    transformer = Transformer(num_layers=2, d_model=512, num_heads=8, 
                              dff=2048, input_vocab_size=8500, target_vocab_size=8000, 
                              pe_input=10000, pe_target=6000)

    # Train the transformer model
    train_transformer(data, transformer)

    # Save the transformer model
    transformer.save_weights("transformer.h5")

if __name__ == "__main__":
    main()

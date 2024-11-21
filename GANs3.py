import mlflow
from typing import Any
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def create_mlflow_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id

class GAN:
    def __init__(self, input_dim, noise_dim, text_dim):
        self.input_dim = input_dim  
        self.noise_dim = noise_dim  
        self.text_dim = text_dim    
        # Create the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Optimizers for both models
        self.generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # Loss function
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.input_dim + self.text_dim, activation='sigmoid')  
        ])
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.input_dim + self.text_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def train_step(self, real_samples, text_samples):
        batch_size = tf.shape(real_samples)[0]

        # Combine real samples and text samples
        real_samples_full = tf.concat([real_samples, text_samples], axis=1)

        # Generate random noise and create fake samples
        noise = tf.random.normal([batch_size, self.noise_dim])
        generated_samples = self.generator(noise)

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(real_samples_full)
            fake_output = self.discriminator(generated_samples)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_samples = self.generator(noise)
            fake_output = self.discriminator(generated_samples)
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return disc_loss, gen_loss

    def train(self, real_data, text_data, epochs=100, batch_size=128):
        Name_of_experiment = "GANs_3_1"
        experiment_id = create_mlflow_experiment(
            experiment_name=Name_of_experiment,
            artifact_location="testing_mlflow1_Gans_3_1",
            tags={"env": "dev", "version": "1.0.5"},
        )
        mlflow.set_experiment(Name_of_experiment)
        with mlflow.start_run() as run:
            print(run.info.run_id)
            print(experiment_id)
            mlflow.log_param("total_epochs", epochs)  # Log total number of epochs
            mlflow.log_param("batch_size", batch_size)  # Log batch size

            last_disc_loss = None
            last_gen_loss = None

            for epoch in range(epochs):
                for i in range(0, len(real_data), batch_size):
                    real_samples = real_data[i:i+batch_size]
                    text_samples = text_data[i:i+batch_size]

                # Perform one training step
                d_loss, g_loss = self.train_step(real_samples, text_samples)
         
                #log maetrics Mlflow
                mlflow.log_metric("discriminator_loss", d_loss.numpy(), step=epoch)
                mlflow.log_metric("generator_loss", g_loss.numpy(), step=epoch)

                # Store the last losses for logging after the last epoch
                last_disc_loss = d_loss.numpy()
                last_gen_loss = g_loss.numpy()

                # Print progress
                print(f"Epoch {epoch+1}/{epochs} [D loss: {d_loss.numpy()}] [G loss: {g_loss.numpy()}]")
# Log the final losses as parameters for the last epoch
            mlflow.log_param("last_discriminator_loss", last_disc_loss)
            mlflow.log_param("last_generator_loss", last_gen_loss)



    def generate_samples(self, num_samples, encoder):
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        generated_samples = self.generator.predict(noise)

        # Split generated samples back into numerical and text
        generated_numerical = generated_samples[:, :self.input_dim]
        generated_text = generated_samples[:, self.input_dim:]

        # Apply constraints: city_pop > 1000, amt > 1
        generated_numerical[:, 3] = 1000 + np.abs(generated_numerical[:, 3]) * 9000  
        generated_numerical[:, 0] = 1 + np.abs(generated_numerical[:, 0]) * 10000   

        # Decode the one-hot encoded text back to original categories
        generated_text_decoded = encoder.inverse_transform(generated_text)

        # Combine the numerical and text data into one array
        return np.concatenate([generated_numerical, generated_text_decoded], axis=1)

# Function to create synthetic data using GAN
def create_synthetic_data(real_data, text_data, input_dim, noise_dim, epochs=100, batch_size=128):
    # OneHotEncode text columns
    encoder = OneHotEncoder(sparse_output=False)  
    encoded_text_data = encoder.fit_transform(text_data)

    # Define text_dim as the number of unique categories across all text columns
    text_dim = encoded_text_data.shape[1]

    # Initialize the GAN model with the encoded text dimensions
    gan = GAN(input_dim=input_dim, noise_dim=noise_dim, text_dim=text_dim)
    gan.train(real_data, encoded_text_data, epochs=epochs, batch_size=batch_size)
    
    # Generate synthetic data
    synthetic_data = gan.generate_samples(real_data.shape[0], encoder)

    return synthetic_data

if __name__ == "__main__":
    
    data = pd.read_csv('fraudTrain.csv')
    data = data.sample(n=50000, random_state=42, replace=True)
    
    ml_columns = ['amt', 'lat', 'long', 'city_pop']  
    text_columns = ['merchant', 'category', 'job']  

    # Split data into numerical and text features
    real_data_numerical = data[ml_columns].values  
    real_data_text = data[text_columns]  
    input_dim = real_data_numerical.shape[1]  
    noise_dim = 100  

    # Generate synthetic data
    synthetic_data = create_synthetic_data(real_data_numerical, real_data_text, input_dim=input_dim, noise_dim=noise_dim, epochs=1000, batch_size=128)

    # Convert synthetic_data array to a DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=ml_columns + text_columns)

    # Ensure city_pop is an integer
    synthetic_df['city_pop'] = synthetic_df['city_pop'].astype(int)

    # Generate label column
    labels = random.choices([0, 1], k=len(synthetic_df))

    # Save synthetic data
    synthetic_df['is_fraud'] = labels  
    synthetic_df.to_csv('synthetic_data3_2.csv', index=False)
    print("Synthetic data generated and saved")

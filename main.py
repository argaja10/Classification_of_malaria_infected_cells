import random
import glob
from data_loader import ImageDataLoader
from autoencoder import AutoEncoder
from anomaly_detector import AnomalyDetector

# Constants
SIZE = 128
BATCH_SIZE = 64

def train(model, train_generator, validation_generator, epochs=50):
    """
    Trains the autoencoder model using the training and validation generators.
    """
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=epochs,
        verbose=1
    )

def main():
    """
    Main function to orchestrate the data loading, model training, and anomaly detection.
    """
    data_loader = ImageDataLoader(SIZE, BATCH_SIZE)
    train_generator, validation_generator, anomaly_generator = data_loader.get_generators()
    
    autoencoder = AutoEncoder(input_shape=(SIZE, SIZE, 3))
    train(autoencoder.model, train_generator, validation_generator, epochs=50)
    
    anomaly_detector = AnomalyDetector(autoencoder, SIZE)
    train_batch = train_generator.next()[0]
    anomaly_batch = anomaly_generator.next()[0]
    
    train_values, anomaly_values = anomaly_detector.fit(train_batch, anomaly_batch)
    
    para_file_paths = glob.glob('cell_images2/parasitized/images/*')
    uninfected_file_paths = glob.glob('cell_images2/uninfected_train/images/*')
    
    # Anomaly image verification
    num = random.randint(0, len(para_file_paths) - 1)
    anomaly_detector.check_anomaly(para_file_paths[num])
    
    # Good/normal image verification
    num = random.randint(0, len(uninfected_file_paths) - 1)
    anomaly_detector.check_anomaly(uninfected_file_paths[num])

if __name__ == "__main__":
    main()

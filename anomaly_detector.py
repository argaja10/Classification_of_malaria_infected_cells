from PIL import Image
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, autoencoder, size):
        """
        Initializes the AnomalyDetector with the given autoencoder model and image size.
        """
        self.autoencoder = autoencoder
        self.size = size
        self.kde = None
        self.out_vector_shape = None
    
    def calculate_density_and_recon_error(self, images):
        """
        Calculates and returns the density and reconstruction error statistics for the given images.
        """
        encoded_imgs = self.autoencoder.encoder_model.predict(images)
        encoded_imgs = [np.reshape(img, self.out_vector_shape) for img in encoded_imgs]
        self.kde = KernelDensity(kernel='gaussian').fit(encoded_imgs)
        
        density = self.kde.score_samples(encoded_imgs)
        average_density = np.mean(density)
        stdev_density = np.std(density)
        
        reconstructions = self.autoencoder.model.predict(images)
        recon_error_list = [self.autoencoder.model.evaluate([recon], [[img]], batch_size=1, verbose=0)[0] for recon, img in zip(reconstructions, images)]
        
        average_recon_error = np.mean(recon_error_list)
        stdev_recon_error = np.std(recon_error_list)
        
        return average_density, stdev_density, average_recon_error, stdev_recon_error
    
    def fit(self, train_images, anomaly_images):
        """
        Fits the KDE model on the training images and calculates the density and reconstruction error statistics for both training and anomaly images.
        """
        train_values = self.calculate_density_and_recon_error(train_images)
        anomaly_values = self.calculate_density_and_recon_error(anomaly_images)
        self.out_vector_shape = train_images[0].shape
        return train_values, anomaly_values
    
    def check_anomaly(self, img_path):
        """
        Checks if the given image is an anomaly based on the density and reconstruction error thresholds.
        """
        density_threshold = 2500
        reconstruction_error_threshold = 0.004
        img = Image.open(img_path)
        img = np.array(img.resize((self.size, self.size), Image.ANTIALIAS)) / 255.
        img = img[np.newaxis, :,:,:]
        
        encoded_img = self.autoencoder.encoder_model.predict([img])
        encoded_img = np.reshape(encoded_img, self.out_vector_shape)
        density = self.kde.score_samples([encoded_img])[0]
        
        reconstruction = self.autoencoder.model.predict([img])
        reconstruction_error = self.autoencoder.model.evaluate([reconstruction], [img], batch_size=1, verbose=0)[0]
        
        if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
            print("The image is an anomaly")
        else:
            print("The image is NOT an anomaly")

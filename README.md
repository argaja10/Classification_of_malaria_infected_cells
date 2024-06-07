

This project demonstrates the detection of infected class of malarial images using AutoEncoders. The project sorts entire images as either normal or infected. The bottleneck layer output from the autoencoder is considered the latent space.

The code uses the malarial dataset but can be easily applied to other applications.

## Data Source

The data is sourced from [Malaria Datasets](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html).

## Project Structure

The project is organized into multiple files for modularity:

- `data_loader.py` - Contains the `ImageDataLoader` class for loading image data.
- `autoencoder.py` - Contains the `AutoEncoder` class for building the autoencoder model architecture.
- `anomaly_detector.py` - Contains the `AnomalyDetector` class for handling anomaly detection.
- `main.py` - Contains the main script to orchestrate data loading, model training, and anomaly detection.

## Requirements

- TensorFlow
- Keras
- PIL (Pillow)
- Matplotlib
- NumPy
- scikit-learn

You can install the required packages using pip:
```bash
pip install tensorflow keras pillow matplotlib numpy scikit-learn

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

class AutoEncoder:
    def __init__(self, input_shape):
        """
        Initializes the AutoEncoder with the given input shape.
        """
        self.model, self.encoder_model = self.build_autoencoder(input_shape)
        
    def build_autoencoder(self, input_shape):
        """
        Builds and returns the autoencoder and encoder models.
        """
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
        encoder_model = Model(inputs=model.input, outputs=model.layers[6].output)
        
        return model, encoder_model

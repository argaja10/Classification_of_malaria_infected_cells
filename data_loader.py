from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageDataLoader:
    def __init__(self, size, batch_size):
        """
        Initializes the ImageDataLoader with image size and batch size.
        """
        self.size = size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255)
        
    def get_generators(self):
        """
        Creates and returns the training, validation, and anomaly data generators.
        """
        train_generator = self.datagen.flow_from_directory(
            'data/uninfected_train/',
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='input'
        )
        
        validation_generator = self.datagen.flow_from_directory(
            'data/uninfected_test/',
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='input'
        )
        
        anomaly_generator = self.datagen.flow_from_directory(
            'data/parasitized/',
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='input'
        )
        
        return train_generator, validation_generator, anomaly_generator

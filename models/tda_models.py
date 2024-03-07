
from tensorflow.keras import models, layers, losses

class TDA_PI34_Model(): 
    """
    
    CNN model that works on persistance images generated from TDA pipeline. 
    Input data should be 28x28 pixel images with 34 channels (hence name) 
    (17 different filtration x 2 homology dimensions)
    """
    def __init__(self, model_path=None) -> None: # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()
        
        if model_path:
             # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    
    def init_network(self): # create model for TDA imput data in the form of persistance images generated from 17 differrent filtration x 2 homology dimensions = 34 channels
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 34)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(),  
              metrics=['accuracy'])


class TDA_PI42_Model():
    """

    CNN model that works on persistance images generated from TDA pipeline.
    Input data should be 28x28 pixel images with 34 channels (hence name)
    (17 different filtration x 2 homology dimensions)
    """

    def __init__(self,
                 model_path=None) -> None:  # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()

        if model_path:
            # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    def init_network(
            self):  # create model for TDA imput data in the form of persistance images generated from 17 differrent filtration x 2 homology dimensions = 34 channels
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 42)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


class VECTOR_STITCHING_PI_Model_34():
    """
    
    CNN model that works on persistance images generated from TDA pipeline stitched with raw-pixel images. 
    Input data should be 56x28 pixel images with 34 channels (for mnist, for other data model should be corrected accordingly)
    34 channels = 17 different filtration x 2 homology dimensions
    """

    def __init__(self, model_path=None) -> None: # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()
        
        if model_path:
             # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    
    def init_network(self): # create model for TDA imput data in the form of persistance images generated from 17 differrent filtration x 2 homology dimensions = 34 channels
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(56, 28, 34)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(),  
              metrics=['accuracy'])


class VECTOR_STITCHING_PI_Model_42():
    """

    CNN model that works on persistance images generated from TDA pipeline stitched with raw-pixel images.
    Input data should be 56x28 pixel images with 34 channels (for mnist, for other data model should be corrected accordingly)
    34 channels = 17 different filtration x 2 homology dimensions
    """

    def __init__(self,
                 model_path=None) -> None:  # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()

        if model_path:
            # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    def init_network(
            self):  # create model for TDA imput data in the form of persistance images generated from 17 differrent filtration x 2 homology dimensions = 34 channels
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(56, 28, 42)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


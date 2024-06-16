"""
Tytuł pracy licencjackiej: Połączenie topologicznej analizy danych z konwolucyjnymi sieciami neuronowymi w rozpoznawaniu obrazów.


Autorzy: Wojciech Jaworek, Adrian Stolarek
Data stworzenia: luty - czerwiec 2024

Opis: 
Plik zawiera modele CNN dla danych surowych.
"""




from tensorflow.keras import models, layers, losses

class Raw_Model():
    """
    CNN Model for raw input data (28x28 pixel images).
    """

    def __init__(self, model_path=None) -> None: # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()

        if model_path:
            # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    
    def init_network(self): # model for raw data input (28x28 pixel images)
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                    loss=losses.SparseCategoricalCrossentropy(),  
                    metrics=['accuracy'])
        






class Dummy_Model():
    """
    Dummy fully dense model for raw input data (28x28 pixel images).
    """

    def __init__(self, model_path=None) -> None: # if path to existing model is provided, load it, else create a new model
        self.model = models.Sequential()

        if model_path:
             # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    
    def init_network(self): # model for raw data input (28x28 pixel images)
        self.model.add(layers.Flatten(input_shape=(28, 28)))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))

        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                        loss=losses.SparseCategoricalCrossentropy(),  
                        metrics=['accuracy'])


"""
Tytuł pracy licencjackiej: Połączenie topologicznej analizy danych z konwolucyjnymi sieciami neuronowymi w rozpoznawaniu obrazów.


Autorzy: Wojciech Jaworek, Adrian Stolarek
Data stworzenia: luty - czerwiec 2024

Opis: 
Plik zawiera implementacje pipeline'ów do przetwarzania obrazów i generowania obrazów persystencji.
Zawiera również klasy pomocnicze do łączenia obrazów persystencji z obrazami surowymi (vector-stitching).
"""


from sklearn import set_config 

from sklearn.pipeline import make_pipeline, make_union, FeatureUnion, Pipeline
from gtda.diagrams import Scaler, PersistenceImage

from gtda.images import HeightFiltration, Binarizer, RadialFiltration
from gtda.images import DensityFiltration, DilationFiltration, ErosionFiltration, SignedDistanceFiltration
from gtda.homology import CubicalPersistence
import numpy as np


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin



def weight_func(x):
    """
    The identity function. Can be replaced with any other weight function.
    See persistence images article for more details.
    """
    return x




def TDA_PI34_Pipeline(dir_list=None, cen_list=None, binarizer_threshold=0.4, bins=28, sig=0.15):
    """
    Generates persistance images of size 28x28, with 34 (default, if parameters are provided this might be different) channels.

    Function creates and returns pipeline object from sklean.pipeline module.
    """
    if dir_list:
        direction_list = dir_list
        center_list = cen_list
    else:
        direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]] 
        center_list = [ [13, 6], [6, 13], [13, 13], [20, 13], [13, 20], [6, 6], [6, 20], [20, 6], [20, 20], ] 

    # list of filtrations
    filtration_list = (
        [ HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list ] +
        [ RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
    )

    # PD pipeline 
    diagram_steps = [
        [ Binarizer(threshold=binarizer_threshold, n_jobs=-1), filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1), ]
        for filtration in filtration_list
    ]


    # dont really need feature union 
    feature_union = make_union(
        PersistenceImage(sigma=sig, n_bins=bins, n_jobs=-1, weight_function=weight_func) # or heat kernel, or possibly any other (but rational and well-fitting to model) vector representation of the diagram
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
        n_jobs=-1
    )



    return tda_union

def TDA_PI42_Pipeline(dir_list=None, cen_list=None, binarizer_threshold=0.5, bins=28, sig=0.15):
    """
    Generates persistance images of size 28x28, with 34 (default, if parameters are provided this might be different) channels.

    Function creates and returns pipeline object from sklean.pipeline module.
    """
    if dir_list:
        direction_list = dir_list
        center_list = cen_list
    else:
        direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        center_list = [ [13, 6], [6, 13], [13, 13], [20, 13], [13, 20], [6, 6], [6, 20], [20, 6], [20, 20], ]

    # list of filtrations
    filtration_list = (
        [ HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list ] +
        [ RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list] +
        [ DensityFiltration(n_jobs=-1), DilationFiltration(n_jobs=-1), ErosionFiltration(n_jobs=-1), SignedDistanceFiltration(n_jobs=-1) ]
    )

    # PD pipeline
    diagram_steps = [
        [ Binarizer(threshold=binarizer_threshold, n_jobs=-1), filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1), ]
        for filtration in filtration_list
    ]


    # feature_union, again
    feature_union = make_union(
        PersistenceImage(sigma=sig, n_bins=bins, n_jobs=-1) 
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
        n_jobs=-1
    )



    return tda_union

class ImageScalerAndFlattener(BaseEstimator, TransformerMixin):
    """
    Helper class for vector stitching pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaled_and_flattened_images = []
        for image in X:
            min_val, max_val = image.min(), image.max()
            scaled_image = (image - min_val) / (max_val - min_val) if max_val - min_val > 0 else image
            scaled_and_flattened_images.append(scaled_image.flatten())
        return np.array(scaled_and_flattened_images).reshape(-1, 1, 784)

class CombineTDAWithRawImages_34(BaseEstimator, TransformerMixin):
    """
    Helper class for vector stitching pipeline.
    """

    def __init__(self, tda_pipeline, raw_image_pipeline):
        self.tda_pipeline = tda_pipeline
        self.raw_image_pipeline = raw_image_pipeline

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tda_features = self.tda_pipeline.fit_transform(X).reshape(-1, 34, 28, 28) # transform method changed to fit_transform in order to initialize binarizer automatically
                
        
        # normalize images to [0,1] scale 
        for i in range(tda_features.shape[0]):
            for j in range(tda_features.shape[1]):
                min_val, max_val = tda_features[i,j].min(), tda_features[i,j].max()
                tda_features[i,j] = (tda_features[i,j] - min_val) / (max_val - min_val) if max_val - min_val > 0 else tda_features[i,j]
        


        raw_images = self.raw_image_pipeline.transform(X).reshape(-1, 28, 28) # .T to transpose into correct orientation
        
        for i in range(raw_images.shape[0]):
            raw_images[i] = raw_images[i].T


        raw_images_expanded = np.expand_dims(raw_images, axis=1).repeat(34, axis=1)

        combined_features = np.concatenate((tda_features, raw_images_expanded), axis=3)  

        return combined_features


class CombineTDAWithRawImages_42(BaseEstimator, TransformerMixin):
    """
    Helper class for vector stitching pipeline.
    """

    def __init__(self, tda_pipeline, raw_image_pipeline):
        self.tda_pipeline = tda_pipeline
        self.raw_image_pipeline = raw_image_pipeline

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tda_features = self.tda_pipeline.fit_transform(X).reshape(-1, 42, 28, 28)  # transform method changed to fit_transform in order to initialize binarizer automatically

        raw_images = self.raw_image_pipeline.transform(X).reshape(-1, 28, 28)


        raw_images_expanded = np.expand_dims(raw_images, axis=1).repeat(42, axis=1)

        combined_features = np.concatenate((tda_features, raw_images_expanded), axis=3)

        return combined_features

def VECTOR_STITCHING_PI_Pipeline_34(dir_list=None, cen_list=None, binarizer_threshold=0.5, bins=28, sig=0.15):
    """
    Returns pipeline that extracts topological features in form of persistence images and combines them with raw images.
    """

    direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]] 
    center_list = [ [13, 6], [6, 13], [13, 13], [20, 13], [13, 20], [6, 6], [6, 20], [20, 6], [20, 20], ] 

    # Raw data processing - Scaling and normalization
    raw_image_processing = Pipeline([('scaler_and_flattener', ImageScalerAndFlattener())])

    # Creating a list of all filtration
    filtration_list = (
        [ HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list ] +
        [ RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
    )

    # PD pipeline 
    diagram_steps = [
        [ Binarizer(threshold=0.4, n_jobs=-1), filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1), ]
        for filtration in filtration_list
    ]

    # feature_union 
    feature_union = make_union(
        PersistenceImage(sigma=.15, n_bins=28, n_jobs=-1, weight_function=weight_func)
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
        n_jobs=-1
    )

    combined_features = CombineTDAWithRawImages_34(tda_union, raw_image_processing)

    final_pipeline = Pipeline([('combine_features', combined_features)])

    return final_pipeline, tda_union

def VECTOR_STITCHING_PI_Pipeline_42(dir_list=None, cen_list=None, binarizer_threshold=0.5, bins=28, sig=0.15):
    """
    Returns pipeline that extracts topological features in form of persistence images and combines them with raw images.
    """

    direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    center_list = [ [13, 6], [6, 13], [13, 13], [20, 13], [13, 20], [6, 6], [6, 20], [20, 6], [20, 20], ]

    # Raw data processing - Scaling and normalization
    raw_image_processing = Pipeline([('scaler_and_flattener', ImageScalerAndFlattener())])

    # Creating a list of all filtration
    filtration_list = (
        [ HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list ] +
        [ RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list] +
        [DensityFiltration(n_jobs=-1), DilationFiltration(n_jobs=-1), ErosionFiltration(n_jobs=-1), SignedDistanceFiltration(n_jobs=-1)]
    )

    # PD
    diagram_steps = [
        [ Binarizer(threshold=0.4, n_jobs=-1), filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1), ]
        for filtration in filtration_list
    ]

    # feature_union
    feature_union = make_union(
        PersistenceImage(sigma=.3, n_bins=28, n_jobs=-1) 
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
        n_jobs=-1
    )

    combined_features = CombineTDAWithRawImages_42(tda_union, raw_image_processing)

    final_pipeline = Pipeline([('combine_features', combined_features)])

    return final_pipeline, tda_union


def display_pipeline(pipeline):
    """
    Function to display the pipeline object.
    """
    set_config(display='diagram')
    print(pipeline)

def transform_data(X_train, X_test_noisy_random, X_test):
    X_train_expanded = np.expand_dims(X_train, -1)
    X_test_noisy_random_expanded = np.expand_dims(X_test_noisy_random, -1)
    X_test_expanded = np.expand_dims(X_test, -1)
    return X_train_expanded, X_test_noisy_random_expanded, X_test_expanded
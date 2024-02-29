"""
TDA Pipelines that extract information from provided data in vectorized form.
Few examples are: 
-- Persistence Images
-- Heat Kernels
-- Amplitudes from various vectorizations
"""
from sklearn import set_config 

from sklearn.pipeline import make_pipeline, make_union 
from gtda.diagrams import PersistenceEntropy, Scaler, PersistenceImage, HeatKernel, Amplitude, BettiCurve, PersistenceLandscape, Silhouette

from gtda.images import HeightFiltration, Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
import numpy as np

def TDA_PI34_Pipeline(dir_list=None, cen_list=None, binarizer_threshold=0.5, bins=28, sig=0.15):
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

    # Creating a list of all filtration transformer
    filtration_list = (
        [ HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list ] +
        [ RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
    )

    # Creating the diagram generation pipeline 
    diagram_steps = [
        [ Binarizer(threshold=binarizer_threshold, n_jobs=-1), filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1), ]
        for filtration in filtration_list
    ]


    # feature_union 
    feature_union = make_union(
        PersistenceImage(sigma=sig, n_bins=bins, n_jobs=-1) # or heat kernel, or possibly any other (but rational and well-fitting to model) vector representation of the diagram
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
        n_jobs=-1
    )


    # TODO: transpose to 0 2 3 1 before returning data from pipeline


    return tda_union



def display_pipeline(pipeline):
    """
    Function to display the pipeline object.
    """
    set_config(display='diagram')
    print(pipeline)
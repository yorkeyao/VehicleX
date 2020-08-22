from __future__ import absolute_import

# from .ai_city_VID_test import AI_City
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .veri import VeRi
from .ai_city_sys import AI_City_Sys
from .vihicle_id import Vihicle_ID_Sys
# from .vihicle_id_veri import Vihicle_ID_VeRi
# from .alice_vehicle import alice


__factory = {
    'market1501': Market1501,
    'duke_tracking': DukeMTMC,
    'duke_reid': DukeMTMC,
    # 'aic_tracking': AI_City,
    # 'aic_reid': AI_City,
    'veri': VeRi,
    'aic_reid_sys': AI_City_Sys,
    'vihicle_id' : Vihicle_ID_Sys, 
    # 'VID_VeRi' : Vihicle_ID_VeRi,
    # 'alice_vehicle' : alice
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)

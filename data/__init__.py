from .cityscapes import CitySegmentation
from .mapillary import MapillarySegmentation
from .target import TargetDataLoader
from .synthia import SynthiaSegmentation
datasets = {
    'citys': CitySegmentation,
    'mapillary':MapillarySegmentation,
    'targetdataset':TargetDataLoader,
    'synthia':SynthiaSegmentation,
    }


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

from .ace import TrainerACE
from .base import TrainerBase

framework = {
	'ace':TrainerACE,
	'base':TrainerBase,
}

def get_framework(name,**kwargs):
	''' framework for semantic segmentation '''
	return framework[name.lower()](**kwargs)

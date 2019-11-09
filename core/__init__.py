from .loss import *
from .deeplabv3_plus import DeepLabV3Plus

models = {
	'deeplabv3plus':DeepLabV3Plus,
}

def get_segmentation_model(name,**kwargs):
	''' segmentation models '''
	return models[name.lower()](**kwargs)

losses = {
	'mixsoftmaxcrossentropyloss':MixSoftmaxCrossEntropyLoss,
	'mseloss':MSELoss,
	'ohemsoftmaxcrossentropyloss':OHEMSoftmaxCrossEntropyLoss,
	'ohemcrossentropy2d':OhemCrossEntropy2d,
	'mixsoftmaxcrossentropyohemloss':MixSoftmaxCrossEntropyOHEMLoss,
	'focalloss2d':FocalLoss2d,
	'bcewithlogitsloss2d':BCEWithLogitsLoss2d,
	'criterionkd':CriterionKD,
	'criterionsdcos':CriterionSDcos,
	'criterionkldivergence':CriterionKlDivergence,
}

def get_loss(name,**kwargs):
	''' awesome losses '''
	return losses[name.lower()](**kwargs)
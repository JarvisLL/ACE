import torch
import torch.nn as nn

def make_layer(cfg,batch_norm=False):
	layers = []
	in_channels = 512
	for v in cfg:
		if v == 'M':
			layers += [nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)]
		else:
			conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
			if batch_norm:
				layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
			else:
				layers += [conv2d,nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

cfgs = {
    'A':[512,512,'M',256,256,'M',128,'M',64,3]
    'B':[512,512,'M',256,256,'M'128,128,'M',64,64,3]
    'D':[512,512,512,'M',256,256,256,'M',128,128,'M',64,64,3]
	'E':[512,512,512,512,'M',256,256,256,256,'M',128,128,'M',64,64,3],
}

class TargetImgGenerator(nn.Module):
	def __init__(self,features,init_weights=True):
		self.features = features
		if init_weights:
			self._initialize_weights()

	def forward(self,x):
		x = self.features(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias,0)
				elif isinstance(m,nn.BatchNorm2d):
					nn.init.constant_(m.weight,1)
					nn.init.constant_(m.bias,0)
				elif isinstance(m,nn.Linear):
					nn.init.normal_(m.weight,0,0.01)
					nn.init.constant_(m.bias,0)

def _targetimggenerator(cfg,batch_norm,pretrained,**kwargs):
	if pretrained:
		kwargs['init_weights'] = False
	model = TargetImgGenerator(make_layer(cfgs[cfg],batch_norm=batch_norm),**kwargs)
	return model
    
def _tar_gen_vgg11(pretrained=False,**kwargs):
    return _targetimggenerator('A', False, pretrained, **kwargs)

def _tar_gen_vgg11_bn(pretrained=False,**kwargs):
    return _targetimggenerator('A', True, pretrained, **kwargs)

def _tar_gen_vgg13(pretrained=False,**kwargs):
    return _targetimggenerator('B', False, pretrained, **kwargs)

def _tar_gen_vgg13_bn(pretrained=False,**kwargs):
    return _targetimggenerator('B', True, pretrained, **kwargs)

def _tar_gen_vgg16(pretrained=False,**kwargs):
    return _targetimggenerator('C', True, pretrained, **kwargs)

def _tar_gen_vgg16_bn(pretrained=False,**kwargs):
    return _targetimggenerator('C', True, pretrained, **kwargs)

def _tar_gen_vgg19(pretrained=False,**kwargs):
	return _targetimggenerator('E', False, pretrained, **kwargs)

def _tar_gen_vgg19_bn(pretrained=False,**kwargs):
	return _targetimggenerator('E', True, pretrained, **kwargs)
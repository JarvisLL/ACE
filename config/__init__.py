from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .default import _C
from .ace_default import _ACE

def update_config(cfg, args):
	cfg.defrost()
	cfg.merge_from_file(args.cfg)
	cfg.merge_from_list(args.opts)
	cfg.freeze()

configs = {
	'base':_C,
	'ace':_ACE,
}

def get_config(name,**kwargs):
	''' get frameworks config '''
	return configs[name]
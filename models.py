
from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from mdeq_forward_backward import MDEQWrapper

class ModelF(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelF, self).__init__()
		self.cfg = cfg

	def forward(self, z_list: List[torch.Tensor], x_list: List[torch.Tensor], *args) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of function f, z[i+1]=f(z[i],x)
		length of z_list and x_list must be the same
		shape of tensors in z_list and x_list must be the same
		DO NOT use BatchNorm or anything that can leak information between samples here
		params:
			z_list: list of z tensors, the fixed point to be solved
			x_list: list of x tensors, input injection
		returns:
			list of z tensors, must be the same size and shape of the input z tensors
		"""
		raise NotImplemented

	def init_z_list(self, x_list: List[torch.Tensor]) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of initializing z tensor list, usually zeros tensors
		Notice here x input injection is provided as a parameter so you can initialize z with the correct shape on the correct device
		params:
			x_list: list of x tensors, input injection
		returns:
			list of initial z tensors
		"""
		raise NotImplemented

class ModelInjection(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelInjection, self).__init__()
		self.cfg = cfg

	def forward(self, x: torch.Tensor) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of input injection
		This module takes your input and extend it to a list of input injections for all scales
		(Notice you can change the input of this module to whatever you need)
		You can use BatchNorm here
		params:
			x: your single input
		returns:
			list of initial x tensors, input injection
		"""
		raise NotImplemented

class MDEQModelBackbone(nn.Module) :
	def __init__(self, cfg: dict[str: Any], f: ModelF, inject: ModelInjection):
		super(MDEQModelBackbone, self).__init__()
		self.parse_cfg(cfg)
		self.f = f
		self.inject = inject
		self.f_copy = copy.deepcopy(self.f)
			
		for param in self.f_copy.parameters():
			param.requires_grad_(False)
		self.deq = MDEQWrapper(self.f, self.f_copy)

	def parse_cfg(self, cfg: dict[str: Any]):
		self.num_layers = cfg.get('num_layers', 2)
		self.f_thres = cfg.get('f_thres', 24)
		self.b_thres = cfg.get('b_thres', 24)
		self.pretrain_steps = cfg.get('pretrain_steps', 0)

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) -> List[torch.Tensor] :
		f_thres = kwargs.get('f_thres', self.f_thres)
		b_thres = kwargs.get('b_thres', self.b_thres)
		writer = kwargs.get('writer', None)     # For tensorboard
		self.f_copy.load_state_dict(self.f.state_dict())
		x_list = self.inject(x)
		z_list = self.f.init_z_list(x_list)
		if 0 <= train_step < self.pretrain_steps:
			for layer_ind in range(self.num_layers):
				z_list = self.f(z_list, x_list)
		else:
			if train_step == self.pretrain_steps:
				torch.cuda.empty_cache()
				print('Switching to DEQ')
			z_list = self.deq(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)
		return z_list

class MDEQModelYourModel(MDEQModelBackbone) :
	def __init__(self, cfg: dict[str: Any]):
		super(MDEQModelYourModel, self).__init__(cfg, ModelF(cfg), ModelInjection(cfg))

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) :
		"""
		Fill here with your implementation of model head
		(Usually a classification head)
		params:
			x: your single input
		returns:
			Whatever your output want to be
		"""
		h: List[torch.Tensor] = super().forward(x, train_step, **kwargs)
		raise NotImplemented


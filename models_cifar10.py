
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
		self.c1 = nn.Sequential(
			nn.GroupNorm(16, 128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1)
		)
		self.c2 = nn.Sequential(
			nn.GroupNorm(16, 128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1)
		)

	def forward(self, z_list: List[torch.Tensor], x_list: List[torch.Tensor], *args) -> List[torch.Tensor] :
		h = z_list[0]
		h = self.c1(h)
		h = h + x_list[0]
		h = self.c2(h)
		return [h]

	def init_z_list(self, x_list: List[torch.Tensor]) -> List[torch.Tensor] :
		return [torch.zeros(a.size(0), 128, 32, 32, dtype=a.dtype, layout=a.layout, device=a.device) for a in x_list]

class ModelInjection(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelInjection, self).__init__()
		self.cfg = cfg
		self.c1 = nn.Sequential(
			nn.Conv2d(3, 128, 3, 1, 1),
			nn.GroupNorm(16, 128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
		)

	def forward(self, x: torch.Tensor) -> List[torch.Tensor] :
		return [self.c1(x)]

class MDEQModelBackbone(nn.Module) :
	def __init__(self, cfg: dict[str: Any], f: ModelF, inject: ModelInjection):
		super(MDEQModelBackbone, self).__init__()
		self.parse_cfg(cfg)
		self.f = f
		self.inject = inject
		self.f_copy = copy.deepcopy(self.f)
			
		for param in self.f.parameters():
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
		return self.deq(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)

class MDEQModelCifar10(MDEQModelBackbone) :
	def __init__(self, cfg: dict[str: Any], f: ModelF, inject: ModelInjection):
		super(MDEQModelCifar10, self).__init__(cfg, f, inject)
		self.cls = nn.Linear(128, 10)

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) :
		h = super().forward(x, train_step, **kwargs)[0]
		h = F.adaptive_avg_pool2d(h, 1).view(-1, 128)
		return self.cls(h)


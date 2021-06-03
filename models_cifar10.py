
from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from mdeq_forward_backward import MDEQWrapper

class BasicBlock(nn.Module) :
	def __init__(self, planes, dilation = 1) :
		super(BasicBlock, self).__init__()
		bottleneck_planes = planes // 2
		#self.bn1 = nn.GroupNorm(planes // 8, planes)
		self.conv_in = nn.Conv2d(planes, bottleneck_planes, kernel_size = 1)

		self.bn2 = nn.GroupNorm(bottleneck_planes // 8, bottleneck_planes)
		self.conv1 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size = 3, stride = 1, padding = dilation, groups = bottleneck_planes // 128 if bottleneck_planes >= 128 else 1, dilation = dilation)
		
		self.bn3 = nn.GroupNorm(bottleneck_planes // 8, bottleneck_planes)
		self.conv2 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size = 3, stride = 1, padding = dilation, groups = bottleneck_planes // 128 if bottleneck_planes >= 128 else 1, dilation = dilation)
		
		self.bn4 = nn.GroupNorm(bottleneck_planes // 8, bottleneck_planes)
		self.conv_out = nn.Conv2d(bottleneck_planes, planes, kernel_size = 1)

		self.bn5 = nn.GroupNorm(planes // 8, planes)

	def forward(self, x, injection = None):
		shortcut = x
		if injection is not None :
			x = x + injection
		x = self.conv_in(x)
		x = self.conv1(F.relu(self.bn2(x)))
		x = self.conv2(F.relu(self.bn3(x)))
		x = self.conv_out(F.relu(self.bn4(x)))
		return self.bn5(shortcut + x)

class StandardMultiscaleFusion(nn.Module) :
	def __init__(self, channels, add_stage = False, add_stage_channels = 0) :
		super(StandardMultiscaleFusion, self).__init__()
		self.channels = channels
		self.add_stage = add_stage
		self.add_stage_channels = add_stage_channels
		self.out_channels = self.channels + ([add_stage_channels] if add_stage else [])
		for out_branch, _ in enumerate(self.out_channels) :
			for in_branch, _ in enumerate(self.channels) :
				name = f'transition_{in_branch}_{out_branch}'
				if hasattr(self, name) :
					continue
				if in_branch < out_branch :
					# downscale
					mods = []
					for i in range(in_branch, out_branch) :
						mods += [
							nn.ReLU(),
							nn.Conv2d(channels[i], self.out_channels[i + 1], 4, 2, padding = 1),
							nn.GroupNorm(self.out_channels[i + 1] // 8, self.out_channels[i + 1]),
						]
					mods = nn.Sequential(*mods)
				elif in_branch > out_branch :
					# upscale
					scale_diff = 2 ** (in_branch - out_branch)
					mods = nn.Sequential(
						nn.UpsamplingBilinear2d(scale_factor = (scale_diff, scale_diff)),
						nn.ReLU(),
						nn.Conv2d(channels[in_branch], self.out_channels[out_branch], 1),
						nn.GroupNorm(self.out_channels[out_branch] // 8, self.out_channels[out_branch]),
					)
				elif out_branch == in_branch :
					mods = nn.Sequential()
				self.add_module(name, mods)
	def forward(self, branches) :
		h = branches
		out = []
		for out_branch, _ in enumerate(self.out_channels) :
			out.append(sum([getattr(self, f'transition_{in_branch}_{out_branch}')(h[in_branch]) for in_branch in range(len(self.channels))]))
		return out

class ParallelStage(nn.Module) :
	def __init__(self, block, channels: List[int], blocks: int = 4, use_dilation = True) :
		super(ParallelStage, self).__init__()
		self.blocks = blocks
		dilation_base = 2 if use_dilation else 1
		for i, ch in enumerate(channels) :
			for j in range(blocks) :
				self.add_module(f'stage_{i}_{j}', block(ch, dilation_base))
				if use_dilation :
					dilation_base *= 2

	def forward(self, branches, injection = None) :
		def run_stage(i, x, injection) :
			h = x
			for j in range(self.blocks) :
				h = getattr(self, f'stage_{i}_{j}')(h, injection)
			return h
		if injection is not None :
			return [run_stage(i, h, inj) for i, (h, inj) in enumerate(zip(branches, injection))]
		else :
			return [run_stage(i, h) for i, h in enumerate(branches)]

class ModelF(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelF, self).__init__()
		self.cfg = cfg
		self.stage = ParallelStage(BasicBlock, [32, 64, 128, 256], use_dilation = False)
		self.fusion = StandardMultiscaleFusion([32, 64, 128, 256])

	def forward(self, z_list: List[torch.Tensor], x_list: List[torch.Tensor], *args) -> List[torch.Tensor] :
		h = self.stage(z_list, x_list)
		return self.fusion(h)

	def init_z_list(self, x_list: List[torch.Tensor]) -> List[torch.Tensor] :
		return [torch.zeros(a.size(0), a.size(1), a.size(2), a.size(3), dtype=a.dtype, layout=a.layout, device=a.device) for a in x_list]

class ModelInjection(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelInjection, self).__init__()
		self.cfg = cfg
		self.c1 = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(32),
		)

	def forward(self, x: torch.Tensor) -> List[torch.Tensor] :
		return [
			self.c1(x),
			torch.zeros(x.size(0), 64, 16, 16, device = x.device, dtype = x.dtype),
			torch.zeros(x.size(0), 128, 8, 8, device = x.device, dtype = x.dtype),
			torch.zeros(x.size(0), 256, 4, 4, device = x.device, dtype = x.dtype)
			]

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

class MDEQModelCifar10(MDEQModelBackbone) :
	def __init__(self, cfg: dict[str: Any], f: ModelF, inject: ModelInjection):
		super(MDEQModelCifar10, self).__init__(cfg, f, inject)
		self.cls = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Linear(128, 10))

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) :
		h = super().forward(x, train_step, **kwargs)
		h = F.adaptive_avg_pool2d(h[-1], 1).view(-1, 256)
		return self.cls(h)


import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.fft as fft

from torchvision import transforms as T

# helper functions
def convert_coeff(x, eps=1e-6):
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amp, phase

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor
class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        TransposeLayer(-2, -1),  # 交换倒数第二和最后一维以实现batchnorm
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        TransposeLayer(-2, -1), #交换回来
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    # 找到中间层（不用）
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    #提取中间层（不用）
    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    #提取中间层（不用）
    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True


    @singleton('projector')
    def _get_projector(self, hidden):
        #print("hidden.shape",hidden.shape)
        _, seq_len, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden.device)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)
        #print("get_representation:", representation.shape)

        if not return_projection:
            return representation
        
        print("representation",representation.shape)
        
        x_freq = fft.fft(representation, axis=1)
        amp, phase = convert_coeff(x_freq) # amp[B, T//2+1, F]
        amp = F.normalize(amp, dim=-1, p=2)
        phase = F.normalize(phase, dim=-1, p=2)
        print("amp",amp.shape)
        print("phase",phase.shape)
        freq_features = torch.cat([amp, phase], dim=1)
        #freq_features_cat = torch.cat([freq_features, representation], dim=1)
        print("x_freq.real",x_freq.real.shape)
        print("x_freq.imag",x_freq.imag.shape)

        x = torch.cat([x_freq.real, x_freq.imag], dim=1)
        #x_cat = torch.cat([x, representation], dim=1)

        projector = self._get_projector(freq_features)
        projection = projector(freq_features)
        print("projection",projection.shape)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        #self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        v1,
        v2,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        image_one, image_two = v1, v2

        images = torch.cat((image_one, image_two), dim = 0)

        online_projections, _ = self.online_encoder(images)
        print("online_projections ",online_projections.shape)
        print('-----------------------')
        online_predictions = self.online_predictor(online_projections)

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim = 0)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

            target_projections, _ = target_encoder(images)
            print("target projections",target_projections.shape)
            print('******************')
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim = 0)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

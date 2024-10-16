import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import TensorDataset, DataLoader
from models.CoTs_encoder import CoSTEncoder as TSEncoder
from models.CoTs_encoder import CoSTEncoderSeason as SEEncoder
from scipy.special import softmax
from models.losses import *
from sklearn.metrics import log_loss
import tasks
import utils
import warnings
from models.basicaug import *
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from models.byol_pytorch.byol_pytorch import BYOL
import torch.fft as fft
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
LAEGE_NUM = 1e7

class SaURL:
    def __init__(
        self,
        input_dims,
        output_dims = 320,
        hidden_dims = 64,
        aug_depth = 3,
        device='cuda',
        lr=0.001,
        meta_lr = 0.01,
        aug_dim = 16,
        batch_size= 16,
        max_train_length = None,
        augmask_mode = 'all_true',
        eval_every_epoch = 20,
        eval_start_epoch = 200,
        aug_net_training = 'PRI',
        gamma_zeta = 0.05,
        hard_mask = True,
        gumbel_bias = 0.001,
        kernels= [1, 2, 4, 8, 16, 32, 64, 128],
        agu_channel = 1,
        reg_weight = 0.1,
        regular_weight = 0,
        noise_weight = 0.25,
        aug_distance_byol_weight = 1.25
    ):
        ''' 
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            meta_lr (int): The learning rate for meta learner.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
        '''
        
        super().__init__()
        print("------SaURL-TS------")
        self.reg_thres = 0.4
        self.device = device
        self.gumbel_bias = gumbel_bias
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.average_trend_strength = 0 #trend strength
        self.average_seasonality_strength = 0 #seasonal strangth
        self.average_cross_strength = 0 # cross strangth
        warnings.filterwarnings("ignore")
        self._net_time = TSEncoder(input_dims=input_dims, output_dims=output_dims,kernels=kernels,length=max_train_length).to(self.device)
        self._net_freq = TSEncoder(input_dims=input_dims, output_dims=output_dims,kernels=kernels,length=max_train_length).to(self.device)
        self._net_cross = TSEncoder(input_dims=input_dims, output_dims=output_dims,kernels=kernels,length=max_train_length).to(self.device)
        #self.adaptive_loss_fn = AdaptiveBYOLLoss().to(self.device)

        self.augsharenet = TSEncoder(input_dims=input_dims, output_dims=aug_dim,kernels=kernels,length=max_train_length,
                        hidden_dims=hidden_dims, depth=aug_depth,mask_mode=augmask_mode).to(self.device)
        self.factor_augnet = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device)########New  h(x)
        self.augmentation_projector = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        self.noise_augmentation_projector = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        self.augmentation_projector_byol = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x) #byol should have two different trans
        self.noise_augmentation_projector_byol = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        #------------------------------time data augmentation module

        self.augsharenet_fft = TSEncoder(input_dims=input_dims, output_dims=aug_dim,kernels=kernels,length=max_train_length,
                        hidden_dims=hidden_dims, depth=aug_depth,mask_mode=augmask_mode).to(self.device)
        self.factor_augnet_fft = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device)########New  h(x)
        self.augmentation_projector_fft = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        self.noise_augmentation_projector_fft = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        self.augmentation_projector_fft_byol = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x) #byol should have two different trans
        self.noise_augmentation_projector_fft_byol = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)
        #------------------------------fft data augmentation module

        self.byol_time = BYOL(self._net_time, image_size = input_dims)
        self.byol_fft = BYOL(self._net_freq, image_size = input_dims)
        self.byol_cross = BYOL(self._net_cross, image_size = input_dims)
        #------------------------------time/fft/cross byol


        self.net_time = torch.optim.swa_utils.AveragedModel(self._net_time)
        self.net_time.update_parameters(self._net_time)
        self.net_freq = torch.optim.swa_utils.AveragedModel(self._net_freq)
        self.net_freq.update_parameters(self._net_freq)
        self.net_cross = torch.optim.swa_utils.AveragedModel(self._net_cross)
        self.net_cross.update_parameters(self._net_cross)

        self.aug_net_training = aug_net_training
        self.hard_mask = hard_mask
        self.n_epochs = 0
        self.n_iters = 0

        self.meta_lr = meta_lr
        self.gamma_zeta = -gamma_zeta
        self.zeta = 1.0

        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.eval_every_epoch = eval_every_epoch
        self.eval_start_epoch = eval_start_epoch

        self.mean_noise = 0
        self.stds_noise = 0

        self.res = None

        # self.mmd_loss = MMDLoss()
  
    def get_dataloader(self,data,shuffle=False, drop_last=False):

        # pre_process to return data loader

        if self.max_train_length is not None:
            sections = data.shape[1] // self.max_train_length
            if sections >= 2:
                data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)

        data = data[~np.isnan(data).all(axis=2).all(axis=1)]
        data = np.nan_to_num(data)
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)),shuffle=shuffle, drop_last=drop_last)
        return data, dataset, loader

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + self.gumbel_bias  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()).to(sampling_weights.device) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        stretched_values = graph * (self.zeta - self.gamma_zeta) + self.gamma_zeta
        cliped = torch.clip(
            stretched_values,
            max=1.0,
            min=0.0)

        return cliped

    def calculate_trend_seasonality_strength(self, data, period=12, plot=False):
        B, T, F = data.shape
        assert B == 1, "Batch size should be 1"
        
        data = data.squeeze(0)  # Remove batch dimension
        trend_strengths = []
        seasonality_strengths = []
        
        for f in range(F):
            time_series = data[:, f]
            
            # STL 
            ts_series = pd.Series(time_series)
            stl = STL(ts_series, period=period, robust=True)
            result = stl.fit()

            # get trend/seasonal/resid component
            trend_component = torch.tensor(result.trend, dtype=torch.float32)
            seasonal_component = torch.tensor(result.seasonal, dtype=torch.float32)
            resid_component = torch.tensor(result.resid, dtype=torch.float32)
            
            # discard NaN 
            valid_idx = ~torch.isnan(trend_component)
            trend_component = trend_component[valid_idx]
            seasonal_component = seasonal_component[valid_idx]
            resid_component = resid_component[valid_idx]

            # cal trend strength
            var_resid = torch.var(resid_component)
            var_trend_plus_resid = torch.var(trend_component + resid_component)
            trend_strength = max(0, 1 - (var_resid / var_trend_plus_resid))

            # cal seasonal strength
            var_seasonal_plus_resid = torch.var(seasonal_component + resid_component)
            seasonal_strength = max(0, 1 - (var_resid / var_seasonal_plus_resid))
            
            trend_strengths.append(trend_strength)
            seasonality_strengths.append(seasonal_strength)
            
            if plot:
                plt.figure(figsize=(14, 10))
                plt.subplot(411)
                plt.plot(ts_series, label='Original')
                plt.legend(loc='upper left')
                plt.subplot(412)
                plt.plot(result.trend, label='Trend')
                plt.legend(loc='upper left')
                plt.subplot(413)
                plt.plot(result.seasonal, label='Seasonal')
                plt.legend(loc='upper left')
                plt.subplot(414)
                plt.plot(result.resid, label='Residual')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.suptitle(f'Feature {f + 1}')
                plt.subplots_adjust(top=0.9)
                plt.show()
        
        # cal avg
        average_trend_strength = torch.mean(torch.tensor(trend_strengths, dtype=torch.float32))
        average_seasonality_strength = torch.mean(torch.tensor(seasonality_strengths, dtype=torch.float32))
        
        return average_trend_strength.item(), average_seasonality_strength.item()

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def reconstruct_from_amp_phase(self, amp, phase):
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.complex(real, imag)
            
    def get_features(self, x, training = True, n_epochs=-1, mask=None, plot = False):

        embedding = self.augsharenet(x)
        weight_h = self.factor_augnet(embedding)
        weight_s = self.augmentation_projector(embedding)

        mask_h = self._sample_graph(weight_h,training= training)

        if self.hard_mask:
            hard_mask_h = (torch.sign(mask_h-0.5)+1)/2
            # print(hard_mask_h)
#             mask_h = (mask_h-hard_mask_h).detach()+hard_mask_h
            mask_h = (hard_mask_h - mask_h).detach()+mask_h

        axm = weight_s * mask_h * x  # main part of augmented x' main
        # print("x.shape", x.shape)
        # print("ax.shape", ax.shape)
        weight_n = self.noise_augmentation_projector(embedding) # x' noise weight

        mask_h_flipped = 1 - mask_h
        xn = mask_h_flipped * x 
        axn = weight_n * mask_h_flipped * x  # noise part of augmented x'

        ax = axm + axn

        if torch.isnan(ax).any() or torch.isnan(x).any():
            exit(1)

        # note: I add mask
        out1 = self._net_time(x,mask)  # representation of x
        out2 = self._net_time(axm,mask)  # representation of main part of augmented x'
        out3 = self._net_time(xn,mask) # representation of noise part of x
        out4 = self._net_time(axn,mask)  # representation of noise part of augmented x'
        return x, ax, out1, out2, out3, out4, weight_h, weight_s
    
    def get_features_byol(self, x, training = True, n_epochs=-1, mask=None, plot = False):

        # embedding = self.augsharenet_byol(x)
        # weight_h = self.factor_augnet_byol(embedding)
        # weight_s = self.augmentation_projector_byol(embedding)

        embedding = self.augsharenet(x)
        weight_h = self.factor_augnet(embedding)
        weight_s = self.augmentation_projector_byol(embedding)

        
        mask_h = self._sample_graph(weight_h,training= training)

        if self.hard_mask:
            hard_mask_h = (torch.sign(mask_h-0.5)+1)/2
            # print(hard_mask_h)
#             mask_h = (mask_h-hard_mask_h).detach()+hard_mask_h
            mask_h = (hard_mask_h - mask_h).detach()+mask_h
        axm = weight_s * mask_h * x  # augmented x'
        # print("x.shape", x.shape)
        # print("ax.shape", ax.shape)
       
        #操作无关部分
        weight_n = self.noise_augmentation_projector_byol(embedding)
        # 将0和1的位置颠倒
        mask_h_flipped = 1 - mask_h
        xn = mask_h_flipped * x 
        axn = weight_n * mask_h_flipped * x  # noise part of augmented x'

        ax = axm + axn

        # note: I add mask
        out1 = self._net_time(x,mask)  # representation of x
        out2 = self._net_time(axm,mask)  # representation of main part of augmented x'
        out3 = self._net_time(xn,mask) # representation of noise part of x
        out4 = self._net_time(axn,mask)  # representation of noise part of augmented x'
        return x, ax, out1, out2, out3, out4, weight_h, weight_s
    
    def get_features_fft(self, x, training = True, n_epochs=-1, mask=None, plot = False):

        #change x to freq domain
        x_freq = fft.rfft(x, axis=1)
        amp, phase = self.convert_coeff(x_freq) # rfft: amp[B, T//2+1, F] fft: amp[B, T, F]
        #amp_len = x.shape[1]//2+1


        embedding = self.augsharenet_fft(amp)
        weight_h = self.factor_augnet_fft(embedding)
        weight_s = self.augmentation_projector_fft(embedding)

        
        mask_h = self._sample_graph(weight_h,training= training)

        if self.hard_mask:
            hard_mask_h = (torch.sign(mask_h-0.5)+1)/2
            # print(hard_mask_h)
#             mask_h = (mask_h-hard_mask_h).detach()+hard_mask_h
            mask_h = (hard_mask_h - mask_h).detach()+mask_h

        axm_amp = weight_s * mask_h * amp  # augmented x'
        # print("x.shape", x.shape)
        # print("ax.shape", ax.shape)

        weight_n = self.noise_augmentation_projector_fft(embedding)

        mask_h_flipped = 1 - mask_h
        xn_amp = mask_h_flipped * amp 
        axn_amp = weight_n * mask_h_flipped * amp  # noise part of augmented x'

        ax_amp = axm_amp + axn_amp

        ax_freq = self.reconstruct_from_amp_phase(ax_amp, phase)
        ax = fft.irfft(ax_freq, n=x.shape[1], axis=1)

        # note: I add mask
        axm_freq = self.reconstruct_from_amp_phase(axm_amp, phase)
        xn_freq = self.reconstruct_from_amp_phase(xn_amp, phase)
        axn_freq = self.reconstruct_from_amp_phase(axn_amp, phase)

        axm = fft.irfft(axm_freq, n=x.shape[1], axis=1)
        xn = fft.irfft(xn_freq, n=x.shape[1], axis=1)
        axn = fft.irfft(axn_freq, n=x.shape[1], axis=1)

        out1 = self._net_freq(x,mask)  # representation of x
        out2 = self._net_freq(axm,mask)  # representation of main part of augmented x'
        out3 = self._net_freq(xn,mask) # representation of noise part of x
        out4 = self._net_freq(axn,mask)  # representation of noise part of augmented x'
        return x, ax, out1, out2, out3, out4, weight_h, weight_s
    
    def get_features_byol_fft(self, x, training = True, n_epochs=-1, mask=None, plot = False):

        #change x to freq domain
        x_freq = fft.rfft(x, axis=1)
        amp, phase = self.convert_coeff(x_freq) # rfft: amp[B, T//2+1, F] fft: amp[B, T, F]
        #amp_len = x.shape[1]//2+1


        embedding = self.augsharenet_fft(amp)
        weight_h = self.factor_augnet_fft(embedding)
        weight_s = self.augmentation_projector_fft_byol(embedding)

        
        mask_h = self._sample_graph(weight_h,training= training)

        if self.hard_mask:
            hard_mask_h = (torch.sign(mask_h-0.5)+1)/2
            # print(hard_mask_h)
#             mask_h = (mask_h-hard_mask_h).detach()+hard_mask_h
            mask_h = (hard_mask_h - mask_h).detach()+mask_h
        axm_amp = weight_s * mask_h * amp  # augmented x'
        # print("x.shape", x.shape)
        # print("ax.shape", ax.shape)
       

        weight_n = self.noise_augmentation_projector_fft_byol(embedding)

        mask_h_flipped = 1 - mask_h
        xn_amp = mask_h_flipped * amp 
        axn_amp = weight_n * mask_h_flipped * amp  # noise part of augmented x'

        ax_amp = axm_amp + axn_amp

        ax_freq = self.reconstruct_from_amp_phase(ax_amp, phase)
        ax = fft.irfft(ax_freq, n=x.shape[1], axis=1)

        if torch.isnan(ax).any() or torch.isnan(x).any():
            exit(1)

        # note: I add mask
        axm_freq = self.reconstruct_from_amp_phase(axm_amp, phase)
        xn_freq = self.reconstruct_from_amp_phase(xn_amp, phase)
        axn_freq = self.reconstruct_from_amp_phase(axn_amp, phase)

        axm = fft.irfft(axm_freq, n=x.shape[1], axis=1)
        xn = fft.irfft(xn_freq, n=x.shape[1], axis=1)
        axn = fft.irfft(axn_freq, n=x.shape[1], axis=1)



        out1 = self._net_freq(x,mask)  # representation of x
        out2 = self._net_freq(axm,mask)  # representation of main part of augmented x'
        out3 = self._net_freq(xn,mask) # representation of noise part of x
        out4 = self._net_freq(axn,mask)  # representation of noise part of augmented x'
        return x, ax, out1, out2, out3, out4, weight_h, weight_s

    def regular_consistency(self,weight):

        B,T,C = weight.shape

        # near
        select0 = torch.randint(1,T-2,[B,])
        left = select0 - 1
        right = select0 + 1
        select1 = torch.randint(1,T-2,[B,])
        #select1 = torch.randint(1,T-1,B)
        mask = torch.where((select1-select0)>1,torch.ones_like(select1),torch.zeros_like(select0)).to(weight.device)

        # near difference
        diff = mask.reshape(1,B,1)*torch.abs(weight[:,select0,:]-weight[:,select1,:]) + \
               torch.abs(weight[:,select0,:]-weight[:,left,:]) + \
               torch.abs(weight[:,select0,:]-weight[:,right,:]) + \
               (1-mask).reshape(1,B,1)*(1-torch.abs(weight[:,select0,:]-weight[:,select1,:]))

        return diff.mean()

    # calculate mutual information MI(v,x)
    def MI(self, data_loader):
        ori_training = self._net.training
        self._net.eval()
        cum_vx = 0
        zvs = []
        zxs = []
        size = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)
                outv, outx = self.get_features(x)
                vx_infonce_loss = L1out(outv, outx) * x.size(0)
                size +=x.size(0)

                zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1,2).squeeze(1)
                zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1,2).squeeze(1)

                cum_vx += vx_infonce_loss.item()
                zvs.append(zv.cpu().numpy())
                zxs.append(zx.cpu().numpy())

        MI_vx_loss = cum_vx / size
        zvs = np.concatenate(zvs,0)
        zxs = np.concatenate(zxs,0)

        if ori_training:
            self._net.train()
        return zvs,MI_vx_loss

    def fit(self, train_data, n_epochs=None, n_iters=None,task_type='classification', 
            valid_dataset=None, miverbose=None,
            # train_labels = None, split_number=8,
            # meta_epoch=2,meta_beta=1.0,verbose=False,beta=1.0,
            ratio_step=1,lcoal_weight=0.1,reg_weight = 0.001,
            regular_weight = 0.001,noise_weight = 0.25,aug_distance_byol_weight = 1.25,evalall =  False):
        '''
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            beta (float): trade-off between global and local contrastive.
            valid_dataset:  (train_data, train_label,test_data,test_label) for Classifier.
            miverbose (bool): Whether to print the information of meta-learner
            meta_epoch (int): meta-parameters are updated every meta_epoch epochs
            meta_beta (float): trade-off between high variety and high fidelity.
            task_type (str): downstream task
        Returns:
            crietira.
        '''

        # check the input formation
        assert train_data.ndim == 3

        do_valid = False if valid_dataset is None else True

        # default param for n_iters
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600
                
        self.average_trend_strength, self.average_seasonality_strength = self.calculate_trend_seasonality_strength(train_data[:,:,7:], plot = False)  
        self.average_cross_strength = (self.average_trend_strength + self.average_seasonality_strength) / 2
        #self.average_cross_strength = 0.1
        self.adaptive_loss_fn = AdaptiveBYOLLoss(init_alpha=self.average_seasonality_strength, init_beta=self.average_trend_strength, init_gamma=self.average_cross_strength).to(self.device)
        
        train_data,train_dataset,train_loader =  self.get_dataloader(train_data,shuffle=True,drop_last=True)
        #print("!!!!!", train_data.shape)
        train_data_tensor = torch.from_numpy(train_data)

        if task_type=='classification' and valid_dataset is not None:
            cls_train_data, cls_train_labels, cls_test_data, cls_test_labels = valid_dataset
            cls_train_data,cls_train_dataset,cls_train_loader = self.get_dataloader(cls_train_data,shuffle=False,drop_last=False)

        import itertools
        params_fft = itertools.chain(self.augsharenet_fft.parameters(),self.factor_augnet_fft.parameters(),self.augmentation_projector_fft.parameters(),self.augmentation_projector_fft_byol.parameters(),self.noise_augmentation_projector_fft.parameters(),self.noise_augmentation_projector_fft_byol.parameters())
        meta_optimizer_fft = torch.optim.AdamW(params_fft, lr=self.meta_lr)

        params = itertools.chain(self.augsharenet.parameters(),self.factor_augnet.parameters(),self.augmentation_projector.parameters(),self.augmentation_projector_byol.parameters(),self.noise_augmentation_projector.parameters(),self.noise_augmentation_projector_byol.parameters())
        meta_optimizer = torch.optim.AdamW(params, lr=self.meta_lr)

        params_byol = itertools.chain(self.byol_time.parameters(),self.byol_fft.parameters(),self.byol_cross.parameters(),self.adaptive_loss_fn.parameters())
        opt = torch.optim.AdamW(params_byol, lr=self.lr)

        self.t0 = 1.0
        self.t1 = 1.0

        acc_log = []
        vy_log = []
        vx_log = []
        loss_log = []

        mses = []
        maes = []

        def eval(final=False,s = True):
            #self._net.eval()
            self.byol_time.eval()
            self.byol_fft.eval()
            self.byol_cross.eval()
            self.factor_augnet.eval()
            self.factor_augnet_fft.eval()
            # try:
            if task_type == 'classification':
                out, eval_res = tasks.eval_classification(self, cls_train_data, cls_train_labels, cls_test_data,
                                                          cls_test_labels, eval_protocol='svm')
                clf = eval_res['clf']
                zvs, MI_vx_loss = self.MI(cls_train_loader)

                v_pred = softmax(clf.decision_function(zvs), -1)
                MI_vy_loss = log_loss(cls_train_labels, v_pred)
                v_acc = clf.score(zvs, cls_train_labels)

                vx_log.append(MI_vx_loss)
                vy_log.append(MI_vy_loss)

                acc_log.append(eval_res['acc'])

                if miverbose:
                    print('acc %.3f (max)vx %.3f (min)vy %.3f (max)vacc %.3f' % (
                    eval_res['acc'], MI_vx_loss, MI_vy_loss, v_acc))
            elif task_type == 'forecasting':
                if not final:
                    valid_dataset_during_train = valid_dataset[0],valid_dataset[1],valid_dataset[2],valid_dataset[3],valid_dataset[4],[valid_dataset[5][-1]],valid_dataset[6]
                    
                    out, eval_res = tasks.eval_forecasting(self, *valid_dataset_during_train, self.max_train_length, self.batch_size)
                else:
                    if s :
                        out, eval_res = tasks.eval_forecasting(self, *valid_dataset, self.max_train_length, self.batch_size)
                    else:
                        valid_dataset_during_train = valid_dataset[0], valid_dataset[1], valid_dataset[2], \
                                                     valid_dataset[3], valid_dataset[4], [valid_dataset[5][0]], \
                                                     valid_dataset[6]
                        out, eval_res = tasks.eval_forecasting(self, *valid_dataset_during_train, self.max_train_length, self.batch_size)

                res = eval_res['ours']
                self.res = eval_res['ours']
                mse = sum([res[t]['norm']['MSE'] for t in res]) / len(res)
                mae = sum([res[t]['norm']['MAE'] for t in res]) / len(res)
                mses.append(mse)
                maes.append(mae)
                for key in eval_res['ours']:
                    print(key,eval_res['ours'][key])
                print("avg.", mse, mae)
                print("avg. total", mse + mae)

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False


            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]

                x = x.to(self.device)
                if self.n_iters % ratio_step == 0 :
                    #self._net.eval()
                    self.byol_time.eval()
                    self.byol_fft.eval()
                    self.byol_cross.eval()
                    self.factor_augnet.train()
                    self.factor_augnet_fft.train()
                    if self.aug_net_training=='PRI':
                        meta_optimizer_fft.zero_grad()
                        x_,ax_,outx,outv,outxn,outvn,weight_h,weight_s = self.get_features_fft(x,mask='all_true')
                        #print(":xxxxxxxx", x.shape, x_.shape, ax_.shape, outx.shape)
                        vx_distance = mmdx(outx,outv+outvn) 
                        noise_distance = mmdx(outxn,outvn) 
                        regular = self.regular_consistency(weight_h) 
                        reg_loss = torch.nn.functional.relu(torch.sum(weight_h,dim=-1).mean()-self.reg_thres) 

                        x_byol,ax_byol,outx_byol,outv_byol,outxn_byol,outvn_byol,weight_h_byol,weight_s_byol = self.get_features_byol_fft(x,mask='all_true')
                        #print(":xxxxxxxx", x.shape, x_.shape, ax_.shape, outx.shape)
                        vx_distance_byol = mmdx(outx_byol,outv_byol+outvn_byol) 
                        noise_distance_byol = mmdx(outxn_byol,outvn_byol) 
                        regular_byol = self.regular_consistency(weight_h_byol) 
                        reg_loss_byol = torch.nn.functional.relu(torch.sum(weight_h_byol,dim=-1).mean()-self.reg_thres) 

                        aug_distance_byol = mmdx(outv+outvn,outv_byol+outvn_byol) 
                        
                        aloss = vx_distance - noise_weight * noise_distance + reg_weight * reg_loss + regular_weight * regular
                        aloss_byol = vx_distance_byol - noise_weight * noise_distance_byol + reg_weight * reg_loss_byol + regular_weight * regular_byol
                        aloss_all = aloss + aloss_byol - aug_distance_byol_weight * aug_distance_byol
                        aloss_all.backward()
                        meta_optimizer_fft.step()
                        #print("--------freq data augmentation done.----------")
                        # print("PRI aug loss ",vx_distance.item(),torch.sum(weight_h,dim=-1).mean().item())

                        meta_optimizer.zero_grad()
                        x_,ax_,outx,outv,outxn,outvn,weight_h,weight_s = self.get_features(x,mask='all_true')
                        #print(":xxxxxxxx", x.shape, x_.shape, ax_.shape, outx.shape)
                        vx_distance = mmdx(outx,outv+outvn) 
                        noise_distance = mmdx(outxn,outvn) 
                        regular = self.regular_consistency(weight_h) 
                        reg_loss = torch.nn.functional.relu(torch.sum(weight_h,dim=-1).mean()-self.reg_thres) 

                        x_byol,ax_byol,outx_byol,outv_byol,outxn_byol,outvn_byol,weight_h_byol,weight_s_byol = self.get_features_byol(x,mask='all_true')
                        #print(":xxxxxxxx", x.shape, x_.shape, ax_.shape, outx.shape)
                        vx_distance_byol = mmdx(outx_byol,outv_byol+outvn_byol) 
                        noise_distance_byol = mmdx(outxn_byol,outvn_byol) 
                        regular_byol = self.regular_consistency(weight_h_byol) 
                        reg_loss_byol = torch.nn.functional.relu(torch.sum(weight_h_byol,dim=-1).mean()-self.reg_thres) 

                        aug_distance_byol = mmdx(outv+outvn,outv_byol+outvn_byol) 

                        aloss = vx_distance - noise_weight * noise_distance + reg_weight * reg_loss + regular_weight * regular
                        aloss_byol = vx_distance_byol - noise_weight * noise_distance_byol + reg_weight * reg_loss_byol + regular_weight * regular_byol
                        aloss_all = aloss + aloss_byol - aug_distance_byol_weight * aug_distance_byol
                        aloss_all.backward()
                        meta_optimizer.step()
                        #print("--------time data augmentation done.----------")

                #self._net.train()
                self.byol_time.train()
                self.byol_fft.train()
                self.byol_cross.train()
                self.factor_augnet.eval()
                self.factor_augnet_fft.eval()

                # net_optimizer.zero_grad()
                # byol_optimizer.zero_grad()
                opt.zero_grad()
                plot = False
                    #plot = True
                #x_, ax_, outx, outv, _ = self.get_features(x, n_epochs=n_epochs, plot = plot)
                x_fft, ax_fft, outx_fft, outv_fft, outxn_fft, outvn_fft, __fft, s__fft  = self.get_features_fft(x, n_epochs=n_epochs, plot = plot)
                x_byol_fft, ax_byol_fft, outx_byol_fft, outv_byol_fft, outxn_byol_fft, outvn_byol_fft, _byol_fft, s_byol_fft = self.get_features_byol_fft(x, n_epochs=n_epochs, plot = plot)

                x_, ax_, outx, outv, outxn, outvn, _, s_  = self.get_features(x, n_epochs=n_epochs, plot = plot)
                x_byol, ax_byol, outx_byol, outv_byol, outxn_byol, outvn_byol, _byol, s_byol = self.get_features_byol(x, n_epochs=n_epochs, plot = plot)


                ax_freq = fft.rfft(ax_fft, axis=1)
                ax_amp, ax_phase = self.convert_coeff(ax_freq) # amp[B, T//2+1, F]
                ax_byol_freq = fft.rfft(ax_byol_fft, axis=1)
                ax_byol_amp, ax_byol_phase = self.convert_coeff(ax_byol_freq) # amp[B, T//2+1, F]

                ax_cat = torch.cat([ax_fft, ax_amp], dim=1)
                ax_byol_cat = torch.cat([ax_byol_fft, ax_byol_amp], dim=1)


                z1 = ax_fft
                z2 = ax_byol_fft

                byol_loss_fft = self.byol_fft(x,z1,z2)

                z3 = ax_
                z4 = ax_byol
                byol_loss_time = self.byol_time(x,z3,z4)

                x_freq = fft.fft(x_, axis=1)
                x_amp, phase = self.convert_coeff(x_freq) # amp[B, T//2+1, F]

                z5 = ax_
                z6 = ax_fft
                byol_loss_cross = self.byol_cross(x,z5,z6)

                byol_loss, self.weight_byol_domain = self.adaptive_loss_fn(byol_loss_fft, byol_loss_time, byol_loss_cross) 
                #byol_loss = self.average_seasonality_strength * byol_loss_fft + self.average_trend_strength * byol_loss_time + self.average_cross_strength * byol_loss_cross
                byol_loss =  byol_loss_fft +  byol_loss_time +  byol_loss_cross

                byol_loss.backward()  

                opt.step()

                self.byol_fft.update_moving_average() # 
                self.byol_time.update_moving_average() # 
                self.byol_cross.update_moving_average() # 

                self.net_freq.update_parameters(self._net_freq)
                self.net_time.update_parameters(self._net_time)
                self.net_cross.update_parameters(self._net_cross)

                cum_loss += byol_loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

            self.n_epochs += 1
            print("epoch ", self.n_epochs)
            if self.n_epochs%self.eval_every_epoch==0 and self.n_epochs > self.eval_start_epoch:
                # print("epoch ",self.n_epochs)
                if do_valid:
                    eval(evalall)

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)

        if do_valid:
            eval(True)

        if task_type == 'classification':
            return loss_log,acc_log,vx_log,vy_log
        else:
            return mses,maes,self.res

    def encode(self, data, mask=None, batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        Returns:
            repr: The representations for data.
        '''


        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                # x_freq = fft.fft(x, axis=1)
                # amp, phase = self.convert_coeff(x_freq) # amp[B, T//2+1, F]
                out = self.net(x.to(self.device, non_blocking=True), mask)
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2).cpu()
                out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def casual_encode(self, data, encoding_window=None, mask=None, sliding_length=None, sliding_padding=0,  batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
           
        '''
        casual = True
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape #(1, 17420, 14)
        #print("0. data.shape",data.shape)

        org_training = self.net_time.training
        self.net_time.eval()
        self.net_freq.eval()
        self.net_cross.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net_time.train(org_training)
        self.net_freq.train(org_training)
        self.net_cross.train(org_training)
        #print("4. casual_encode_output", output.shape)
        return output.numpy()

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):

        out_time = self.net_time(x.to(self.device, non_blocking=True), mask)
        #print("out_time dim", out_time.shape)
        out_freq = self.net_freq(x.to(self.device, non_blocking=True), mask)        
        #print("out_freq dim", out_freq.shape)
        out_cross = self.net_cross(x.to(self.device, non_blocking=True), mask)
        # print("out dim", out.shape)
        # print("encoding_window", encoding_window)
        #print("2. _eval_with_pooling_net", out.shape)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = torch.cat((self.average_trend_strength * out_time, self.average_seasonality_strength * out_freq, self.average_cross_strength * out_cross), dim=-1) 
                out = out[:, slicing]

        return out.cpu()
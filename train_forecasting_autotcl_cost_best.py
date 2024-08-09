import numpy as np
import argparse
import itertools
import time
import datetime
import data_load as datautils #
from utils import init_dl_program,dict2class #
from SACL_TS_Best import AutoTCL 
from models.augclass import * 
from config import * 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str, default='ETTh1', help='The dataset name') # ETTh1,ETTh2,ETTm1,electricity, WTH,Lora
parser.add_argument('--load_default',type=bool, default=True, help='load default setting for dataset') #
parser.add_argument('--archive', type=str, default='forecast_csv', help='forecast_csv_univar or forecast_csv -->univar or multivar')


parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference')
parser.add_argument('--seed', type=int, default=4, help='seed')
parser.add_argument('--max_threads', type=int, default=12, help='threads')
parser.add_argument('--eval', type=bool, default=True, help='do eval')


parser.add_argument('--batch-size', type=int, default=8, help='The batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='The embedding learning rate')
parser.add_argument('--meta_lr', type=float, default=0.012, help='The augmentation learning rate')
parser.add_argument('--mask_mode', type=str, default='mask_last', help='Do not use noise add model for embedding network') #?
parser.add_argument('--augmask_mode', type=str, default='binomial', help='noise add model for argumentation network') #


parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension')
parser.add_argument('--hidden_dims', type=int, default=64, help='The hidden dimension for embedding')
parser.add_argument('--aug_dim', type=int, default=16, help='The hidden dimension for argumentation')
parser.add_argument('--depth', type=int, default=10, help='Depths of embedding network') 
parser.add_argument('--aug_depth', type=int, default=1, help='Depths of argumentation network')

parser.add_argument('--max_train_length', type=int, default=1024, help='The training length') #?
parser.add_argument('--iters', type=int, default=4000, help='The training iters')
parser.add_argument('--epochs', type=int, default=400, help='epochs')

parser.add_argument('--bias_init', type=float, default=0.0, help='')
parser.add_argument('--local_weight', type=float, default=0.72, help='The weight of local contrastive loss')
parser.add_argument('--reg_weight', type=float, default=0.2, help='The weight of H(x)')
parser.add_argument('--regular_weight', type=float, default=0.002, help='The weight of Regularization')
parser.add_argument('--noise_weight', type=float, default=0.25, help='The weight of add noise')
parser.add_argument('--aug_distance_byol_weight', type=float, default=1.25, help='The weight of distance of two augs of byol')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of embedding network')
parser.add_argument('--augdropout', type=float, default=0.1, help='Dropout of argumentation network')

parser.add_argument('--ratio_step', type=int, default=1, help='M')
parser.add_argument('--gamma_zeta', type=float, default=0.005, help=' ')
parser.add_argument('--hard_mask', type=bool, default=True, help=' ')
parser.add_argument('--gumbel_bias', type=float, default=0.4, help=' ')

paras = parser.parse_args()
if paras.load_default:
    params = merege_config(paras,paras.dataset,paras.archive =='forecast_csv_univar')
else:
    params = paras
args = params # dict2class(**params)
print("args", args)
# args = dict2class(**params)
device = init_dl_program(args.gpu, seed=args.seed, max_threads=None )

if args.dataset == "lora":
    task_type = 'forecasting'
    if args.archive == 'forecast_csv':
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv_lora()
    else:
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv_lora(True)
    train_data = data[:, train_slice]

elif args.archive == 'forecast_csv':
    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
    train_data = data[:, train_slice]
    #print("data.shape",data.shape) #(1, 35064, 19)
    #print("train_data.shape",train_data.shape) #(1, 21038, 19)

elif args.archive == 'forecast_csv_univar':
    task_type = 'forecasting'
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
    train_data = data[:, train_slice]

valid_dataset = (data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols) #

if train_data.shape[0] == 1: #
    train_slice_number = int(train_data.shape[1] / args.max_train_length) #
    if train_slice_number < args.batch_size:
        args.batch_size = train_slice_number
else:
    if train_data.shape[0] < args.batch_size:
        args.batch_size = train_data.shape[0]

config = dict(
    batch_size=args.batch_size,
    lr=args.lr,
    meta_lr = args.meta_lr,
    output_dims=args.repr_dims,
    max_train_length=args.max_train_length,
    input_dims=train_data.shape[-1],
    device=device,
    # depth =  args.depth,
    hidden_dims = args.hidden_dims,
    # dropout = args.dropout,
    # augdropout = args.augdropout,
    # mask_mode = args.mask_mode,
    augmask_mode = args.augmask_mode,
    # bias_init = args.bias_init,
    gamma_zeta = args.gamma_zeta,
    aug_dim = args.aug_dim,
    hard_mask = bool(args.hard_mask),
    gumbel_bias = args.gumbel_bias
)

t = time.time()

print("model")
model = AutoTCL(
    eval_every_epoch =100,
    eval_start_epoch =100,
    agu_channel = data.shape[-1],
    **config
)

print("fit")
print(args.augmask_mode)
print(task_type)
print(valid_dataset[1])
print(train_data.shape)
res = model.fit(train_data,
     task_type = task_type,
     n_epochs=args.epochs,
     n_iters=args.iters,
     miverbose=True,
     valid_dataset = valid_dataset,
    ratio_step= args.ratio_step,
    lcoal_weight = args.local_weight,
    reg_weight = args.reg_weight,
    regular_weight = args.regular_weight,
    noise_weight = args.noise_weight,
    aug_distance_byol_weight = args.aug_distance_byol_weight,
    evalall =  True
    )

mse, mae, ress = res
mi_info = 'mse %.5f  mae%.5f' % (mse[-1], mae[-1])

print(mi_info)

t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
print("Finished.")


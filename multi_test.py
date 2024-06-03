# System / Python
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity,normalized_root_mse
# PyTorch
import torch
from torch.utils.data.dataloader import DataLoader
# Custom
from net import ParallelNetwork as Network
from mri_tools import *
from torch.utils.tensorboard import SummaryWriter
from HPjay_mutilcontrast_normal import IXIData as Dataset
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to model
parser.add_argument('--num-layers', type=int, default=1, help='number of iterations')#swin=1,using MCsingle
parser.add_argument('--in-channels', type=int, default=1, help='number of model input channels')
parser.add_argument('--out-channels', type=int, default=1, help='number of model output channels')
# batch size, num workers
parser.add_argument('--batch-size', type=int, default=1, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
# parameters related to test data
parser.add_argument('--T2test-path', type=str, default='', help='path of test data')
parser.add_argument('--PDtest-path', type=str, default='', help='path of test data')
parser.add_argument('--u-mask-path', type=str, default='', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='', help='selection mask in down network')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.1, help='sampling rate of test data')#D:\SSL-MRI-reconstruction\checkpoints\random_sampling\8x
# others
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')#random_sampling/8x/

def validate(args):
    torch.cuda.set_device(0)
    test_set = Dataset(args.T2test_path, args.PDtest_path, args.u_mask_path,args.s_mask_up_path,args.s_mask_down_path,args.test_sample_rate)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,drop_last=True)
    model = Network(rank=0,num_layers=args.num_layers)
    model_path = os.path.join(args.model_save_path, '')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(0))
    model.load_state_dict(checkpoint['model'])
    print('The model is loaded.')
    model = model.network.cuda(0)
    print('Now testing {}.'.format(args.exp_name))
    model.eval()
    # model_another.eval()
    with torch.no_grad():
        average_psnr, average_ssim,average_nmse, average_psnr_zerof, average_ssim_zerof,average_nmse_zerof ,average_time, total_num = 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0, 0
        t = tqdm(test_loader, desc='testing', total=int(len(test_loader)))
        batch_psnr, batch_ssim, batch_nrmse, batch_psnr_zerof, batch_ssim_zerof, batch_nrmse_zerof = [], [], [], [], [], []
        for iter_num, data_batch in enumerate(t):
            T2label = data_batch[0].to(0, non_blocking=True)  # (1,256,256,1)
            PDlabel = data_batch[1].to(0, non_blocking=True)  # (1,256,256,1)
            mask_under = data_batch[2].to(0, non_blocking=True)

            PDlabel =rAtA(PDlabel,mask_under)
            under_img = rAtA(T2label, mask_under)

            # inference
            start_time = time.time()
            output=model(under_img.permute(0, 3, 1, 2).contiguous(),mask_under,PDlabel.permute(0, 3, 1, 2).contiguous())#PDlabel.permute(0, 3, 1, 2).contiguous()
            output=output.permute(0, 2, 3, 1).contiguous()

            under_img=torch.abs(under_img)

            infer_time = time.time() - start_time
            average_time += infer_time
            under_img_np, output_np, label_np = under_img.detach().cpu().numpy(), output.detach().cpu().numpy(), T2label.float().detach().cpu().numpy()
            total_num += under_img_np.shape[0]


            for i in range(under_img_np.shape[0]):
                under_slice, output_slice, label_slice = under_img_np[i].squeeze(), output_np[i].squeeze(), label_np[i].squeeze()
                psnr = peak_signal_noise_ratio(label_slice, output_slice,data_range=label_slice.max())
                psnr_zerof = peak_signal_noise_ratio(label_slice, under_slice,data_range=label_slice.max())
                ssim = structural_similarity(label_slice, output_slice,data_range=label_slice.max())
                ssim_zerof = structural_similarity(label_slice, under_slice,data_range=label_slice.max())
                nrmse=normalized_root_mse(label_slice,output_slice)
                nrmse_zerof=normalized_root_mse(label_slice,under_slice)

                batch_psnr.append(psnr)
                batch_ssim.append(ssim)
                batch_nrmse.append(nrmse)
                batch_psnr_zerof.append(psnr_zerof)
                batch_ssim_zerof.append(ssim_zerof)
                batch_nrmse_zerof.append(nrmse_zerof)

        average_psnr=np.mean(batch_psnr)
        std_psnr=np.std(batch_psnr,ddof=1)
        average_ssim=np.mean(batch_ssim)
        std_ssim=np.std(batch_ssim,ddof=1)
        average_nrmse=np.mean(batch_nrmse)
        std_nrmse=np.std(batch_nrmse,ddof=1)
        average_psnr_zerof=np.mean(batch_psnr_zerof)
        std_psnr_zerof=np.std(batch_psnr_zerof,ddof=1)
        average_ssim_zerof=np.mean(batch_ssim_zerof)
        std_ssim_zerof=np.std(batch_ssim_zerof,ddof=1)
        average_nrmse_zerof=np.mean(batch_nrmse_zerof)
        std_nrmse_zerof=np.std(batch_nrmse_zerof,ddof=1)
    print('average_time:{:.3f}s\tzerof_psnr:{:.3f}\tzerof_ssim:{:.4f}\tzerof_nrmse:{:.3f}\ttest_psnr:{:.3f}\ttest_ssim:{:.4f}\ttest_nrmse:{:.3f}'.format(
        average_time, average_psnr_zerof, average_ssim_zerof,average_nrmse_zerof,average_psnr, average_ssim,average_nrmse))
    print('std_psnr_zerof:{:.3f}\tstd_ssim_zerof:{:.3f}\tstd_nrmse_zerof:{:.3f}\tstd_psnr:{:.3f}\tstd_ssim:{:.3f}\tstd_nrmse:{:.3f}'.format(
            std_psnr_zerof, std_ssim_zerof, std_nrmse_zerof, std_psnr, std_ssim, std_nrmse))

if __name__ == '__main__':
    args_ = parser.parse_args()
    validate(args_)


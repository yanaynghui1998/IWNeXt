# import random
import pathlib
import scipy.io as sio
import numpy as np
# import nibabel as nib
# import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one
import datatable as dt
from torch.utils.data import DataLoader
from tqdm import tqdm
# # from nilearn.image import resample_img
# # from ssdu_masks_create import ssdu_masks
from mri_tools import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from wavelet_transform import IWT,DWT
# from wavelet_fusion import localenergy,fuseCoeff
class IXIData(Dataset):
    def __init__(self, T2data_path,T1data_path,u_mask_path,s_mask_up_path,s_mask_down_path, sample_rate):
        super(IXIData, self).__init__()
        self.T2data_path = T2data_path
        self.T1data_path = T1data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate

        self.T2examples = []
        self.T1examples = []
        T2files = list(pathlib.Path(self.T2data_path).iterdir())
        T1files = list(pathlib.Path(self.T1data_path).iterdir())
        # The middle slices have more detailed information, so it is more difficult to reconstruct.
        start_id, end_id = 0,1#30,100
        for file in sorted(T2files):
            self.T2examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        for file in sorted(T1files):
            self.T1examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        if self.sample_rate < 1:
            # random.shuffle(self.examples)
            T2num_examples = round(len(self.T2examples) * self.sample_rate)
            self.T2examples = self.T2examples[:T2num_examples]
            T1num_examples = round(len(self.T1examples) * self.sample_rate)
            self.T1examples = self.T1examples[:T1num_examples]
        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])#maskRS1
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])
        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.T2examples)

    def __getitem__(self, item):
        T2file, T2slice_id = self.T2examples[item]
        T1file, T1slice_id = self.T1examples[item]
        T2data = dt.fread(T2file)
        T2data=np.array(T2data)
        T1data = dt.fread(T1file)
        T1data = np.array(T1data)
        # T2label = T2data.dataobj[T2slice_id,:,:]
        # T1label = T1data.dataobj[T2slice_id,:,:]
        t2label = normalize_zero_to_one(T2data, eps=1e-6)
        t1label = normalize_zero_to_one(T1data, eps=1e-6)
        t2label = torch.from_numpy(t2label).unsqueeze(-1).float()
        t1label = torch.from_numpy(t1label).unsqueeze(-1).float()
        return t2label,t1label,self.mask_under, self.mask_net_up, self.mask_net_down,T2file.name, T1file.name


# import numpy as np
# import sigpy as sp
# import sigpy.mri as mr
# import sigpy.plot as pl
#
# u_mask_path="D:/MUlSSL-MRI-reconstruction/mask1/undersampling_mask/mask_6.00x_acs24.mat"
# mask_up_path= "D:/MUlSSL-MRI-reconstruction/mask1/selecting_mask/mask_2.00x_acs16.mat"
# mask_down_path= "D:/MUlSSL-MRI-reconstruction/mask1/selecting_mask/mask_2.50x_acs16.mat"
# train_loader=IXIData(T2data_path='E:/HCPjay/T2/train',T1data_path='E:/HCPjay/T1/train',u_mask_path=u_mask_path,s_mask_up_path=mask_up_path,s_mask_down_path=mask_down_path, sample_rate=1)
# train_loaders= DataLoader(dataset=train_loader, batch_size=1)
# t = tqdm(train_loaders, desc='train' + 'ing', total=int(len(train_loader)))
# # # # # # #
# batch_psnr, batch_ssim, batch_psnr_zerof, batch_ssim_zerof = [],  [], [], []
# for batch in enumerate(t):#batch[0]表示所有矩阵的的总和，int类型，batch[1]是一个列表，有四个元素
#         T2label=batch[1][0]
#         T1label=batch[1][1]
#         mask_under =batch[1][2]
#         under_img = rAtA(T2label, mask_under)
#         under_img=torch.abs(under_img)
#         under_kspace_w = rA(T2label, mask_under)
#         under_kspace_w=torch.view_as_complex(under_kspace_w)
#         # print(under_kspace_w.shape)
#         under_kspace_w_np=under_kspace_w.numpy()
#         # img_rss = np.sum(np.fft.ifftshift(sp.ifft(under_kspace_w_np, axes=(-2, -1))) ** 2, axis=0) ** 0.5
#         # pl.ImagePlot(img_rss, title='Root-sum-of-squares Zero-filled')
#         # pl.ImagePlot(under_kspace_w_np, mode='l', z=0, title='Log magnitude of k-space')
#         mps = mr.app.EspiritCalib(under_kspace_w_np).run()
#         mps=np.fft.ifftshift(mps)
#         # pl.ImagePlot(mps, z=0, title='Sensitivity Maps Estimated by ESPIRiT')
#         lamda = 0.005
#         # img_l1wav = mr.app.L1WaveletRecon(under_kspace_w_np, mps, lamda).run()
#         # img_l1wav=np.fft.ifftshift(img_l1wav)
#         # pl.ImagePlot(img_l1wav, title='L1 Wavelet Regularized Reconstruction')
#
#         img_tv = mr.app.TotalVariationRecon(under_kspace_w_np, mps, lamda).run()
#         img_tv=np.fft.ifftshift(img_tv)
#         img_tv_real=np.real(img_tv)
#         # print(img_tv.shape)
#         # pl.ImagePlot(img_tv, title='Total Variation Regularized Reconstruction')
#         under_slice=under_img.numpy()[0].squeeze()
#         label_slice=T2label.numpy()[0].squeeze()
#         output_slice=img_tv_real
#         psnr = peak_signal_noise_ratio(label_slice, output_slice, data_range=label_slice.max())
#         psnr_zerof = peak_signal_noise_ratio(label_slice, under_slice, data_range=label_slice.max())
#         ssim = structural_similarity(label_slice, output_slice, data_range=label_slice.max())
#         ssim_zerof = structural_similarity(label_slice, under_slice, data_range=label_slice.max())
#         batch_psnr.append(psnr)
#         batch_ssim.append(ssim)
#         batch_psnr_zerof.append(psnr_zerof)
#         batch_ssim_zerof.append(ssim_zerof)
# average_psnr=np.mean(batch_psnr)
# std_psnr=np.std(batch_psnr,ddof=1)
# average_ssim=np.mean(batch_ssim)
# std_ssim=np.std(batch_ssim,ddof=1)
# average_psnr_zerof = np.mean(batch_psnr_zerof)
# std_psnr_zerof = np.std(batch_psnr_zerof, ddof=1)
# average_ssim_zerof = np.mean(batch_ssim_zerof)
# std_ssim_zerof = np.std(batch_ssim_zerof, ddof=1)
# print(
#     'zerof_psnr:{:.3f}\tzerof_ssim:{:.4f}\ttest_psnr:{:.3f}\ttest_ssim:{:.4f}'.format(
#          average_psnr_zerof, average_ssim_zerof, average_psnr, average_ssim))
# print(
#     'std_psnr_zerof:{:.3f}\tstd_ssim_zerof:{:.3f}\tstd_psnr:{:.3f}\tstd_ssim:{:.3f}'.format(
#         std_psnr_zerof, std_ssim_zerof, std_psnr, std_ssim))
        # reference_fill_up = rA(T1label, (1 - mask_under))
        # under_img = rAtA(T2label, mask_under)
        # under_kspace_w = rA(T2label, mask_under)
        # reference_fill_up = rA(T1label, (1 - mask_under))
        # under_kspace = under_kspace_w + reference_fill_up
        # under_img_fill = rifft2(under_kspace)
# # # #
# under_img_fill=under_img_fill.permute(0,3,1,2).contiguous()
# # avg=nn.AdaptiveAvgPool2d(1)
# # under=avg(under_img_fill)
# # print(under.shape)
# under_img_fill=torch.abs(under_img_fill)
# under_img_fill = under_img_fill.permute(0, 3, 1, 2).contiguous()
# under_img = torch.abs(under_img)
# dwt=DWT()
# iwt=IWT()
# # norm=nn.BatchNorm2d(1)
# # liner=nn.LeakyReLU(0.2,inplace=True)
# T2label=T2label.permute(0, 3, 1, 2).contiguous()
# T1label = T1label.permute(0, 3, 1, 2).contiguous()
# label=torch.cat([T1label,under_img],dim=1)
# T1label_wave=dwt(T1label)
# T2label_wave=dwt(T2label)
# # under_img_wave=dwt(under_img)
# # T=fuseCoeff(under_img_wave,T1label_wave)
# # T2=fuseCoeff(T2label_wave,T1label_wave)
# # T_i=iwt(T)
# # T2_i=iwt(T2)
# n=nn.Conv2d(in_channels=8,out_channels=4,kernel_size=(3,3), padding=1)
# z=torch.cat([T1label_wave,T2label_wave],dim=1)
# T2label1=n(z)
# T2label2=iwt(T2label1)
# # T2label1_c=iwt(T2label_wave)
# # T1label_r=iwt(T1label_wave)
# # #
# writer = SummaryWriter('reconstruction img')
# writer.add_images('fill',T1label)
# # writer.add_images('under',under_img_fill)
# # writer.add_images('full',T2label2)
# # # writer.add_images('rec_img',T2_i)
# writer.add_images('residual',T2label)
# # # # writer.add_images('fullsample_img',T1label_wave[:,2:3,:,:])
# # # # writer.add_images('undersample',T1label_wave[:,3:4,:,:])
# # # # writer.add_images('sample',T1label_r)
# writer.close()


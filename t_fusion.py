import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
from data.VIDataset_one import FusionDataset as FD
import time
import numpy as np
import cv2
from thop import profile
from thop import clever_format

time_s = [5, 10, 20]
def test_main(test_loader, MT):
    time_list = []
    for current_step, (test_data, file_names, yuv) in enumerate(test_loader):
        opt['model_df']['t'] = time_s
        start = time.time()
        diffusion.feed_data(test_data)
        fes_1, fes_2 = [], []
        fds_1, fds_2 = [], []
        # print(test_data['img'].shape,test_data['vis'].shape,test_data['ir'].shape)
        for t in opt['model_df']['t']:
            # print(file_names[0])
            test_data['img'] = test_data['ir']  # MRI
            fe_t_ir, fd_t_ir, _, _ = diffusion.get_feats(t=t)
            test_data['img'] = test_data['vis']  # CT/PET/SPECT
            fe_t_vis, fd_t_vis, _, _ = diffusion.get_feats(t=t)

            if opt['model_df']['feat_type'] == "dec":
                fds_1.append(fd_t_ir)
                del fd_t_ir
                fds_2.append(fd_t_vis)
                del fd_t_vis
            else:
                fes_1.append(fe_t_ir)
                del fe_t_ir
                fes_2.append(fe_t_vis)
                del fe_t_vis
            
        fds = {}
        fds['MRI'] = fds_1
        fds['Other'] = fds_2

        # Feeding features
        fussion_net.feed_data(fds, test_data)
        fussion_net.test()
        visuals = fussion_net.get_current_visuals()
        grid_img = visuals['pred_rgb'].detach()
        grid_img = Metrics.tensor2img(grid_img)

        yuv = yuv.squeeze(0).cpu()
        cb, cr = yuv[:, :, 1], yuv[:, :, 2]

        all_img = np.stack([grid_img[:,:,0], cb, cr], axis=-1)
        result = cv2.cvtColor(all_img, cv2.COLOR_YCrCb2BGR)
        test_result_path1 = test_result_path + f'{MT}/'
        os.makedirs(test_result_path1, exist_ok=True)
        cv2.imwrite('{}/{}'.format(test_result_path1, file_names[0]), result)
        # Metrics.save_img(grid_img, '{}/{}'.format(test_result_path, file_names[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_test_MMIF.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    # /media/sata1/hedan/test_imgs_IN/MRI-PET
    # '/media/sata1/hedan/MSRS-main/test/ir/'
    # '/media/sata1/hedan/MSRS-main/test/vi/'

    MTs = ['G','CT', 'PET', 'SPECT']
    # MTs = ['vi']
    test_loader = {}
    for MT in MTs:
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train' and args.phase != 'val':
                print("Creating train dataloader.")
                test_dataset = FD(split='val',
                                crop_size=dataset_opt['resolution'],
                                ir_path=f'/media/sata1/hedan/test_imgs_IN/MRI-{MT}/MRI/',
                                vi_path=f'/media/sata1/hedan/test_imgs_IN/MRI-{MT}/{MT}/',
                                # ir_path=f'/media/sata1/hedan/MSRS-main/test/ir/',
                                # vi_path=f'/media/sata1/hedan/MSRS-main/test/vi/',
                                is_crop=False)
                print("the training dataset is length:{}".format(test_dataset.length))
                test_loader[MT] = DataLoader(
                    dataset=test_dataset,
                    batch_size=dataset_opt['batch_size'],
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True,
                    drop_last=False,
                )
                test_loader[MT].n_iter = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Creating Fusion model
    fussion_net = Model.create_fusion_model(opt)

    logger.info('Begin Model Evaluation (testing).')
    test_result_path = '{}/test/'.format(opt['path']['results'])
    os.makedirs(test_result_path, exist_ok=True)
    temp_save = '{}/temp/'.format(opt['path']['results'])
    os.makedirs(temp_save, exist_ok=True)
    logger_test = logging.getLogger('test')  # test logger

    for MT in MTs:
        test_main(test_loader[MT], MT)

    logger.info('End of Testing.')

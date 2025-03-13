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
import cv2
import numpy as np
import random
from evaluator import ev_one


time_s = [5, 10, 20]
def test_main(test_loader, MT, test_result_path):
    time_list = []
    for current_step, (test_data, file_names, yuv) in enumerate(test_loader):
        diffusion.feed_data(test_data)
        fes_1, fes_2 = [], []
        fds_1, fds_2 = [], []

        opt['model_df']['t'] = time_s
        for t in opt['model_df']['t']:
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
        test_result_path1 = test_result_path + f'/MRI-{MT}/'
        os.makedirs(test_result_path1, exist_ok=True)
        cv2.imwrite('{}/{}'.format(test_result_path1, file_names[0]), result)
        # Metrics.save_img(grid_img, '{}/{}'.format(test_result_path, file_names[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
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

    Logger.setup_logger(None, opt['path']['log'],'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger1 = logging.getLogger('test')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    # Path_IR = 'Path_IR'
    # Path_VIS = 'Path_VIS'
    # train_mri_ct
    # Path_IR = '/media/sata1/hedan/KTZ_DATA/train_new/pair1'
    # Path_VIS = '/media/sata1/hedan/KTZ_DATA/train_new/pair2'
    Path_IR = '/media/sata1/hedan/KTZ_DATA/train_14/pair1'
    Path_VIS = '/media/sata1/hedan/KTZ_DATA/train_14/pair2'
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            train_dataset = FD(split='train',
                               crop_size=dataset_opt['resolution'],
                               ir_path=Path_IR,
                               vi_path=Path_VIS,
                               is_crop=True)

            print("the training dataset is length:{}".format(train_dataset.length))
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            train_loader.n_iter = len(train_loader)

    MTs = ['CT', 'PET', 'SPECT']
    test_loader = {}
    for MT in MTs:
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'test':
                print("Creating train dataloader.")
                test_dataset = FD(split='val',
                                crop_size=dataset_opt['resolution'],
                                ir_path=f'/media/sata1/hedan/test_imgs_IN/MRI-{MT}/MRI/',
                                vi_path=f'/media/sata1/hedan/test_imgs_IN/MRI-{MT}/{MT}/',
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

    # time_s = random.sample(range(range_time[0],range_time[1]),num_time)
    opt['model_df']['t'] = time_s

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Creating Fusion model
    fussion_net = Model.create_fusion_model(opt)

    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):
            train_result_path = '{}/train/{}'.format(opt['path']['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % fussion_net.optDF.param_groups[0]['lr']
            logger.info(message)
            for current_step, (train_data, file_names, _) in enumerate(train_loader):
                # train_data['img'] = torch.cat([train_data['vis'], train_data['ir']], dim=0)
                diffusion.feed_data(train_data)
                fes_1, fes_2 = [], []
                fds_1, fds_2 = [], []
 
                temp_save = '{}/temp/'.format(opt['path']['results'])
                os.makedirs(temp_save, exist_ok=True)
                
                for t in opt['model_df']['t']:

                    train_data['img'] = train_data['ir']  # MRI
                    fe_t_ir, fd_t_ir, _, _ = diffusion.get_feats(t=t)
                    train_data['img'] = train_data['vis']  # CT/PET/SPECT
                    fe_t_vis, fd_t_vis, x_noisy, x_final = diffusion.get_feats(t=t)

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
                fussion_net.feed_data(fds, train_data)
                fussion_net.optimize_parameters()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    fussion_net.update_loss()
                    logs = fussion_net.get_current_log()
                    message = '[Training FS]. epoch: [%d/%d]. Itter: [%d/%d], ' \
                              'All_loss: %.5f,Intensity_loss: %.5f, Grad_loss: %.5f' % \
                              (current_epoch, n_epoch - 1, current_step, len(train_loader), logs['l_all'],
                               logs['l_in'], logs['l_grad'])
                    logger.info(message)
        
            visuals = fussion_net.get_current_visuals()
            grid_img = torch.cat((visuals['pred_rgb'].detach(),
                                  visuals['gt_vis'],
                                  # visuals['gt_ir'].repeat(1, 3, 1, 1)), dim=0)
                                  visuals['gt_ir']), dim=0)
            grid_img = Metrics.tensor2img(grid_img)
            Metrics.save_img(grid_img, '{}/img_fused_e{}_b{}.png'.format(train_result_path,
                                                                         current_epoch,
                                                                         current_step))
            
            if (current_epoch >= 149) & ((current_epoch+1) % 20 == 0):
            # if current_epoch+1 == n_epoch:
                fussion_net.save_network(current_epoch)
                test_result_path = os.path.join(opt['path']['results'], 'pth_{}'.format(current_epoch))
                for MT in MTs:
                    test_main(test_loader[MT], MT, test_result_path)


        fussion_net._update_lr_schedulers()
        logger.info('End of fusion training.')
        fussion_net.save_network(current_epoch)



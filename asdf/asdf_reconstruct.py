import argparse
import json
import logging
import os
import random
import time
import torch
from torch.nn import functional as F
import numpy as np
import re

import asdf
import asdf.workspace as ws

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
    
def generation_atc_config(specs, reconstruction_codes_dir, reconstruction_models_dir=False, test_time_training=False, sampling_difficulty='easy', dataset_name='shape2motion'):
    '''
    Modify atc_dic to the angles to test on.
    
    '''
    if specs["NumAtcParts"]==1:

        if dataset_name=='rbo':
            lat_vec_dic = {}
            atc_vec_dic = {}
            if test_time_training==True:
                model_vec_dic = {}
            names = {'0001_000091_-021':[-38,-59],'0002_000091_-064':[-43,-16],'0003_000064_-021':[-21,-22],'0004_000114_-008':[-17,-56],
                     '0005_000101_-054':[-44,-25],'0006_000091_-062':[-35,-18],'0007_000049_-019':[-36,-74],'0008_000094_-041':[-27,-31],
                     '0009_000085_-038':[-52,-64],'0010_000119_-052':[-64,-34],'0011_000197_-014':[-34,-51],'0012_000040_-051':[-10,-48],
                     '0013_000076_-021':[-10,-61],'0014_000100_-021':[-34,-47],'0016_000122_-030':[-56,-44],
                     '0017_000115_-010':[-36,-67],'0018_000142_-049':[-83,-30],'0019_000092_-052':[-71,-36],'0020_000106_-051':[-71,-37]}
            for name in names.keys():
                for trg in names[name]:
                    trg += 20 # compensate for the angle difference between shape2motion and rbo dataset
                    if test_time_training==True:
                        model_vec_dic[(name[:4], trg)] = os.path.join(os.path.join(reconstruction_models_dir, dataset_name, name+'.pth'))
                    lat_vec_dic[(name[:4], trg)] = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, name+'.pth'))
                    atc_vec_dic[(name[:4], trg)] = np.array([trg])

        else:

            if specs["Class"]=='laptop':
                test_instances = [66,67,68,69,70,72,74,75,78,80,82]
                # angle range [-72,18]
                if sampling_difficulty=='easy':
                    angles_mapping = {-72:[-63,-45,-27],-54:[-63,-45,9],-36:[-45,-27,9],-18:[-27,-9,-63],0:[-9,9,-27],18:[9,-9,-36]} # 6-angle dataset
                elif sampling_difficulty=='hard':
                    angles_mapping = {-72:[-60,15,-48],-54:[-69,-39,3],-36:[-24,0,-57],-18:[-30,-42,-57],0:[15,-30,-42],18:[-12,-24,-30]} 
                else:
                    angles_mapping = {-36:np.arange(-72,19)} # 6-angle dataset animation
                
            if specs["Class"]=='stapler' or specs["Class"]=='washing_machine' or specs["Class"]=='door' or specs["Class"]=='oven':
                if specs["Class"]=='stapler':
                    test_instances = [24,25,26,27,28,29,30,31,32,33]
                if specs["Class"]=='washing_machine':
                    test_instances = [5,26,27,34,37,42,43,60]
                if specs["Class"]=='door':
                    test_instances = [10,27,38,59,86]
                if specs["Class"]=='oven':
                    test_instances = [2,7,10,16,18,28,30,33,34,41]
     
                # angle range [0,90]
                if sampling_difficulty=='easy':
                    angles_mapping = {18:[9,27,81],36:[27,45,81],54:[45,63,9],72:[63,81,9],90:[81,63,9]} # 6-angle dataset
                elif sampling_difficulty=='hard':
                    angles_mapping = {0:[39,60,24],18:[30,60,66],36:[69,84,21],54:[75,3,21],72:[33,12,66],90:[57,30,66]}
                else:
                    angles_mapping = {54:np.arange(0,91)} # 6-angle dataset animation
    
            lat_vec_dic = {}
            atc_vec_dic = {}
            if test_time_training==True:
                model_vec_dic = {}
            for test_ins in test_instances:
                for start in angles_mapping.keys():
                    for trg in angles_mapping[start]:
                        if test_time_training==True:
                            model_vec_dic[(test_ins, start, trg)] = os.path.join(os.path.join(reconstruction_models_dir, dataset_name, '{:04d}art{:04d}.pth'.format(test_ins, start)))
                        lat_vec_dic[(test_ins, start, trg)] = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}.pth'.format(test_ins, start)))
                        atc_vec_dic[(test_ins, start, trg)] = np.array([trg])

    if specs["NumAtcParts"]==2:      

        angles_mapping = {(0,0):[20,20],(0,20):[20,30],(0,40):[20,20], \
                (20,0):[30,20],(20,20):[10,30],(20,40):[10,20], \
                (40,0):[20,20],(40,20):[20,10],(40,40):[20,20]}

        if specs["Class"]=='eyeglasses':
            test_instances = [34,35,36,37,38,39,40,41,43]
            # eyeglasses angle range [0,50]
            if sampling_difficulty=='easy':
                # easy angles: aligned with interpolation
                angles_mapping = {(0,0):[[25,15]],(0,10):[[25,25]],(0,20):[[15,35]],(0,30):[[25,15]],(0,40):[[15,25]],(0,50):[[25,35]], \
                       (10,0):[[15,25]],(10,10):[[5,25]],(10,20):[[25,35]],(10,30):[[25,15]],(10,40):[[25,25]],(10,50):[[25,25]], \
                       (20,0):[[35,25]],(20,10):[[15,35]],(20,20):[[35,15]],(20,30):[[15,15]],(20,40):[[35,25]],(20,50):[[35,25]], \
                       (30,0):[[25,25]],(30,10):[[15,25]],(30,20):[[15,35]],(30,30):[[25,35]],(30,40):[[15,45]],(30,50):[[25,25]], \
                       (40,0):[[25,5]],(40,10):[[45,15]],(40,20):[[25,35]],(40,30):[[25,15]],(40,40):[[45,35]],(40,50):[[35,35]], \
                       (50,0):[[35,15]],(50,10):[[25,25]],(50,20):[[35,35]],(50,30):[[45,15]],(50,40):[[35,25]],(50,50):[[45,45]]}
                angles_mapping = angles_mapping
            else:
                angles_mapping = {}
                angles_mapping[(40,20)] = []
                for i in range(0,55,5):
                    for j in range(0,55,5):
                        angles_mapping[(40,20)].append([i,j])
                print(angles_mapping)

        if specs["Class"]=='refrigerator':
            # refrigerator angle range [40,90]
            test_instances = [6,17,27,46,61,65,78]
            #test_instances = [61]
            if sampling_difficulty=='easy':
                angles_mapping = {(90,90):[[65,75]],(90,80):[[65,65]],(90,70):[[75,55]],(90,60):[[65,75]],(90,50):[[75,65]],(90,40):[[65,55]], \
                       (80,90):[[75,65]],(80,80):[[85,65]],(80,70):[[65,55]],(80,60):[[65,75]],(80,50):[[65,65]],(80,40):[[65,65]], \
                       (70,90):[[55,65]],(70,80):[[75,55]],(70,70):[[55,75]],(70,60):[[75,75]],(70,50):[[55,65]],(70,40):[[55,65]], \
                       (60,90):[[65,65]],(60,80):[[75,65]],(60,70):[[75,55]],(60,60):[[65,55]],(60,50):[[75,45]],(60,40):[[65,65]], \
                       (50,90):[[65,85]],(50,80):[[45,75]],(50,70):[[65,55]],(50,60):[[65,75]],(50,50):[[45,65]],(50,40):[[55,55]], \
                       (40,90):[[55,75]],(40,80):[[65,65]],(40,70):[[55,55]],(40,60):[[55,75]],(40,50):[[55,65]],(40,40):[[45,45]]}
            else:
                angles_mapping = {}
                angles_mapping[(90,70)] = []
                for i in range(90,35,-5):
                    for j in range(90,35,-5):
                        angles_mapping[(90,70)].append([i,j])
                print(angles_mapping)

        lat_vec_dic = {}
        atc_vec_dic = {}
        if test_time_training==True:
            model_vec_dic = {}
        for test_ins in test_instances:
            for start in angles_mapping.keys():
                for trg in angles_mapping[start]:
                    if test_time_training==True:
                        model_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = os.path.join(os.path.join(reconstruction_models_dir, dataset_name, '{:04d}art{:04d}{:04d}.pth'.format(test_ins, start[0], start[1])))

                    lat_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = torch.load(
                        os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}{:04d}.pth'.format(test_ins, start[0], start[1]))
                    )
                    atc_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = np.array([trg])

    if test_time_training==True:
        return lat_vec_dic, atc_vec_dic, model_vec_dic
    else:
        return lat_vec_dic, atc_vec_dic

def interpolation_atc_config(specs, reconstruction_codes_dir, dataset_name='shape2motion'):
    '''
    Modify atc_dic to the angles to test on.
    
    '''
    if specs["NumAtcParts"]==1:
        if specs["Class"]=='laptop':
            test_instances = [66,67,68,69,70,72,74,75,78,80,82]

            angles_mapping = {(-72,18):[13/30,17/30,23/30],(-54,0):[2/18,4/18,5/18],(-36,18):[5/18,7/18,11/18],(18,-54):[17/24,15/24,7/24]}
    #         # extrapolation
    #         angles_mapping = {(-54,-72):[1/6,2/6,3/6,4/6], (0,18):[1/6,2/6,3/6,4/6]}

        if specs["Class"]=='stapler' or specs["Class"]=='washing_machine' or specs["Class"]=='oven' or specs["Class"]=='door':
            if specs["Class"]=='stapler':
                test_instances = [24,25,26,27,28,29,30,31,32,33]
            if specs["Class"]=='washing_machine':
                test_instances = [5,26,27,34,37,42,43,60]
            if specs["Class"]=='door':
                test_instances = [10,27,38,59,86]
            if specs["Class"]=='oven':
                test_instances = [2,7,10,16,18,28,30,33,34,41]

            angles_mapping = {(90,0):[13/30,17/30,23/30],(54,0):[2/18,4/18,5/18],(36,90):[5/18,7/18,11/18],(18,90):[17/24,15/24,7/24]}

        lat_vec_dic = {}
        atc_vec_dic = {}
        for test_ins in test_instances:
            for (start, end) in angles_mapping.keys():
                for factor in angles_mapping[(start,end)]:
                    trg = int(factor * (end-start) + start)
                    #trg = int(factor * (end-start) + end) # extra
                    lat_start = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}.pth'.format(test_ins, start)))
                    lat_end = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}.pth'.format(test_ins, end)))
                    #lat_vec_dic[(test_ins, start, trg)] = factor * (lat_end - lat_start) + lat_start
                    lat_vec_dic[(test_ins, start, trg)] = lat_start
                    #code_dic[(test_ins, start, trg)] = factor * (lat_end - lat_start) + lat_end # extra
                    if specs["Articulation"]==True:
                        atc_start = np.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}.npy'.format(test_ins, start)))
                        atc_end = np.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}.npy'.format(test_ins, end)))
                        atc_vec_dic[(test_ins, start, trg)] = factor * (atc_end - atc_start) + atc_start
                        #atc_dic[(test_ins, start, trg)] = factor * (atc_end - atc_start) + atc_end # extra

    if specs["NumAtcParts"]==2:

        angle_mapping = {(0,0):[40,40],(0,20):[40,40],(0,40):[40,0], \
                (20,0):[40,40],(20,20):[0,40],(20,40):[0,0], \
                (40,0):[0,40],(40,20):[0,0],(40,40):[0,0]}

        if specs["Class"]=='eyeglasses':
            # angles_aligned
            test_instances = [34,35,36,37,38,39,40,41,43]

        if specs["Class"]=='refrigerator':
            # angles_aligned
            test_instances = [6,17,27,46,61,65,78]
            angle_mapping_90 = {}
            for (i,j) in angle_mapping.keys():
                angle_mapping_90[(90-i,90-j)] = [90-angle_mapping[(i,j)][0], 90-angle_mapping[(i,j)][1]]
            angle_mapping = angle_mapping_90

        lat_vec_dic = {}
        atc_vec_dic = {}
        for test_ins in test_instances:
            for k in angle_mapping.keys():
                    start = list(k)
                    end = angle_mapping[k]
                    trg = (np.array(list(start)) + np.array(end))//2
                    lat_start = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}{:04d}.pth'.format(test_ins, start[0], start[1])))
                    lat_end = torch.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}{:04d}.pth'.format(test_ins, end[0], end[1])))
                    #lat_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = (lat_end + lat_start)/2
                    lat_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = lat_start
                    if specs["Articulation"]==True:
                        atc_start = np.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}{:04d}.npy'.format(test_ins, start[0], start[1])))
                        atc_end = np.load(os.path.join(reconstruction_codes_dir, dataset_name, '{:04d}art{:04d}{:04d}.npy'.format(test_ins, end[0], end[1])))
                        atc_vec_dic[(test_ins, start[0], start[1], trg[0], trg[1])] = (atc_end + atc_start)/2

    return lat_vec_dic, atc_vec_dic

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    clamp_dist,
    num_samples=8000,
    lr=5e-3,
    l2reg=True,
    articulation=True,
    specs=None,
    infer_with_gt_atc=False,
    num_atc_parts=1,
    do_sup_with_part=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        #lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        if num_iterations<adjust_lr_every*2:
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        else:
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every - 2))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    # init latent code optimizer
    latent_rand = torch.ones(1, latent_size).normal_(mean=0, std=0.01).cuda()
    lat_vec = latent_rand.clone()
    lat_vec.requires_grad = True
    lat_optimizer = torch.optim.Adam([lat_vec], lr=lr)

    # init articulation code optimizer
    if infer_with_gt_atc==False:
        if num_atc_parts==1:
            if specs["Class"]=='laptop':
                atc_vec = torch.Tensor([-30]).view(1,1).cuda()
            elif specs["Class"]=='stapler' or specs["Class"]=='washing_machine' or specs["Class"]=='door' or specs["Class"]=='oven':
                atc_vec = torch.Tensor([45]).view(1,1).cuda()
            else:
                raise Exception("Undefined classes")
        if num_atc_parts==2:
            if specs["Class"]=='eyeglasses':
                atc_vec = torch.Tensor([25, 25]).view(1,2).cuda()
            elif specs["Class"]=='refrigerator':
                atc_vec = torch.Tensor([75, 75]).view(1,2).cuda()
            else:
                raise Exception("Undefined classes")
        atc_vec.requires_grad = True
        atc_optimizer = torch.optim.Adam([atc_vec], lr=lr*1000)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    decoder.eval()

    # two-step optimization
    for e in range(num_iterations*2):

        # re-initialize lat_vec
        if e==num_iterations:
            lat_vec = latent_rand.clone() 
            lat_vec.requires_grad = True
            lat_optimizer = torch.optim.Adam([lat_vec], lr=lr)

        sdf_data = asdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples, articulation,
        )
        if articulation==True:
            xyz = sdf_data[0][:, 0:3].float().cuda()
            sdf_gt = sdf_data[0][:, 3].unsqueeze(1).cuda()
            part_gt = sdf_data[0][:, 4].unsqueeze(1).long().cuda()
            if infer_with_gt_atc:
                atc_vecs = sdf_data[1].view(1,num_atc_parts).expand(xyz.shape[0],num_atc_parts).cuda()
            else:
                atc_vecs = atc_vec.expand(xyz.shape[0],num_atc_parts).cuda()
        else:
            xyz = sdf_data[:, 0:3].float().cuda()
            sdf_gt = sdf_data[:, 3].unsqueeze(1).cuda()

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, lat_optimizer, e, decreased_by, adjust_lr_every)
        if infer_with_gt_atc==False:
            adjust_learning_rate(lr*1000, atc_optimizer, e, decreased_by, adjust_lr_every)

        lat_optimizer.zero_grad()
        if infer_with_gt_atc==False:
            atc_optimizer.zero_grad()

        lat_vecs = lat_vec.expand(num_samples, -1)

        if articulation==True:
            inputs = torch.cat([lat_vecs, xyz, atc_vecs], 1).cuda()
        else:
            inputs = torch.cat([lat_vecs, xyz], 1).cuda()
        if do_sup_with_part:
            pred_sdf, pred_part = decoder(inputs)
        else:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        loss = loss_l1(pred_sdf, sdf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(lat_vec.pow(2))
        if do_sup_with_part:
            loss += 1e-3 * F.cross_entropy(pred_part, part_gt.view(-1).cuda())

        loss.backward()
        lat_optimizer.step()
        if infer_with_gt_atc==False and e<num_iterations:
            atc_optimizer.step()

        loss_num = loss.cpu().data.numpy()

    #pos_mask = (torch.sign(pred_sdf)!=torch.sign(sdf_gt)).data & (sdf_gt>0).data
    #neg_mask = (torch.sign(pred_sdf)!=torch.sign(sdf_gt)).data & (sdf_gt<0).data
    #print(torch.sum(pos_mask), torch.sum(neg_mask))

    if articulation==True:
        if infer_with_gt_atc:
            return loss_num, None, lat_vec, sdf_data[1].view(1,-1)
        else:
            # computer angle pred acc
            atc_err = torch.mean(torch.abs(atc_vec.detach() - sdf_data[1].cuda())).cpu().data.numpy()
            print(atc_vec)
            return loss_num, atc_err, lat_vec, atc_vec

    else:
        return loss_num, lat_vec
    

def reconstruct_ttt(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    clamp_dist,
    num_samples=8000,
    lr=5e-3,
    l2reg=True,
    articulation=True,
    specs=None,
    infer_with_gt_atc=False,
    num_atc_parts=1,
    do_sup_with_part=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        if num_iterations<adjust_lr_every*2:
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        elif num_iterations<adjust_lr_every*4:
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every - 2))
        else:
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every - 4))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    # init latent code optimizer
    latent_rand = torch.ones(1, latent_size).normal_(mean=0, std=0.01).cuda()
    lat_vec = latent_rand.clone()
    lat_vec.requires_grad = True
    lat_optimizer = torch.optim.Adam([lat_vec], lr=lr)

    opt_params = []
    for idx, (name, param) in enumerate(decoder.named_parameters()):
        n = re.split('\.', name)
        if n[1] in ['lin0', 'lin1', 'lin2', 'lin3']:
            opt_params.append(param)
    
        decoder_optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, opt_params)}], lr=0.01*lr)

    # init articulation code optimizer
    if infer_with_gt_atc==False:
        if num_atc_parts==1:
            if specs["Class"]=='laptop':
                atc_vec = torch.Tensor([-30]).view(1,1).cuda()
            elif specs["Class"]=='stapler' or specs["Class"]=='washing_machine' or specs["Class"]=='door' or specs["Class"]=='oven':
                atc_vec = torch.Tensor([45]).view(1,1).cuda()
            else:
                raise Exception("Undefined classes")
        if num_atc_parts==2:
            if specs["Class"]=='eyeglasses':
                atc_vec = torch.Tensor([25, 25]).view(1,2).cuda()
            elif specs["Class"]=='refrigerator':
                atc_vec = torch.Tensor([75, 75]).view(1,2).cuda()
            else:
                raise Exception("Undefined classes")
        atc_vec.requires_grad = True
        atc_optimizer = torch.optim.Adam([atc_vec], lr=1000*lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    decoder.eval()

    # two-step optimization
    for e in range(num_iterations*3):

        # re-initialize lat_vec
        if e==num_iterations:
            lat_vec = latent_rand.clone() 
            lat_vec.requires_grad = True
            lat_optimizer = torch.optim.Adam([lat_vec], lr=lr)

        sdf_data = asdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples, articulation,
        )
        if articulation==True:
            xyz = sdf_data[0][:, 0:3].float().cuda()
            sdf_gt = sdf_data[0][:, 3].unsqueeze(1).cuda()
            part_gt = sdf_data[0][:, 4].unsqueeze(1).long().cuda()
            if infer_with_gt_atc:
                atc_vecs = sdf_data[1].view(1,num_atc_parts).expand(xyz.shape[0],num_atc_parts).cuda()
            else:
                atc_vecs = atc_vec.expand(xyz.shape[0],num_atc_parts).cuda()
        else:
            xyz = sdf_data[:, 0:3].float().cuda()
            sdf_gt = sdf_data[:, 3].unsqueeze(1).cuda()

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, lat_optimizer, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(0.01*lr, decoder_optimizer, e, decreased_by, adjust_lr_every)
        if infer_with_gt_atc==False:
            adjust_learning_rate(1000*lr, atc_optimizer, e, decreased_by, adjust_lr_every)

        lat_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        if infer_with_gt_atc==False:
            atc_optimizer.zero_grad()

        lat_vecs = lat_vec.expand(num_samples, -1)

        if articulation==True:
            inputs = torch.cat([lat_vecs, xyz, atc_vecs], 1).cuda()
        else:
            inputs = torch.cat([lat_vecs, xyz], 1).cuda()
        if do_sup_with_part:
            pred_sdf, pred_part = decoder(inputs)
        else:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        loss = loss_l1(pred_sdf, sdf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(lat_vec.pow(2))
        if do_sup_with_part:
            loss += 1e-3 * F.cross_entropy(pred_part, part_gt.view(-1).cuda())

        loss.backward()
        #lat_optimizer.step()
        if infer_with_gt_atc==False and e<num_iterations:
            atc_optimizer.step()
            lat_optimizer.step()
        elif e<num_iterations*2:
            lat_optimizer.step()
        else:
            decoder_optimizer.step()

        loss_num = loss.cpu().data.numpy()

    #pos_mask = (torch.sign(pred_sdf)!=torch.sign(sdf_gt)).data & (sdf_gt>0).data
    #neg_mask = (torch.sign(pred_sdf)!=torch.sign(sdf_gt)).data & (sdf_gt<0).data
    #print(torch.sum(pos_mask), torch.sum(neg_mask))

    if articulation==True:
        if infer_with_gt_atc:
            return loss_num, None, lat_vec, sdf_data[1].view(1,-1)
        else:
            # computer angle pred acc
            atc_err = torch.mean(torch.abs(atc_vec.detach() - sdf_data[1].cuda())).cpu().data.numpy()
            print(atc_vec)
            return loss_num, atc_err, lat_vec, atc_vec

    else:
        return loss_num, lat_vec

    
def reconstruct_testset(args, ws, specs, decoder, npz_filenames, saved_model_epoch, dataset_name):

    # build saving directory
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.recon_testset_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)    
    
    err_sum = 0.0
    atc_err_sum = 0.0
    save_latvec_only = False

    # generate meshes
    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        if dataset_name=='rbo':
            data_sdf = asdf.data.read_sdf_samples_into_ram_rbo(full_filename, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"])
        else:
            data_sdf = asdf.data.read_sdf_samples_into_ram(full_filename, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"])

        #dataset_name = re.split('/', npz)[-3]
        npz_name = re.split('/', npz)[-1][:-4]
        mesh_filename = os.path.join(reconstruction_meshes_dir, dataset_name, npz_name)
        latent_filename = os.path.join(reconstruction_codes_dir, dataset_name, npz_name + ".pth")

        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
            and os.path.isfile(latent_filename)
        ):
            continue

        logging.info("reconstructing {}".format(npz))

        if specs["Articulation"]==True:
            data_sdf[0][0] = data_sdf[0][0][torch.randperm(data_sdf[0][0].shape[0])]
            data_sdf[0][1] = data_sdf[0][1][torch.randperm(data_sdf[0][1].shape[0])]
        else:
            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        start = time.time()
        if specs["Articulation"]==True:
            err, atc_err, lat_vec, atc_vec = reconstruct(
                decoder,
                int(args.iterations),
                specs["CodeLength"],
                data_sdf,
                specs["ClampingDistance"],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                articulation=specs["Articulation"],
                specs=specs,
                infer_with_gt_atc=args.infer_with_gt_atc,
                num_atc_parts=specs["NumAtcParts"],
                do_sup_with_part=specs["TrainWithParts"],
            )

        else:
            err, lat_vec = reconstruct(
                decoder,
                int(args.iterations),
                specs["CodeLength"],
                data_sdf,
                specs["ClampingDistance"],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                articulation=specs["Articulation"],
                specs=specs,
                infer_with_gt_atc=True,
                num_atc_parts=specs["NumAtcParts"],
                do_sup_with_part=False,
            )
            atc_vec = None

        if specs["Articulation"]==True and args.infer_with_gt_atc==False:
            print("err: ", err, "atc_err: ", atc_err)
            atc_err_sum += atc_err
            err_sum += err
            print("err avg: ", err_sum/(ii+1), "atc_err avg: ", atc_err_sum/(ii+1))
        else:
            err_sum += err
            print("err avg: ", err_sum/(ii+1))

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                asdf.mesh.create_mesh(
                    decoder, lat_vec, mesh_filename, N=256, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
                )
            logging.info("total time: {}".format(time.time() - start))

        if not os.path.exists(os.path.dirname(latent_filename)):
            os.makedirs(os.path.dirname(latent_filename))

        torch.save(lat_vec.unsqueeze(0), latent_filename)
        if specs["Articulation"]==True:
            print("save atc npy: ", latent_filename[:-4]+'.npy', atc_vec.detach().cpu().numpy())
            with open(latent_filename[:-4]+'.npy', 'wb') as f:
                np.save(f, atc_vec.detach().cpu().numpy())


def reconstruct_testset_ttt(args, ws, specs, decoder, npz_filenames, saved_model_state, dataset_name):

    # build saving directory
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.recon_testset_ttt_subdir, str(saved_model_state["epoch"])
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir) 

    reconstruction_models_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_models_subdir
    )
    if not os.path.isdir(reconstruction_models_dir):
        os.makedirs(reconstruction_models_dir) 
    
    err_sum = 0.0
    atc_err_sum = 0.0
    save_latvec_only = False

    # generate meshes
    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        decoder.load_state_dict(saved_model_state["model_state_dict"])

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        if dataset_name=='rbo':
            data_sdf = asdf.data.read_sdf_samples_into_ram_rbo(full_filename, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"])
        else:
            data_sdf = asdf.data.read_sdf_samples_into_ram(full_filename, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"])

        #dataset_name = re.split('/', npz)[-3]
        npz_name = re.split('/', npz)[-1][:-4]
        mesh_filename = os.path.join(reconstruction_meshes_dir, dataset_name, npz_name)
        latent_filename = os.path.join(reconstruction_codes_dir, dataset_name, npz_name + ".pth")
        model_filename = os.path.join(reconstruction_models_dir, dataset_name, npz_name + ".pth")

        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
            and os.path.isfile(latent_filename)
        ):
            continue

        logging.info("reconstructing {}".format(npz))

        if specs["Articulation"]==True:
            data_sdf[0][0] = data_sdf[0][0][torch.randperm(data_sdf[0][0].shape[0])]
            data_sdf[0][1] = data_sdf[0][1][torch.randperm(data_sdf[0][1].shape[0])]
        else:
            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        start = time.time()
        if specs["Articulation"]==True:
            err, atc_err, lat_vec, atc_vec = reconstruct_ttt(
                decoder,
                int(args.iterations),
                specs["CodeLength"],
                data_sdf,
                specs["ClampingDistance"],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                articulation=specs["Articulation"],
                specs=specs,
                infer_with_gt_atc=args.infer_with_gt_atc,
                num_atc_parts=specs["NumAtcParts"],
                do_sup_with_part=specs["TrainWithParts"],
            )

        else:
            err, lat_vec = reconstruct_ttt(
                decoder,
                int(args.iterations),
                specs["CodeLength"],
                data_sdf,
                specs["ClampingDistance"],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                articulation=specs["Articulation"],
                specs=specs,
                infer_with_gt_atc=True,
                num_atc_parts=specs["NumAtcParts"],
                do_sup_with_part=False,
            )
            atc_vec = None

        if specs["Articulation"]==True and args.infer_with_gt_atc==False:
            print("err: ", err, "atc_err: ", atc_err)
            atc_err_sum += atc_err
            err_sum += err
            print("err avg: ", err_sum/(ii+1), "atc_err avg: ", atc_err_sum/(ii+1))
        else:
            err_sum += err
            print("err avg: ", err_sum/(ii+1))

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                asdf.mesh.create_mesh(
                    decoder, lat_vec, mesh_filename, N=256, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
                )
            logging.info("total time: {}".format(time.time() - start))

        if not os.path.exists(os.path.dirname(latent_filename)):
            os.makedirs(os.path.dirname(latent_filename))
        if not os.path.exists(os.path.dirname(model_filename)):
            os.makedirs(os.path.dirname(model_filename))

        torch.save(lat_vec.unsqueeze(0), latent_filename)
        torch.save(decoder.state_dict(), model_filename)
        if specs["Articulation"]==True:
            print("save atc npy: ", latent_filename[:-4]+'.npy', atc_vec.detach().cpu().numpy())
            with open(latent_filename[:-4]+'.npy', 'wb') as f:
                np.save(f, atc_vec.detach().cpu().numpy())


def generation(args, ws, specs, decoder, reconstruction_codes_dir, saved_model_epoch, dataset_name='shape2motion'):
    
    # build saving directory
    generation_dir = os.path.join(
        args.experiment_directory, ws.generation_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(generation_dir):
        os.makedirs(generation_dir)

    gen_meshes_dir = os.path.join(
        generation_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(gen_meshes_dir):
        os.makedirs(gen_meshes_dir)

    save_latvec_only = False
    num_atc_parts = specs["NumAtcParts"]
    lat_vec_dic, atc_vec_dic = generation_atc_config(specs, reconstruction_codes_dir, dataset_name=dataset_name)

    for ii, k in enumerate(lat_vec_dic.keys()):

        lat_vec = lat_vec_dic[k].view(1, specs["CodeLength"])
        atc_vec = torch.Tensor(atc_vec_dic[k]).float().view(1,num_atc_parts)

        if num_atc_parts==1:
            if dataset_name=='rbo':
                mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{}_{:04d}'.format(k[0],int(k[1])))
            else:
                mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{:04d}art{:04d}_lat_from{:04d}to{:04d}'.format(k[0],k[1],k[1],k[2]))
        if num_atc_parts==2:
            mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{:04d}art{:04d}{:04d}_lat_from{:04d}{:04d}to{:04d}{:04d}'.format(k[0],k[1],k[2],k[1],k[2],k[3],k[4]))

        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
        ):
            continue

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        logging.info("reconstructing {}".format(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                asdf.mesh.create_mesh(
                    decoder, lat_vec, mesh_filename, N=256, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
                )
            logging.info("total time: {}".format(time.time() - start))

            if specs["TrainWithParts"]==True:
                xyz = sample_uniform_points_in_unit_sphere(30000) * 0.75
                xyz = torch.Tensor(xyz).cuda()
                if num_atc_parts==1:
                    atc_vec = atc_vec.view(1,1).expand(xyz.shape[0],1).cuda()
                if num_atc_parts==2:
                    atc_vec = atc_vec.view(1,2).expand(xyz.shape[0],2).cuda()
                lat_vec = lat_vec.detach().expand(xyz.shape[0], -1)
                inputs = torch.cat([lat_vec, xyz, atc_vec], 1).cuda()
                sdf, pred_part = decoder(inputs)

                _, part_pred = pred_part.topk(1, 1, True, True)
                part_pred = part_pred.detach()
                sdf = sdf.detach()
                pos = np.concatenate([xyz[sdf.view(-1)>0].cpu().numpy(), sdf[sdf.view(-1)>0].cpu().numpy().reshape(-1,1), part_pred[sdf.view(-1)>0].cpu().numpy().reshape(-1,1)], axis=1)
                neg = np.concatenate([xyz[sdf.view(-1)<0].cpu().numpy(), sdf[sdf.view(-1)<0].cpu().numpy().reshape(-1,1), part_pred[sdf.view(-1)<0].cpu().numpy().reshape(-1,1)], axis=1)
                np.savez(mesh_filename+'.npz', pos=pos, neg=neg)


def generation_ttt(args, ws, specs, decoder, reconstruction_codes_dir, reconstruction_models_dir, saved_model_epoch, dataset_name='shape2motion'):
    
    # build saving directory
    generation_dir = os.path.join(
        args.experiment_directory, ws.generation_ttt_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(generation_dir):
        os.makedirs(generation_dir)

    gen_meshes_dir = os.path.join(
        generation_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(gen_meshes_dir):
        os.makedirs(gen_meshes_dir)

    save_latvec_only = False
    num_atc_parts = specs["NumAtcParts"]
    lat_vec_dic, atc_vec_dic, model_vec_dic = generation_atc_config(specs, reconstruction_codes_dir, reconstruction_models_dir, test_time_training=True, dataset_name=dataset_name)

    for ii, k in enumerate(lat_vec_dic.keys()):

        lat_vec = lat_vec_dic[k].view(1,specs["CodeLength"])
        atc_vec = torch.Tensor(atc_vec_dic[k]).float().view(1,num_atc_parts)

        saved_model_state = torch.load(model_vec_dic[k])
        decoder.load_state_dict(saved_model_state)

        if num_atc_parts==1:
            if dataset_name=='rbo':
                mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{}_{:04d}'.format(k[0],int(k[1]))) # real data
            else:
                mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{:04d}art{:04d}_lat_from{:04d}to{:04d}'.format(k[0],k[1],k[1],k[2]))
        if num_atc_parts==2:
            mesh_filename = os.path.join(gen_meshes_dir, dataset_name, '{:04d}art{:04d}{:04d}_lat_from{:04d}{:04d}to{:04d}{:04d}'.format(k[0],k[1],k[2],k[1],k[2],k[3],k[4]))

        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
        ):
            continue

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        logging.info("reconstructing {}".format(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                asdf.mesh.create_mesh(
                    decoder, lat_vec, mesh_filename, N=256, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
                )
            logging.info("total time: {}".format(time.time() - start))

            if specs["TrainWithParts"]==True:
                xyz = sample_uniform_points_in_unit_sphere(30000) * 0.75
                xyz = torch.Tensor(xyz).cuda()
                if num_atc_parts==1:
                    atc_vec = atc_vec.view(1,1).expand(xyz.shape[0],1).cuda()
                if num_atc_parts==2:
                    atc_vec = atc_vec.view(1,2).expand(xyz.shape[0],2).cuda()
                lat_vec = lat_vec.detach().expand(xyz.shape[0], -1)
                inputs = torch.cat([lat_vec, xyz, atc_vec], 1).cuda()
                sdf, pred_part = decoder(inputs)

                _, part_pred = pred_part.topk(1, 1, True, True)
                part_pred = part_pred.detach()
                sdf = sdf.detach()
                pos = np.concatenate([xyz[sdf.view(-1)>0].cpu().numpy(), sdf[sdf.view(-1)>0].cpu().numpy().reshape(-1,1), part_pred[sdf.view(-1)>0].cpu().numpy().reshape(-1,1)], axis=1)
                neg = np.concatenate([xyz[sdf.view(-1)<0].cpu().numpy(), sdf[sdf.view(-1)<0].cpu().numpy().reshape(-1,1), part_pred[sdf.view(-1)<0].cpu().numpy().reshape(-1,1)], axis=1)
                np.savez(mesh_filename+'.npz', pos=pos, neg=neg)


def interpolate_testset(args, ws, specs, decoder, reconstruction_codes_dir, saved_model_epoch, dataset_name='shape2motion'):

    # build saving directory
    inter_dir = os.path.join(
        args.experiment_directory, ws.inter_testset_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(inter_dir):
        os.makedirs(inter_dir)

    inter_meshes_dir = os.path.join(
        inter_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(inter_meshes_dir):
        os.makedirs(inter_meshes_dir)

    save_latvec_only = False
    num_atc_parts = specs["NumAtcParts"]
    lat_vec_dic, atc_vec_dic = interpolation_atc_config(specs, reconstruction_codes_dir)
    
    for ii, k in enumerate(lat_vec_dic.keys()):

        lat_vec = lat_vec_dic[k].view(1,specs["CodeLength"])
        
        if specs["Articulation"]==True:
            atc_vec = torch.Tensor(atc_vec_dic[k]).view(1,num_atc_parts)
        else:
            atc_vec = None

        if num_atc_parts==1:
            mesh_filename = os.path.join(inter_meshes_dir, dataset_name, '{:04d}art{:04d}_lat_from{:04d}to{:04d}'.format(k[0],k[1],k[1],k[2]))
        if num_atc_parts==2:
            mesh_filename = os.path.join(inter_meshes_dir, dataset_name, '{:04d}art{:04d}{:04d}_lat_from{:04d}{:04d}to{:04d}{:04d}'.format(k[0],k[1],k[2],k[1],k[2],k[3],k[4]))

        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
        ):
            continue

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        logging.info("reconstructing {}".format(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                asdf.mesh.create_mesh(
                    decoder, lat_vec, mesh_filename, N=256, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
                )
            logging.info("total time: {}".format(time.time() - start))

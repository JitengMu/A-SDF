#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.nn import functional as F

import signal
import sys
import os
import logging
import math
import json
import time
import numpy as np
import argparse

import asdf
from asdf.utils import *
import asdf.workspace as ws

def load_checkpoints(continue_from, ws, experiment_directory, lat_vecs, decoder, optimizer_all):

    logging.info('continuing from "{}"'.format(continue_from))

    lat_vecs, lat_epoch = load_latent_vectors(
        ws, experiment_directory, continue_from + ".pth", lat_vecs
    )

    decoder, model_epoch = ws.load_model_parameters(
        experiment_directory, continue_from, decoder
    )

    optimizer_all, optimizer_epoch = load_optimizer(
        ws, experiment_directory, continue_from + ".pth", optimizer_all
    )

    loss_log, lr_log, timing_log, log_epoch = load_logs(
        ws, experiment_directory
    )

    if not log_epoch == model_epoch:
        loss_log, lr_log, timing_log = clip_logs(
            loss_log, lr_log, timing_log, model_epoch
        )

    if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
        raise RuntimeError(
            "epoch mismatch: {} vs {} vs {} vs {}".format(
                model_epoch, optimizer_epoch, lat_epoch, log_epoch
            )
        )

    start_epoch = model_epoch + 1

    return lat_vecs, decoder, optimizer_all, start_epoch, loss_log, lr_log, timing_log, start_epoch


def main_function(experiment_directory, continue_from, batch_split):

    def save_latest(epoch):
        save_model(ws, experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(ws, experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(ws, experiment_directory, "latest.pth", lat_vecs, epoch)
    
    def save_checkpoints(epoch):
        save_model(ws, experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(ws, experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(ws, experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    signal.signal(signal.SIGINT, signal_handler)

    # load specs 
    specs = ws.load_experiment_specifications(experiment_directory)

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]

    num_epochs = specs["NumEpochs"]
    lr_schedules = get_learning_rate_schedules(specs)
    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))
    do_sup_with_part = specs["TrainWithParts"]
    num_samp_per_scene = specs["SamplesPerScene"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_bound = get_spec_with_default(specs, "CodeBound", 0.1)

    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    # init dataloader
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = asdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"])

    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads =specs["DataLoaderThreads"]
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    
    # init model and shape codes
    decoder = arch.Decoder(num_atc_parts=specs["NumAtcParts"], do_sup_with_part=specs["TrainWithParts"]).cuda()
    decoder = torch.nn.DataParallel(decoder)

    if specs["Articulation"]==True:
        num_scenes = specs["NumInstances"]
    else:
        num_scenes = len(sdf_dataset)
    logging.info("There are {} scenes".format(num_scenes))
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)

    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    # loss and optimizer
    loss_l1 = torch.nn.L1Loss(reduction='sum')

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )


    loss_log = []
    lr_log = []
    timing_log = []
    start_epoch = 1

    if continue_from is not None:
        lat_vecs, decoder, optimizer_all, start_epoch, loss_log, lr_log, timing_log, start_epoch = load_checkpoints(continue_from, ws, experiment_directory, lat_vecs, decoder, optimizer_all)

    logging.info("starting from epoch {}".format(start_epoch))
    logging.info("Number of decoder parameters: {}".format(sum(p.data.nelement() for p in decoder.parameters())))
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for all_sdf_data, indices in sdf_loader:
            # Process the input data
            if specs["Articulation"]==True:
                sdf_data = all_sdf_data[0].reshape(-1, 5)
                atc = all_sdf_data[1].view(-1,specs["NumAtcParts"])
                instance_idx = all_sdf_data[2].view(-1,1)
                atc = atc.repeat(1, all_sdf_data[0].size(1)).reshape(-1, specs["NumAtcParts"])
                instance_idx = instance_idx.repeat(1, all_sdf_data[0].size(1)).reshape(-1, 1)
                num_sdf_samples = sdf_data.shape[0]
                sdf_data[0].requires_grad = False
                sdf_data[1].requires_grad = False
                xyz = sdf_data[:, 0:3].float()
                sdf_gt = sdf_data[:, 3].unsqueeze(1)
                part_gt = sdf_data[:, 4].unsqueeze(1).long()

            else:
                sdf_data = all_sdf_data.reshape(-1, 5)
                num_sdf_samples = sdf_data.shape[0]
                sdf_data.requires_grad = False
                xyz = sdf_data[:, 0:3].float()
                sdf_gt = sdf_data[:, 3].unsqueeze(1)
                part_gt = sdf_data[:, 4].unsqueeze(1).long()

            xyz = torch.chunk(xyz, batch_split)

            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            sdf_gt = torch.chunk(sdf_gt, batch_split)
            part_gt = torch.chunk(part_gt, batch_split)

            if specs["Articulation"]==True:
                atc = torch.chunk(atc, batch_split)
                instance_idx = torch.chunk(instance_idx, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                if specs["Articulation"]==True:
                    batch_vecs = lat_vecs(instance_idx[i].view(-1)-1)
                else:
                    batch_vecs = lat_vecs(indices[i])

                # NN optimization
                if specs["Articulation"]==True:
                    input = torch.cat([batch_vecs, xyz[i], atc[i]], dim=1)
                else:
                    input = torch.cat([batch_vecs, xyz[i]], dim=1)

                if do_sup_with_part:
                    pred_sdf, pred_part = decoder(input)
                else:
                    pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                if do_sup_with_part:
                    part_loss = F.cross_entropy(pred_part, part_gt[i].view(-1).cuda())
                    part_loss *= 1e-3
                    chunk_loss = chunk_loss + part_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            if do_sup_with_part:
                print(batch_loss, part_loss.item(), reg_loss.item(), pred_sdf.min(), pred_sdf.max())
            else:
                print(batch_loss, reg_loss.item(), pred_sdf.min(), pred_sdf.max())
            loss_log.append(batch_loss)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                ws,
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                epoch,
            )


if __name__ == "__main__":


    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    asdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    asdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))

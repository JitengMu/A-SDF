#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import time
import torch
from torch.nn import functional as F
import numpy as np

import asdf
import asdf.workspace as ws
from asdf.asdf_reconstruct import *

def test(args, ws, specs):

    # init model
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    #decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"], articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], do_sup_with_part=specs["TrainWithParts"]).cuda()
    decoder = arch.Decoder(latent_size, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], do_sup_with_part=specs["TrainWithParts"]).cuda()
    decoder = torch.nn.DataParallel(decoder)
    decoder.eval()

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    dataset_name=args.dataset
    test_split_file = specs["TestSplit"]
    # test objs
    if args.mode=='recon_testset':
        # test files
        with open(test_split_file, "r") as f:
            split = json.load(f)
        npz_filenames = asdf.data.get_instance_filenames(args.data_source, split)
        # reconstruct test files
        reconstruct_testset(args, ws, specs, decoder, npz_filenames, saved_model_epoch, dataset_name)

    elif args.mode=='recon_testset_ttt':
        # test files
        with open(test_split_file, "r") as f:
            split = json.load(f)
        npz_filenames = asdf.data.get_instance_filenames(args.data_source, split)
        # reconstruct test files
        reconstruct_testset_ttt(args, ws, specs, decoder, npz_filenames, saved_model_state, dataset_name)
    
    elif args.mode=='inter_testset':
        # test files
        reconstruction_dir = os.path.join(
            args.experiment_directory, ws.recon_testset_subdir, str(saved_model_epoch)
        )

        if not os.path.isdir(reconstruction_dir):
            raise Exception("Testset reconstruction results is required")

        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_subdir
        )
        if not os.path.isdir(reconstruction_codes_dir):
            raise Exception("Testset reconstruction codes is required")

        # reconstruct test files
        interpolate_testset(args, ws, specs, decoder, reconstruction_codes_dir, saved_model_epoch, dataset_name)
    
    elif args.mode=='generation':
        # test files
        reconstruction_dir = os.path.join(
            args.experiment_directory, ws.recon_testset_subdir, str(saved_model_epoch)
        )

        if not os.path.isdir(reconstruction_dir):
            raise Exception("Testset reconstruction results is required")

        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_subdir
        )
        if not os.path.isdir(reconstruction_codes_dir):
            raise Exception("Testset reconstruction codes is required")

        # reconstruct test files
        generation(args, ws, specs, decoder, reconstruction_codes_dir, saved_model_epoch, dataset_name)
    
    elif args.mode=='generation_ttt':
        # test files
        reconstruction_dir = os.path.join(
            args.experiment_directory, ws.recon_testset_ttt_subdir, str(saved_model_epoch)
        )

        if not os.path.isdir(reconstruction_dir):
            raise Exception("Testset reconstruction results is required")

        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_subdir
        )
        if not os.path.isdir(reconstruction_codes_dir):
            raise Exception("Testset reconstruction codes is required")

        reconstruction_models_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_models_subdir
        )
        if not os.path.isdir(reconstruction_models_dir):
            raise Exception("Testset reconstruction models is required")

        # reconstruct test files
        generation_ttt(args, ws, specs, decoder, reconstruction_codes_dir, reconstruction_models_dir, saved_model_epoch, dataset_name)
 
    else:
        raise Exception("Unknown mode")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        required=True,
        help="choose from recon_testset(_ttt) | inter_testset | generation(_ttt)",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="data",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--infer_with_gt_atc",
        dest="infer_with_gt_atc",
        action="store_true",
        help="Infer with ground truth aticulations.",
    )
    arg_parser.add_argument(
        "--dataset",
        dest="dataset",
        default="shape2motion",
        help="shape2motion/shape2motion-1-view/real(for laptop only)",
    )
    
    # args and specs
    asdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    asdf.configure_logging(args)
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))

    test(args, ws, specs)
    

    
    




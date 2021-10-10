#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
import csv

import asdf
import asdf.workspace as ws
import glob
import re


def compute_chamfer_distance(chamfer_dist_file):
    chamfer_distance = []
    with open(chamfer_dist_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for idx, row in enumerate(spamreader):
            if idx>0:
                chamfer_distance.append(float(row[-1]))
    print("avg chamfer distance: ", np.mean(np.array(chamfer_distance)))


def evaluate(experiment_directory, checkpoint, data_dir, mode, specs):

    with open(specs["TestSplit"], "r") as f:
        split = json.load(f)

    chamfer_results = []

    for dataset in split:
        for class_name in split[dataset]:
            if mode=='recon_testset':
                all_names = sorted(glob.glob(os.path.join( experiment_directory, 'Results_recon_testset', checkpoint, 'Meshes', dataset, '*.ply' )))
            elif mode=='recon_testset_ttt':
                all_names = sorted(glob.glob(os.path.join( experiment_directory, 'Results_recon_testset_ttt', checkpoint, 'Meshes', dataset, '*.ply' )))
            elif mode=='generation':
                all_names = sorted(glob.glob(os.path.join( experiment_directory, 'Results_generation', checkpoint, 'Meshes', dataset, '*.ply' )))
            elif mode=='generation_ttt':
                all_names = sorted(glob.glob(os.path.join( experiment_directory, 'Results_generation_ttt', checkpoint, 'Meshes', dataset, '*.ply' )))
 
            logging.debug(
                'Num of files to be evaluated: {}'.format(len(all_names))
            )
            for instance_name in all_names:
                reconstructed_name = re.split('/', instance_name)[-1]
                if class_name=='laptop':
                    #instance_name = reconstructed_name[:7] + reconstructed_name[-8:-4]
                    if 'generation' in mode:
                        instance_name = glob.glob(os.path.join('data/SurfaceSamples/rbo/laptop/' , reconstructed_name[:4]+'*'+'{:04d}'.format(int(reconstructed_name[5:9])-20)+'.npz'))[0]
                        instance_name = re.split('/', instance_name)[-1][:-4]
                    else:
                        instance_name = reconstructed_name[:-4]
                else:
                    raise Exception("no such class")

                logging.debug(
                    "evaluating " + os.path.join(dataset, class_name, reconstructed_name)
                )
                
                if mode=='recon_testset':
                    reconstructed_mesh_filename = ws.get_recon_testset_mesh_filename(
                        experiment_directory, checkpoint, dataset, class_name, instance_name
                    )
                elif mode=='recon_testset_ttt':
                    reconstructed_mesh_filename = ws.get_recon_testset_ttt_mesh_filename(
                        experiment_directory, checkpoint, dataset, class_name, instance_name
                    )
                elif mode=='generation':
                    reconstructed_mesh_filename = ws.get_generation_mesh_filename(
                        experiment_directory, checkpoint, dataset, class_name, reconstructed_name[:-4]
                    )
                elif mode=='generation_ttt':
                    reconstructed_mesh_filename = ws.get_generation_ttt_mesh_filename(
                        experiment_directory, checkpoint, dataset, class_name, reconstructed_name[:-4]
                    )

                logging.debug(
                    'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
                )

                ground_truth_samples_filename = os.path.join(
                    data_dir,
                    "SurfaceSamples",
                    dataset,
                    class_name,
                    instance_name + ".npz",
                )

                logging.debug(
                    "ground truth samples are " + ground_truth_samples_filename
                )


                #ground_truth_points = trimesh.load(ground_truth_samples_filename)
                ground_truth_points = np.load(ground_truth_samples_filename)
                ground_truth_points = ground_truth_points["pos"]
                reconstruction = trimesh.load(reconstructed_mesh_filename)

                chamfer_dist = asdf.metrics.chamfer.compute_depth_chamfer(
                    ground_truth_points,
                    reconstruction,
                )

                logging.debug("chamfer distance: " + str(chamfer_dist))

                chamfer_results.append(
                    (os.path.join(dataset, class_name, instance_name), chamfer_dist)
                )

    chamfer_dist_file = os.path.join(ws.get_evaluation_dir(experiment_directory+'/Evaluation_with_Latent/', checkpoint, True), "chamfer.csv")
        
    with open(os.path.join(chamfer_dist_file),"w") as f:
        f.write("shape, chamfer_dist\n")
        for result in chamfer_results:
            f.write("{}, {}\n".format(result[0], result[1]))
    
    return chamfer_dist_file


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="data",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        required=True,
        help="choose from recon_testset | inter_testset | genration",
    )

    asdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    asdf.configure_logging(args)
    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))

    chamfer_dist_file = evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.mode,
        specs,
    )
    
    compute_chamfer_distance(chamfer_dist_file)

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
recon_testset_subdir = "Results_recon_testset"
inter_testset_subdir = "Results_inter_testset"
generation_subdir = "Results_generation"
recon_testset_ttt_subdir = "Results_recon_testset_ttt"
inter_testset_ttt_subdir = "Results_inter_testset_ttt"
generation_ttt_subdir = "Results_generation_ttt"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
reconstruction_models_subdir = "Models"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return decoder, data["epoch"]

# def build_decoder(experiment_directory, experiment_specs):

#     arch = __import__(
#         "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
#     )

#     latent_size = experiment_specs["CodeLength"]

#     decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

#     return decoder


# def load_decoder(
#     experiment_directory, experiment_specs, checkpoint, data_parallel=True
# ):

#     decoder = build_decoder(experiment_directory, experiment_specs)

#     if data_parallel:
#         decoder = torch.nn.DataParallel(decoder)

#     epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

#     return (decoder, epoch)


#def load_latent_vectors(experiment_directory, checkpoint):
#
#    filename = os.path.join(
#        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
#    )
#
#    if not os.path.isfile(filename):
#        raise Exception(
#            "The experiment directory ({}) does not include a latent code file"
#            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
#        )
#
#    data = torch.load(filename)
#
#    if isinstance(data["latent_codes"], torch.Tensor):
#
#        num_vecs = data["latent_codes"].size()[0]
#
#        lat_vecs = []
#        for i in range(num_vecs):
#            lat_vecs.append(data["latent_codes"][i].cuda())
#
#        return lat_vecs
#
#    else:
#
#        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
#
#        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
#
#        lat_vecs.load_state_dict(data["latent_codes"])
#
#        return lat_vecs.weight.data.detach()


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_recon_testset_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_recon_testset_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )



def get_recon_testset_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        recon_testset_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_inter_testset_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_inter_testset_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )


def get_inter_testset_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        inter_testset_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_generation_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )

def get_generation_ttt_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_ttt_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        instance_name + ".ply",
    )


def get_generation_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        generation_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )

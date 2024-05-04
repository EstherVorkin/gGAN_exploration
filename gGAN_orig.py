"""Main function of gGAN for the paper: Foreseeing Brain Graph Evolution Over Time
Using Deep Adversarial Network Normalizer
    Details can be found in: (https://arxiv.org/abs/2009.11166)
    (1) the original paper .
    ---------------------------------------------------------------------
    This file contains the implementation of two key steps of our gGAN framework:
        netNorm(v, nbr_of_sub, nbr_of_regions)
                Inputs:
                        v: (n × t x t) matrix stacking the source graphs of all subjects
                            n the total number of subjects
                            t number of regions
                Output:
                        CBT: (t x t) matrix representing the connectional brain template

        gGAN(sourceGraph, nbr_of_regions, nbr_of_folds, nbr_of_epochs, hyper_param1, CBT)
                Inputs:
                        sourceGraph: (n × t x t) matrix stacking the source graphs of all subjects
                                     n the total number of subjects
                                     t number of regions
                        CBT: (t x t) matrix stacking the connectional brain template generated by netNorm

                Output:
                        translatedGraph: (t x t) matrix stacking the graph translated into CBT

    This code has been slightly modified to be compatible across all PyTorch versions.

    (2) Dependencies: please install the following libraries:
        - matplotlib
        - numpy
        - scikitlearn
        - pytorch
        - pytorch-geometric
        - pytorch-scatter
        - pytorch-sparse
        - scipy

    ---------------------------------------------------------------------
    Copyright 2020 ().
    Please cite the above paper if you use this code.
    All rights reserved.
    """


# If you are using Google Colab please uncomment the three following lines.
# !pip install torch_geometric
# !pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# !pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html


import argparse
import os
import pdb
import numpy as np
import math
import itertools
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm, EdgePooling, TopKPooling, global_add_pool
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import random

import seaborn as sns

torch.cuda.empty_cache()
torch.cuda.empty_cache()

# random seed
manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('running on GPU')
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

else:
    device = torch.device("cpu")
    print('running on CPU')

def netNorm(v, nbr_of_sub, nbr_of_regions):
    nbr_of_feat = int((np.square(nbr_of_regions) - nbr_of_regions) / 2)

    def upper_triangular():
        All_subj = np.zeros((nbr_of_sub, nbr_of_feat))
        for j in range(nbr_of_sub):
            subj_x = v[j, :, :]
            subj_x = np.reshape(subj_x, (nbr_of_regions, nbr_of_regions))
            subj_x = subj_x[np.triu_indices(nbr_of_regions, k=1)]
            subj_x = np.reshape(subj_x, (1, nbr_of_feat))
            All_subj[j, :] = subj_x

        return All_subj

    def distances_inter(All_subj):
        theta = 0
        distance_vector = np.zeros(1)
        distance_vector_final = np.zeros(1)
        x = All_subj
        for i in range(nbr_of_feat):
            ROI_i = x[:, i]
            for j in range(nbr_of_sub):
                subj_j = ROI_i[j:j+1]

                distance_euclidienne_sub_j_sub_k = 0
                for k in range(nbr_of_sub):
                    if k != j:
                        subj_k = ROI_i[k:k+1]

                        distance_euclidienne_sub_j_sub_k = distance_euclidienne_sub_j_sub_k + np.square(subj_k - subj_j)
                        theta +=1
                if j == 0:
                    distance_vector = np.sqrt(distance_euclidienne_sub_j_sub_k)
                else:
                    distance_vector = np.concatenate((distance_vector, np.sqrt(distance_euclidienne_sub_j_sub_k)), axis=0)

            print("Distance Vector Shape:", distance_vector.shape)
            print("Distance Vector Final Shape:", distance_vector_final.shape)

            # Ensure both arrays are 2D
            distance_vector_final = np.atleast_2d(distance_vector_final)
            distance_vector = np.atleast_2d(distance_vector)
            # Ensure both arrays are 2D
            distance_vector = np.atleast_2d(distance_vector)
            if distance_vector_final.shape[0] != distance_vector.shape[0]:
                # Expand distance_vector_final to match the number of rows in distance_vector
                distance_vector_final = np.tile(distance_vector_final, (distance_vector.shape[0], 1))

            # Now concatenate along axis=1
            distance_vector_final = np.concatenate((distance_vector_final, distance_vector), axis=1)

        print(theta)
        return distance_vector_final


    def minimum_distances(distance_vector_final):
        if distance_vector_final.ndim == 1:
            distance_vector_final = distance_vector_final.reshape(1, -1)  # Ensure it is at least 2D

        nbr_of_sub = distance_vector_final.shape[0]
        nbr_of_feat = distance_vector_final.shape[1]

        for i in range(nbr_of_feat):
            minimum_sub = distance_vector_final[0, i]
            general_minimum = 0
            for k in range(1, nbr_of_sub):
                local_sub = distance_vector_final[k, i]
                if local_sub < minimum_sub:
                    general_minimum = k
                    minimum_sub = local_sub
            if i == 0:
                final_general_minimum = np.array([general_minimum])
            else:
                final_general_minimum = np.vstack((final_general_minimum, general_minimum))

        return final_general_minimum.T  # Transpose to ensure proper shape

    def new_tensor(final_general_minimum, All_subj):
        y = All_subj
        x = final_general_minimum
        for i in range(nbr_of_feat):
            optimal_subj = x[:, i:i+1]
            optimal_subj = np.reshape(optimal_subj, (1))
            optimal_subj = int(optimal_subj)
            if i == 0:
                final_new_tensor = y[optimal_subj: optimal_subj+1, i:i+1]
            else:
                final_new_tensor = np.concatenate((final_new_tensor, y[optimal_subj: optimal_subj+1, i:i+1]), axis=1)

        return final_new_tensor

    def make_sym_matrix(nbr_of_regions, feature_vector):
        my_matrix = np.zeros([nbr_of_regions, nbr_of_regions], dtype=np.double)

        my_matrix[np.triu_indices(nbr_of_regions, k=1)] = feature_vector
        my_matrix = my_matrix + my_matrix.T
        my_matrix[np.diag_indices(nbr_of_regions)] = 0

        return my_matrix

    def re_make_tensor(final_new_tensor, nbr_of_regions):
        x = final_new_tensor
        #x = np.reshape(x, (nbr_of_views, nbr_of_feat))

        x = make_sym_matrix(nbr_of_regions, x)
        x = np.reshape(x, (1, nbr_of_regions, nbr_of_regions))

        return x

    Upp_trig = upper_triangular()
    Dis_int = distances_inter(Upp_trig)
    Min_dis = minimum_distances(Dis_int)
    New_ten = new_tensor(Min_dis, Upp_trig)
    Re_ten = re_make_tensor(New_ten, nbr_of_regions)
    Re_ten = np.reshape(Re_ten, (nbr_of_regions, nbr_of_regions))
    np.fill_diagonal(Re_ten, 0)
    network = np.array(Re_ten)
    return network

def gGAN(data, nbr_of_regions, nbr_of_epochs, nbr_of_folds, hyper_param1, CBT):
    def cast_data(array_of_tensors, version):
        version1 = torch.tensor(version, dtype=torch.int)

        N_ROI = array_of_tensors[0].shape[0]
        CHANNELS = 1
        dataset = []
        edge_index = torch.zeros(2, N_ROI * N_ROI)
        edge_attr = torch.zeros(N_ROI * N_ROI, CHANNELS)
        x = torch.zeros((N_ROI, N_ROI))  # 35 x 35
        y = torch.zeros((1,))

        counter = 0
        for i in range(N_ROI):
            for j in range(N_ROI):
                edge_index[:, counter] = torch.tensor([i, j])
                counter += 1
        for mat in array_of_tensors:  #1,35,35,4

            if version1 == 0:
                edge_attr = mat.view((nbr_of_regions*nbr_of_regions), 1)
                x = mat.view(nbr_of_regions, nbr_of_regions)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                x = torch.tensor(x, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                dataset.append(data)

            elif version1 == 1:
                edge_attr = torch.randn(N_ROI * N_ROI, CHANNELS)
                x = torch.randn(N_ROI, N_ROI)  # 35 x 35
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                x = torch.tensor(x, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                dataset.append(data)

        return dataset

    # ------------------------------------------------------------

    def plotting_loss(losses_generator, losses_discriminator, epoch):
        plt.figure(1)
        plt.plot(epoch, losses_generator, 'r-')
        plt.plot(epoch, losses_discriminator, 'b-')
        plt.legend(['G Loss', 'D Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.savefig('./plot/loss' + str(epoch) + '.png')
        output_directory = '/content/drive/My Drive/gGAN_project/orig_data/'  
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_path = os.path.join(output_directory, 'orig_Data_loss' + str(epoch) + '.png')
        plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)

    # -------------------------------------------------------------

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            nn = Sequential(Linear(1, (nbr_of_regions*nbr_of_regions)), ReLU())
            self.conv1 = NNConv(nbr_of_regions, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
            self.conv11 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, nbr_of_regions), ReLU())
            self.conv2 = NNConv(nbr_of_regions, 1, nn, aggr='mean', root_weight=True, bias=True)
            self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, nbr_of_regions), ReLU())
            self.conv3 = NNConv(1, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
            self.conv33 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
            x1 = F.dropout(x1, training=self.training)

            x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
            x2 = F.dropout(x2, training=self.training)

            x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
            x4 = x3[:, 0:nbr_of_regions]
            x5 = x3[:, nbr_of_regions:2*nbr_of_regions]

            x6 = (x4 + x5) / 2
            return x6

    class Discriminator1(torch.nn.Module):
        def __init__(self):
            super(Discriminator1, self).__init__()
            nn = Sequential(Linear(2, (nbr_of_regions*nbr_of_regions)), ReLU())
            self.conv1 = NNConv(nbr_of_regions, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
            self.conv11 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(2, nbr_of_regions), ReLU())
            self.conv2 = NNConv(nbr_of_regions, 1, nn, aggr='mean', root_weight=True, bias=True)
            self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


        def forward(self, data, data_to_translate):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr_data_to_translate = data_to_translate.edge_attr

            edge_attr_data_to_translate_reshaped = edge_attr_data_to_translate.view(nbr_of_regions*nbr_of_regions, 1)

            gen_input = torch.cat((edge_attr, edge_attr_data_to_translate_reshaped), -1)
            x = F.relu(self.conv11(self.conv1(x, edge_index, gen_input)))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv22(self.conv2(x, edge_index, gen_input)))

            return F.sigmoid(x)

    # ----------------------------------------
    #                Training
    # ----------------------------------------

    n_fold_counter = 1
    plot_loss_g = np.empty((nbr_of_epochs), dtype=float)
    plot_loss_d = np.empty((nbr_of_epochs), dtype=float)

    kfold = KFold(n_splits=nbr_of_folds, shuffle=True, random_state=manualSeed)

    source_data = torch.from_numpy(data)  # convert numpy array to torch tensor
    source_data = source_data.type(torch.FloatTensor)

    target_data = np.reshape(CBT, (1, nbr_of_regions, nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)

    for train, test in kfold.split(source_data):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        # Initialize generator and discriminator
        generator = Generator()
        discriminator1 = Discriminator1()

        generator.to(device)
        discriminator1.to(device)
        adversarial_loss.to(device)
        l1_loss.to(device)

        # Optimizers
        optimizer_G = torch.optim.AdamW(generator.parameters(), lr=0.005, betas=(0.5, 0.999))
        optimizer_D = torch.optim.AdamW(discriminator1.parameters(), lr=0.01, betas=(0.5, 0.999))

        # ------------------------------- select source data and target data -------------------------------

        train_source, test_source = source_data[train], source_data[test]  ## from a specific source view

        # 1: everything random; 0: everything is the matrix in question

        train_casted_source = [d.to(device) for d in cast_data(train_source, 0)]
        train_casted_target = [d.to(device) for d in cast_data(target_data, 0)]

        for epoch in range(nbr_of_epochs):
            # Train Generator
            with torch.autograd.set_detect_anomaly(True):

                losses_generator = []
                losses_discriminator = []

                for data_A in train_casted_source:
                    generators_output_ = generator(data_A)  # 35 x35
                    generators_output = generators_output_.view(1, nbr_of_regions, nbr_of_regions, 1).type(torch.FloatTensor)

                    generators_output_casted = [d.to(device) for d in cast_data(generators_output, 0)]
                    for (data_discriminator) in generators_output_casted:
                        discriminator_output_of_gen = discriminator1(data_discriminator, data_A).to(device)

                        g_loss_adversarial = adversarial_loss(discriminator_output_of_gen, torch.ones_like(discriminator_output_of_gen))

                        g_loss_pix2pix = l1_loss(generators_output_, train_casted_target[0].edge_attr.view(nbr_of_regions, nbr_of_regions))

                        g_loss = g_loss_adversarial + (hyper_param1 * g_loss_pix2pix)
                        losses_generator.append(g_loss)

                        discriminator_output_for_real_loss = discriminator1(data_A, train_casted_target[0])

                        real_loss = adversarial_loss(discriminator_output_for_real_loss,
                                                     (torch.ones_like(discriminator_output_for_real_loss, requires_grad=False)))
                        fake_loss = adversarial_loss(discriminator_output_of_gen.detach(), torch.zeros_like(discriminator_output_of_gen))

                        d_loss = (real_loss + fake_loss) / 2
                        losses_discriminator.append(d_loss)

                optimizer_G.zero_grad()
                losses_generator = torch.mean(torch.stack(losses_generator))
                losses_generator.backward(retain_graph=True)
                optimizer_G.step()

                optimizer_D.zero_grad()
                losses_discriminator = torch.mean(torch.stack(losses_discriminator))

                losses_discriminator.backward(retain_graph=True)
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, nbr_of_epochs, losses_discriminator, losses_generator))

                plot_loss_g[epoch] = losses_generator.detach().cpu().clone().numpy()
                plot_loss_d[epoch] = losses_discriminator.detach().cpu().clone().numpy()

                torch.save(generator.state_dict(), "./weight_" + str(n_fold_counter) + "generator" + "_" + ".model")
                torch.save(discriminator1.state_dict(), "./weight_" + str(n_fold_counter) + "dicriminator" + "_" + ".model")

        interval = range(0, nbr_of_epochs)
        plotting_loss(plot_loss_g, plot_loss_d, interval)
        n_fold_counter += 1
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


nbr_of_sub = int(input('Please select the number of subjects: '))
if nbr_of_sub < 5:
    print("You can not give less than 5 to the number of subjects. ")
    nbr_of_sub = int(input('Please select the number of subjects: '))
nbr_of_sub_for_cbt = int(input('Please select the number of subjects to generate the CBT: '))
nbr_of_regions = int(input('Please select the number of regions: '))
nbr_of_epochs = int(input('Please select the number of epochs: '))
nbr_of_folds = int(input('Please select the number of folds: '))
hyper_param1 = 100
nbr_of_feat = int((np.square(nbr_of_regions) - nbr_of_regions) / 2)

data = np.random.normal(0.6, 0.3, (nbr_of_sub, nbr_of_regions, nbr_of_regions))
data = np.abs(data)
independent_data = np.random.normal(0.6, 0.3, (nbr_of_sub_for_cbt, nbr_of_regions, nbr_of_regions))
independent_data = np.abs(independent_data)
CBT = netNorm(independent_data, nbr_of_sub_for_cbt, nbr_of_regions)
gGAN(data, nbr_of_regions, nbr_of_epochs, nbr_of_folds, hyper_param1, CBT)

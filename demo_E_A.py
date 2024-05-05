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
import pandas as pd
import seaborn as sns
import random
from gGAN import gGAN, netNorm
import pydicom

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


def demo():
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
        for mat in array_of_tensors:  # 1,35,35,4

            if version1 == 0:
                edge_attr = mat.view(1225, 1)
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

    #####################################################################################################
    def find_dicom_files(root_dir):
        """Recursively find all DICOM files in the specified directory and its subdirectories."""
        dicom_files = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.dcm'):
                    full_path = os.path.join(subdir, file)
                    dicom_files.append(full_path)
        return dicom_files

    def load_dicom_image(dicom_path):
        """Load DICOM image and return its pixel data as a numpy array."""
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(float)
        image -= np.min(image)
        image /= np.max(image)  # Normalize to [0, 1]
        return image

    def image_to_graph(image, num_regions):
        """Convert an image to a graph. Here, this function needs to be defined based on your use case.
        For demonstration, let's assume it returns a matrix num_regions x num_regions with random connectivity."""
        graph = np.random.rand(num_regions, num_regions)
        graph = (graph + graph.T) / 2  # Make it symmetric
        return graph

    def prepare_data_from_dicom(dicom_folder, num_subjects, num_regions):
        dicom_files = find_dicom_files(dicom_folder)
        dicom_files = dicom_files[:num_subjects]  # Limit the number of subjects to num_subjects
        data = np.array([image_to_graph(load_dicom_image(f), num_regions) for f in dicom_files])
        return data
    
    def linear_features(data):
        n_roi = data[0].shape[0]
        n_sub = data.shape[0]
        counter = 0

        num_feat = (n_roi * (n_roi - 1) // 2)
        final_data = np.empty([n_sub, num_feat], dtype=float)
        for k in range(n_sub):
            for i in range(n_roi):
                for j in range(i+1, n_roi):
                    final_data[k, counter] = data[k, i, j]
                    counter += 1
            counter = 0

        return final_data

    def make_sym_matrix(nbr_of_regions, feature_vector):
        sym_matrix = np.zeros([9, feature_vector.shape[1], nbr_of_regions, nbr_of_regions], dtype=np.double)
        for j in range(9):
            for i in range(feature_vector.shape[1]):
                my_matrix = np.zeros([nbr_of_regions, nbr_of_regions], dtype=np.double)

                my_matrix[np.triu_indices(nbr_of_regions, k=1)] = feature_vector[j, i, :]
                my_matrix = my_matrix + my_matrix.T
                my_matrix[np.diag_indices(nbr_of_regions)] = 0
                sym_matrix[j, i,:,:] = my_matrix

        return sym_matrix

    def plot_predictions(predicted, fold):
        plt.clf()
        for j in range(predicted.shape[0]):
            for i in range(predicted.shape[1]):
                predicted_sub = predicted[j, i, :, :]
                plt.pcolor(abs(predicted_sub))
                if(j == 0 and i == 0):
                    plt.colorbar()
                plt.imshow(predicted_sub)
                #plt.savefig('./plot/img' + str(fold) + str(j) + str(i) + '.png')
                output_directory = '/content/drive/My Drive/gGAN_project/data_output_5_5_24/new/'  
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                file_path = os.path.join(output_directory, 'ourData_img' + str(fold) + str(j) + str(i) + '.png')
                plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)

    def plot_MAE(prediction, data_next, test, fold):
        MAE = np.zeros((9), dtype=np.double)
        valid_indices = test[test < len(data_next)]

        for i in range(9):
            if len(valid_indices) > 0:
                prediction_valid = prediction[i, :len(valid_indices), :]
                MAE_i = np.abs(prediction_valid - data_next[valid_indices])
                MAE[i] = np.mean(MAE_i)
            else:
                MAE[i] = np.nan

        plt.clf()
        k = ['k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9', 'k=10']
        sns.set(style="whitegrid")
        df = pd.DataFrame(dict(x=k, y=MAE))
        ax = sns.barplot(x="x", y="y", data=df)
        min_val = np.nanmin(MAE) - 0.01 if not np.isnan(np.nanmin(MAE)) else 0
        max_val = np.nanmax(MAE) + 0.01 if not np.isnan(np.nanmax(MAE)) else 1
        ax.set(ylim=(min_val, max_val))

        output_directory = '/content/drive/My Drive/gGAN_project/data_output_5_5_24/new/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_path = os.path.join(output_directory, 'mae_updated' + str(fold) + '.png')
        plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)

    ######################################################################################################################################

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            nn = Sequential(Linear(1, 1225), ReLU())
            self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
            self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, 35), ReLU())
            self.conv2 = NNConv(35, 1, nn, aggr='mean', root_weight=True, bias=True)
            self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, 35), ReLU())
            self.conv3 = NNConv(1, 35, nn, aggr='mean', root_weight=True, bias=True)
            self.conv33 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)



        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
            x1 = F.dropout(x1, training=self.training)

            x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
            x2 = F.dropout(x2, training=self.training)

            embedded = x2.detach().cpu().clone().numpy()

            return embedded

    def embed(Casted_source):
        embedded_data = np.zeros((1, 35), dtype=float)
        i = 0
        for data_A in Casted_source:  ## take a subject from source and target data
            embedded = generator(data_A)  # 35 x35

            if i == 0:
                embedded = np.transpose(embedded)
                embedded_data = embedded
            else:
                embedded = np.transpose(embedded)
                embedded_data = np.append(embedded_data, embedded, axis=0)
            i = i + 1
        return embedded_data

    def test_gGAN(data_next, embedded_train_data, embedded_test_data, embedded_CBT):
        def x_to_x(x_train, x_test, nbr_of_trn, nbr_of_tst):
            result = np.empty((nbr_of_tst, nbr_of_trn), dtype=float)
            for i in range(nbr_of_tst):
                x_t = np.transpose(x_test[i])
                for j in range(nbr_of_trn):
                    result[i, j] = np.matmul(x_train[j], x_t)
            return result

        def check(neighbors, i, j):
            for val in neighbors[i, :]:
                if val == j:
                    return 1
            return 0

        def k_neighbors(x_to_x, k_num, nbr_of_trn, nbr_of_tst):
            neighbors = np.zeros((nbr_of_tst, k_num), dtype=int)
            current = 0
            for i in range(nbr_of_tst):
                for k in range(k_num):
                    for j in range(nbr_of_trn):
                        if abs(x_to_x[i, j]) > current:
                            if check(neighbors, i, j) == 0:
                                neighbors[i, k] = j
                                current = abs(x_to_x[i, neighbors[i, k]])
                    current = 0

            return neighbors

        def subtract_cbt(x, cbt, length):
            for i in range(length):
                x[i] = abs(x[i] - cbt[0])

            return x

        def predict_samples(k_neighbors, t1, nbr_of_tst):
            nbr_of_feat = t1.shape[1]
            average = np.zeros((nbr_of_tst, nbr_of_feat), dtype=float)
            for i in range(nbr_of_tst):
                for j in range(len(k_neighbors[i])):
                    if k_neighbors[i, j] < len(t1):
                        average[i] += t1[k_neighbors[i, j], :]

                average[i] /= len(k_neighbors[i])

            return average

        residual_of_tr_embeddings = subtract_cbt(embedded_train_data, embedded_CBT, len(embedded_train_data))
        residual_of_ts_embeddings = subtract_cbt(embedded_test_data, embedded_CBT, len(embedded_test_data))

        dot_of_residuals = x_to_x(residual_of_tr_embeddings, residual_of_ts_embeddings, len(embedded_train_data), len(embedded_test_data))

        predictions = []
        for k in range(2, 11):
            k_neighbors_ = k_neighbors(dot_of_residuals, k, len(embedded_train_data), len(embedded_test_data))
            prediction = predict_samples(k_neighbors_, data_next, len(embedded_test_data))
            prediction = np.reshape(prediction, (1, len(embedded_test_data), nbr_of_feat))
            predictions.append(prediction)

        predictions = np.concatenate(predictions, axis=0)
        return predictions

    
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

    #dicom_folder = '/content/drive/My Drive/gGAN_project/data_output'
    dicom_folder = '/content/netNorm-PY/Brain_dataset/'

    data = prepare_data_from_dicom(dicom_folder, nbr_of_sub, nbr_of_regions)
    data = np.abs(data)  # Ensure all values are positive, might depend on your preprocessing
    independent_data = data[:nbr_of_sub_for_cbt]
    data_next = data[nbr_of_sub_for_cbt:100]

    CBT = netNorm(independent_data, nbr_of_sub_for_cbt, nbr_of_regions)
    gGAN(data, nbr_of_regions, nbr_of_epochs, nbr_of_folds, hyper_param1, CBT)

    # embed train and test subjects
    kfold = KFold(n_splits=nbr_of_folds, shuffle=True, random_state=manualSeed)

    source_data = torch.from_numpy(data)  # convert numpy array to torch tensor
    source_data = source_data.type(torch.FloatTensor)

    target_data = np.reshape(CBT, (1, nbr_of_regions, nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)

    i = 1
    for train, test in kfold.split(source_data):
        adversarial_loss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        trained_model_gen = torch.load('./updated_weight_' + str(i) + 'generator_.model')
        generator = Generator()
        generator.load_state_dict(trained_model_gen)

        train_data = source_data[train]
        test_data = source_data[test]

        generator.to(device)
        adversarial_loss.to(device)
        l1_loss.to(device)

        X_train_casted_source = [d.to(device) for d in cast_data(train_data, 0)]
        X_test_casted_source = [d.to(device) for d in cast_data(test_data, 0)]
        data_B = [d.to(device) for d in cast_data(target_data, 0)]

        embedded_train_data = embed(X_train_casted_source)
        embedded_test_data = embed(X_test_casted_source)
        embedded_CBT = embed(data_B)

        if i == 1:
            data_next = linear_features(data_next)
        predicted_flat = test_gGAN(data_next, embedded_train_data, embedded_test_data, embedded_CBT)

        plot_MAE(predicted_flat, data_next, test, i)
        i = i + 1

        predicted = make_sym_matrix(nbr_of_regions, predicted_flat)
        #plot_predictions(predicted, i - 1)

demo()

import numpy as np
import torch
import math


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


def get_etf(feature_dim, classes):
    orth_vec = generate_random_orthogonal_matrix(feature_dim, classes)
    i_nc_nc = torch.eye(classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(classes, classes), (1 / classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(classes / (classes - 1)))
    etf_rect = torch.ones((1, classes), dtype=torch.float32)  # the length of etf point
    return etf_vec


def generate_designated_ortMat(designated_mat):
    num_classes = designated_mat.shape[1]
    designated_mat = designated_mat.cpu().numpy()
    orth_vec, _ = np.linalg.qr(designated_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


def design_etf(designated_mat):
    classes = designated_mat.shape[1]
    orth_vec = generate_designated_ortMat(designated_mat)
    i_nc_nc = torch.eye(classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(classes, classes), (1 / classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(classes / (classes - 1)))
    return etf_vec.to(designated_mat.device)

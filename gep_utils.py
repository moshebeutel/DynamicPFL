from typing import Optional, Tuple
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import Tensor
from abc import ABC, abstractmethod


# def flatten_tensor(tensor_list) -> torch.Tensor:
#
#     # for i in range(len(tensor_list)):
#     #     tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
#     #     # tensor_list[i] = tensor_list[i].reshape(1, -1)
#     flatten_param = torch.stack(tensor_list)
#     flatten_param = flatten_param.reshape(flatten_param.shape[0], -1)
#     return flatten_param

def flatten_tensor(tensor_list) -> torch.Tensor:
    """
    Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
    """
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
        # tensor_list[i] = tensor_list[i].reshape(1, -1)
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param

@torch.no_grad()
def check_approx_error(L, target) -> float:
    L = L.to(target.device)
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))

    return -1.0 if target.item() == 0 else error.item() / target.item()


class Subspace(ABC):

    @abstractmethod
    def _get_bases(self, pub_grad, num_bases):
        pass
    @abstractmethod
    def compute_subspace(self, basis_gradients: torch.Tensor, num_basis_elements: int):
        pass

    @abstractmethod
    def embed_grad(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def project_back_embedding(self, embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def explained_variance(self):
        pass

    @property
    @abstractmethod
    def explained_variance_ratio(self):
        pass

    @property
    @abstractmethod
    def num_components(self):
        pass

    @property
    @abstractmethod
    def components(self):
        pass

class SubspacePCA(Subspace):

    def __init__(self, group=0, center=False, scale=True):
        super().__init__()
        self._base_computed: bool = False
        self._pca: Optional[PCA] = None

        self._components: Optional[torch.Tensor] = None
        self._explained_variance: Optional[torch.Tensor] = None
        self._explained_variance_ratio: Optional[torch.Tensor] = None
        self._scale = scale
        self._center = center

        self._num_bases: int = 0
        self._group: int = group


    def _get_bases(self, pub_grad, num_bases):
        num_k = pub_grad.shape[0]
        num_p = pub_grad.shape[1]

        num_bases = min(num_bases, min(num_p, num_k))

        self._pca = PCA(n_components=num_bases, iterated_power=1, random_state=43)

        X = pub_grad.cpu().detach().numpy()

        # print(f'before scale X: {np.min(X), np.max(X), np.mean(X), np.std(X)}')
        if self._scale:
            self._scaler: Optional[MinMaxScaler] = MinMaxScaler()
            X = self._scaler.fit_transform(X)
            # print(f'after scale X: {np.min(X), np.max(X), np.mean(X), np.std(X)}')

        self._pca.fit(X)

        self._explained_variance = torch.from_numpy(self._pca.explained_variance_)
        self._explained_variance_ratio = torch.from_numpy(self._pca.explained_variance_ratio_)

        # print(f'explained variance: {self._explained_variance.shape} numpy {self._pca.explained_variance_.shape}')
        # print(f'explained variance ratio: {self._explained_variance_ratio.shape} numpy {self._pca.explained_variance_ratio_.shape}')


        self._components = torch.from_numpy(self._pca.components_)



    def compute_subspace(self, basis_gradients, num_basis_elements):
        num_bases: int
        pub_error: float
        self._get_bases(basis_gradients, num_basis_elements)
        self._base_computed = True

    def embed_grad(self, grad: torch.Tensor) -> torch.Tensor:
        assert self._base_computed, 'subspace needs to be computed first'
        grad_np: np.ndarray = grad.cpu().detach().numpy()

        if self._scale:
            assert self._scaler is not None, 'Expected scaler initialized'
            grad_np = self._scaler.transform(grad_np)
        embedding: np.ndarray = self._pca.transform(grad_np)
        return torch.from_numpy(embedding)

    def project_back_embedding(self, embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self._base_computed, 'subspace needs to be computed first'
        embedding_np: np.ndarray = embedding.cpu().detach().numpy()

        grad_np: np.ndarray = self._pca.inverse_transform(embedding_np)
        if self._scale:
            assert self._scaler is not None, 'Expected scaler initialized'
            grad_np = self._scaler.inverse_transform(grad_np)
        return torch.from_numpy(grad_np).to(device)


    @property
    def explained_variance(self):
        assert self._base_computed, 'subspace needs to be computed first'
        return self._explained_variance

    @property
    def explained_variance_ratio(self):
        assert self._base_computed, 'subspace needs to be computed first'
        return self._explained_variance_ratio

    @property
    def num_components(self):
        assert self._base_computed, 'subspace needs to be computed first'
        return self._num_bases

    @property
    def components(self):
        assert self._base_computed, f'Explained variance not computed yet'
        return self._components

    @property
    def group(self):
        return self._group

#  GEP UTILS  numpy variants
#  *************************
def get_bases(pub_grad, num_bases):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, min(num_p, num_k))

    pca = PCA(n_components=num_bases, iterated_power=1, random_state=43)

    X = pub_grad.cpu().detach().numpy()

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    pca.fit(X)

    return num_bases, (pca, scaler, pca.explained_variance_, pca.explained_variance_ratio_)


def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int) :
    num_bases: int
    pub_error: float
    # pca: PCA|Tuple[PCA, StandardScaler]
    num_bases, pca = get_bases(basis_gradients, num_basis_elements)
    # num_bases, pub_error, pca = get_bases(basis_gradients, num_basis_elements)
    return pca


def embed_grad(grad: torch.Tensor, pca) -> torch.Tensor:
    grad_np: np.ndarray = grad.cpu().detach().numpy()
    if isinstance(pca, tuple):
        assert len(pca) == 2, 'Expected PCA and StandardScaler'
        pca, scaler = pca
    else:
        assert isinstance(pca, PCA), 'Expected PCA'
        pca, scaler = pca, None
    if scaler is not None:
        grad_np = scaler.transform(grad_np)
    embedding: np.ndarray = pca.transform(grad_np)
    return torch.from_numpy(embedding)


def project_back_embedding(embedding: torch.Tensor, pca, device: torch.device) -> torch.Tensor:
    embedding_np: np.ndarray = embedding.cpu().detach().numpy()
    if isinstance(pca, tuple):
        assert len(pca) == 2, 'Expected PCA and StandardScaler'
        pca, scaler = pca
    else:
        assert isinstance(pca, PCA), 'Expected PCA'
        pca, scaler = pca, None
    grad_np: np.ndarray = pca.inverse_transform(embedding_np)
    if scaler is not None:
        grad_np = scaler.inverse_transform(grad_np)
    return torch.from_numpy(grad_np).to(device)
#  End of GEP UTILS  numpy variants
#  *************************


#  GEP UTILS  torch variants
#  *************************
# @torch.no_grad()
# def get_bases(pub_grad, num_bases) -> Tuple[int, torch.Tensor]:
#     num_k = pub_grad.shape[0]
#     num_p = pub_grad.shape[1]
#
#     num_bases = min(num_bases, min(num_p, num_k))
#
#     pca = torch.pca_lowrank(pub_grad, q=num_bases, niter=10)
#     # error_rate = check_approx_error(pca[-1], pub_grad)
#
#     # print(f'\n\t\t\t\t\t\t\t\tnum_bases {num_bases}\tPCA error: {error_rate}')
#
#     return num_bases, pca[-1]
#
#
# def embed_grad(grad: torch.Tensor, pca: torch.Tensor) -> torch.Tensor:
#     embedding: torch.Tensor = torch.matmul(grad, pca)
#     return embedding
#
#
# def project_back_embedding(embedding: torch.Tensor, pca: torch.Tensor, device) -> torch.Tensor:
#     reconstructed = torch.matmul(embedding, pca.t())
#     return reconstructed
#
#
# @torch.no_grad()
# def compute_subspace(basis_gradients: torch.Tensor, num_basis_elements: int) -> torch.Tensor:
#     pca: torch.Tensor
#     _, pca = get_bases(basis_gradients, num_basis_elements)
#     return pca

#  End of GEP UTILS  torch variants
#  *************************

@torch.no_grad()
def add_new_gradients_to_history(new_gradients: torch.Tensor, basis_gradients: Optional[torch.Tensor],
                                 gradients_history_size: int) -> Tensor:
    # print(f'\n\t\t\t\t\t\t\t\t1 - basis gradients shape {basis_gradients.shape if basis_gradients is not None else None}')

    basis_gradients = torch.cat((basis_gradients, new_gradients), dim=0) \
        if basis_gradients is not None \
        else new_gradients
    # print(f'\n\t\t\t\t\t\t\t\t2 - basis gradients shape {basis_gradients.shape}')

    basis_gradients = basis_gradients[-gradients_history_size:] \
        if gradients_history_size < basis_gradients.shape[0] \
        else basis_gradients

        # print(f'\n\t\t\t\t\t\t\t\t3 - basis gradients shape {basis_gradients.shape}')

    return basis_gradients




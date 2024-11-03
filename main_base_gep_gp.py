import copy
import gc
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader, default_collate
from torchvision.transforms import v2
from tqdm.auto import trange
import wandb
from data import get_mnist_datasets, get_clients_datasets, get_CIFAR10, get_SVHN
from emg_utils import get_dataloaders
# >>>  ***GEP
from gep_torch_utils import (compute_subspace, embed_grad, flatten_tensor,
                             project_back_embedding, add_new_gradients_to_history, flatten_tensor1)
from net import mnistNet, cifar10NetGPkernel, SVHNNet, EMGModel, save_checkpoint, load_checkpoint
from options import parse_args
from utils import compute_noise_multiplier
# <<< ***GEP
from pFedGP.pFedGP.Learner import pFedGPFullLearner
from gp_utils import build_tree


def norm_of_rows_squared(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 2, f'Expected a 2D Tensor, got {t.dim()}'
    m, n = t.shape
    # norms_vector = torch.norm(t, p=2, dim=1, keepdim=True)
    norms_sqaured_vector = torch.sum(t ** 2, dim=1, keepdim=True)
    assert norms_sqaured_vector.shape == (
        m, 1), f'Expected a 1D Column Tensor with {m} squared norm values, got {norms_sqaured_vector.shape}'
    return norms_sqaured_vector


def norm_of_rows(t: torch.Tensor) -> torch.Tensor:
    norm_squared = norm_of_rows_squared(t)
    return torch.sqrt(norm_squared)


def mean_norm_of_rows(t: torch.Tensor) -> float:
    norms_vector = norm_of_rows(t)
    return float(norms_vector.mean())


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
# >>>  ***GEP
num_public_clients = args.num_public_clients
num_virtual_public_clients = args.virtual_publics
virtual_ratio = num_virtual_public_clients // num_public_clients
num_basis_elements = args.basis_size
gradient_history_size = args.history_size
# <<< ***GEP
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
pub_batch_size = 64
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
clipping_bound_residual = args.clipping_bound_residual
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device('cuda'):
    current_device = torch.cuda.current_device()
    device = torch.device(f'cuda:{current_device}')
print(f'Device: {device}')
if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--user_sample_rate {args.user_sample_rate} '
        f'--local_epoch {local_epoch} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--clipping_bound {clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        , 'a'
    )
    sys.stdout = file

def filter_outliar_grads(grads: torch.Tensor, z_thresh: float = 3.0) -> Tensor:
    assert grads.dim() == 2, f'Expected a 2D Tensor, got {grads.dim()}'
    mean = torch.mean(grads)
    std = torch.std(grads)

    z_scores = (grads - mean) / std

    outlier_mask = (z_scores > z_thresh) | (z_scores < -z_thresh)

    return outlier_mask


def local_update(model, dataloader, cid, GPs):
    model.train()
    # model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # build tree at each step
    GPs[cid], label_map, _, _ = build_tree(model, dataloader, cid, GPs)
    GPs[cid].train()
    total_loss = 0.0
    for _ in range(local_epoch):
        optimizer.zero_grad()
        for k, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            # forward prop
            pred = model(data)

            X = torch.cat((X, pred), dim=0) if k > 0 else pred
            Y = torch.cat((Y, labels), dim=0) if k > 0 else labels

            del data, labels, pred

        offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype, device=Y.device)
        loss = GPs[cid](X, offset_labels, to_print=args.eval_every)
        # loss *= args.loss_scaler
        total_loss += float(loss)
        # propagate loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
        optimizer.step()
        del loss, offset_labels

    GPs[cid].tree = None
    gc.collect()
    return model.state_dict(), total_loss / local_epoch, GPs[cid]


def test(client_model, client_testloader, client_trainloader, cid, GPs):
    client_model.eval()
    # client_model = client_model.to(device)

    num_data = 0
    correct = 0
    is_first_iter = True
    running_loss = 0.

    # build tree at each step
    GPs[cid], label_map, Y_train, X_train = build_tree(client_model, client_trainloader, cid, GPs)
    GPs[cid].eval()

    with torch.no_grad():
        for data, labels in client_testloader:
            num_data += labels.size(0)
            img, label = data.to(device), labels.to(device)
            Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
                                  device=label.device)
            X_test = client_model(img)
            loss, pred = GPs[cid].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
            batch_size = Y_test.shape[0]
            running_loss += (loss.item() * batch_size)
            correct += pred.argmax(1).eq(Y_test).sum().item()
            is_first_iter = False
    accuracy = 100.0 * correct / num_data
    # client_model = client_model.to('cpu')
    return accuracy


class AugStackTransform(v2.Transform):
    def __init__(self, multiplicity: int):
        super().__init__()
        assert 0 < multiplicity < 201, f'multiplicity {multiplicity} is out of range'

        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        self.multiple_augments = torch.nn.ModuleList([cutmix_or_mixup for i in range(multiplicity)])
        self._multiplicity = multiplicity

    def forward(self, batch):
        batch = [aug(batch) for aug in self.multiple_augments]
        images = torch.cat([b[0] for b in batch], dim=0)
        labels = torch.cat([b[1] for b in batch], dim=0).long()
        return images, labels


def main():
    best_acc = 0.0
    mean_acc_s = []
    acc_matrix = []
    classes_per_client = 0
    if dataset == 'MNIST':
        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
                                 clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]
        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
        classes_per_client = 10
    elif dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients,
                                                                                     args.batch_size)
        # clients_models = [cifar10NetGPkernel() for _ in range(num_clients)]
        global_model = cifar10NetGPkernel()
        classes_per_client = 10
    # elif dataset == 'FEMNIST':
    #     clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
    #     clients_models = [femnistNet() for _ in range(num_clients)]
    #     global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients,
                                                                                  args.batch_size)
        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
        classes_per_client = 62
    elif dataset == 'putEMG':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_dataloaders()
        clients_models = [EMGModel(num_features=24 * 8, num_classes=8, use_softmax=True) for _ in range(num_clients)]
        global_model = EMGModel(num_features=24 * 8, num_classes=8, use_softmax=True)
        classes_per_client = 8
    else:
        print('undefined dataset')
        assert 1 == 0
    assert classes_per_client > 0, f'classes per client not defined'
    # global_model.to(device)
    base_epoch = 0
    if args.resume_path:
        resume_path = Path(args.save_model_path) / args.resume_path
        assert resume_path.exists(), f'{resume_path} does not exist'
        global_model, base_epoch, loss, acc, best_acc = load_checkpoint(model=global_model, path=resume_path,
                                                                        device=torch.device('cpu'))

    # for client_model in clients_models:
    #     client_model.load_state_dict(global_model.state_dict())
    #     client_model.to(device)

    num_model_param_list = [p.numel() for p in global_model.parameters() if p.requires_grad]
    num_trainable_params = sum(num_model_param_list)

    # separate learnable params to smaller groups to ease subspace computation
    # separate by layers
    # It's manual, It's ugly but ze ma yesh
    # num_param_list = [14336,
    #                   16384, 16384, 16384, 16384,
    #                   16384, 16384, 16384, 16384,
    #                   16384, 16384, 16384, 16384,
    #                   16384, 16384, 16384, 16384,
    #                   16866, 16867, 16866, 16867
    #                   ]

    # num_param_list = [112640, 114688, 116618]
    num_param_list = [num_trainable_params]

    assert sum(num_param_list) == num_trainable_params, \
        (f'Expected modified list have the same total number of params,'
         f' got {sum(num_param_list)} != {num_trainable_params}')

    noise_multiplier = 0
    noise_multiplier_residual = 0

    noise_multiplier = 0
    if not args.no_noise:
        noise_multiplier = compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size,
                                                    client_data_sizes) if args.noise_multiplier == 0 else args.noise_multiplier
        # >>> *** GEP
        noise_multiplier_residual = noise_multiplier if args.noise_multiplier_residual == 0 else args.noise_multiplier_residual
        # <<< *** GEP

    GPs = torch.nn.ModuleList([])
    for client_id in range(num_clients):
        GPs.append(pFedGPFullLearner(args, classes_per_client).to(device))
    GPs.append(pFedGPFullLearner(args, classes_per_client).to(device)) # gp learner for public data
    public_gp_idx = -1
    # >>> ***GEP
    def get_aux_dataset(num_virtual_clients: int, original_public_loaders: List[DataLoader]) -> DataLoader:
        public_data_list, public_label_list = [], []
        for loader in original_public_loaders:
            for (data, labels) in loader:
                public_data_list.append(data)
                public_label_list.append(labels)

        X = torch.cat(public_data_list, dim=0)

        y = torch.cat(public_label_list, dim=0)

        public_dataset = TensorDataset(X, y)

        # cutmix = v2.CutMix(num_classes=10)
        # mixup = v2.MixUp(num_classes=10)
        # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        #
        #
        # print(f'virtual_ratio {virtual_ratio}  ')
        #
        # def collate_fn(batch):
        #     super_batch = [cutmix_or_mixup(*default_collate(batch))] * virtual_ratio
        #     images = torch.cat([data[0] for data in super_batch])
        #     labels = torch.cat([data[1] for data in super_batch])
        #
        #     return images, labels

        # return DataLoader(public_dataset, batch_size=pub_batch_size, shuffle=True)
        return DataLoader(public_dataset, batch_size=pub_batch_size, shuffle=True, drop_last=True)

    print(f'virtual_ratio {virtual_ratio}  ')
    augmenter = AugStackTransform(multiplicity=virtual_ratio)
    public_clients_loader = get_aux_dataset(num_virtual_clients=num_virtual_public_clients,
                                            original_public_loaders=clients_train_loaders[:num_public_clients])

    basis_gradients: Optional[torch.Tensor] = None
    basis_gradients_cpu: Optional[torch.Tensor] = None

    # Divide basis elements to groups proportional to the sqrt of group size
    sqrt_num_param_list = np.sqrt(np.array(num_param_list))
    num_bases_list = (num_basis_elements * (sqrt_num_param_list / np.sum(sqrt_num_param_list))).astype(int)
    num_basis_elements_actual = sum(num_bases_list)
    print(f'num bases list {num_bases_list} ')
    print(f'element actual {num_basis_elements_actual}')
    print('****')

    # <<< ***GEP
    pbar = trange(global_epoch)
    for epoch in pbar:
        to_eval = ((epoch + 1) > args.eval_after and (epoch + 1) % args.eval_every == 0) or (epoch + 1) == global_epoch
        # >>>  ***GEP

        # get public clients gradients for current global model state
        public_clients_model_updates = {}
        clients_model_updates = {}
        prev_params = OrderedDict()

        for n, p in global_model.named_parameters():
            public_clients_model_updates[n] = []
            clients_model_updates[n] = []
            prev_params[n] = p.detach()

        public_client_model = cifar10NetGPkernel()

        public_client_model.to(device)
        GPs[public_gp_idx], label_map, _, _ = build_tree(public_client_model, public_clients_loader, public_gp_idx, GPs)
        public_client_model = extend(public_client_model)
        GPs[public_gp_idx].train()
        public_client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(params=public_client_model.parameters(), lr=args.lr)
        optimizer.zero_grad()
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = extend(loss_fn)
        num_pub_samples = 0
        pub_data_list, public_label_list = [], []
        for k, (data, labels) in enumerate(public_clients_loader):
            data, labels = data.to(device), labels.to(device)
            pub_data_list.append(data)
            public_label_list.append(labels)
            # with torch.no_grad():
            #     data, labels = augmenter((data, labels))

            # current_batch_size = labels.size(0)
            # num_pub_samples += current_batch_size
            # print(f'data shape: {data.shape}')


            outputs = public_client_model(data)

            if torch.any(torch.isnan(outputs)):
                raise Exception(f'NaNs in model output')

            X = torch.cat((X, outputs), dim=0) if k > 0 else outputs
            Y = torch.cat((Y, labels), dim=0) if k > 0 else labels
            # Y = torch.cat((Y, torch.argmax(labels, dim=-1)), dim=0) if k > 0 else torch.argmax(labels, dim=-1)
            data, labels, outputs = data.cpu(), labels.cpu(), outputs.cpu()
            del data, labels, outputs

        # Define the  number of rows per split
        num_splits = num_virtual_public_clients

        # Calculate the number of splits
        # rows_per_split = int(public_grads_flat.size(0)) // num_splits
        rows_per_split = int(X.size(0)) // num_splits

        # Reshape the tensor to (num_splits, rows_per_split, n) and calculate the mean along the rows
        # public_virtual_grads = public_grads_flat[:num_splits * rows_per_split].reshape(num_splits, rows_per_split,
        #                                                                                -1).mean(dim=1)

        public_virtual_X = X[:num_splits * rows_per_split].reshape(num_splits, rows_per_split)
        public_virtual_Y = Y[:num_splits * rows_per_split].reshape(num_splits, rows_per_split)

        # Handle the remaining rows (4360 % 500 != 0)
        if X.size(0) % rows_per_split != 0:
            remaining_rows_X = X[num_splits * rows_per_split:]
            remaining_rows_Y = Y[num_splits * rows_per_split:]
            public_virtual_X = torch.cat([public_virtual_X, remaining_rows_X], dim=0)
            public_virtual_Y = torch.cat([public_virtual_Y, remaining_rows_Y], dim=0)

        for i in range(public_virtual_X.shape[0]):

            GPs[public_gp_idx], label_map, _, _ = build_tree(public_client_model, public_clients_loader, public_gp_idx,
                                                             GPs)
            public_client_model = extend(public_client_model)
            GPs[public_gp_idx].train()

            X = public_virtual_X[i]
            Y = public_virtual_Y[i]

            offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype, device=Y.device)
            # offset_labels = torch.tensor([label_map[torch.argmax(Y[i]).item()] for i in range(len(Y))], dtype=Y.dtype, device=Y.device)
            # GPs[public_gp_idx].to(device)
            loss = GPs[public_gp_idx](X, offset_labels, to_print=args.eval_every)

            loss.backward()

            # with backpack(BatchGrad()):
            #     loss.backward()

            for n, p in public_client_model.named_parameters():
                # assert p.grad_batch.shape[0] == current_batch_size, f'Expected {current_batch_size} per sample grads'
                # assert p.grad_batch.numel() == current_batch_size * p.numel(), f'Expected {current_batch_size}*{p.numel()} per sample grads elements'

                # public_clients_model_updates[n].extend(p.grad_batch.detach().cpu())
                public_clients_model_updates[n].extend(p.grad.detach().cpu())
                # public_clients_model_updates_not_used[n].append(p.grad.detach())
                # p.grad_batch.cpu()
                # del p.grad_batch, p.grad
                del p.grad

            optimizer.step()

        X, Y, loss= X.cpu(), Y.cpu(), loss.cpu()
        del X, Y, loss
        public_client_model.cpu()
        del public_client_model
        gc.collect()
        torch.cuda.empty_cache()

        public_grads_list = [torch.stack(public_clients_model_updates[n]) for n, p in global_model.named_parameters()]
        public_grads_flat = flatten_tensor(public_grads_list)
        assert public_grads_flat.dim() == 2, f'Expected flat grads per sample'
        # assert public_grads_flat.shape[0] == num_pub_samples, f'Expected grad per sample'
        assert public_grads_flat.shape[1] == num_trainable_params, f'Expected {num_trainable_params} element per grad'
        assert public_grads_flat.device == torch.device(
            'cpu'), f'Expected public_grads_flat.device cpu, got {public_grads_flat.device}'
        public_grads_flat = public_grads_flat.squeeze()
        # # Define the number of rows per split
        # num_splits = num_virtual_public_clients
        #
        # # Calculate the number of splits
        # # rows_per_split = int(public_grads_flat.size(0)) // num_splits
        # rows_per_split = int(public_grads_flat.size(0)) // num_splits
        #
        # # Reshape the tensor to (num_splits, rows_per_split, n) and calculate the mean along the rows
        # # public_virtual_grads = public_grads_flat[:num_splits * rows_per_split].reshape(num_splits, rows_per_split,
        # #                                                                                -1).mean(dim=1)
        #
        # public_virtual_grads = public_grads_flat[:num_splits * rows_per_split].reshape(num_splits, rows_per_split)
        #
        # # Handle the remaining rows (4360 % 500 != 0)
        # # if public_grads_flat.size(0) % rows_per_split != 0:
        # #     remaining_mean = public_grads_flat[num_splits * rows_per_split:].mean(dim=0, keepdim=True)
        # #     public_virtual_grads = torch.cat([public_virtual_grads, remaining_mean], dim=0)

        assert public_virtual_grads.device==torch.device('cpu'), f'Expected public_virtual_grads.device cpu , got {public_virtual_grads.device}'

        basis_gradients, basis_gradients_cpu, filled_history_size = add_new_gradients_to_history(public_virtual_grads, basis_gradients, basis_gradients_cpu, gradient_history_size)
        assert basis_gradients.shape[0] <= gradient_history_size, (f'Expected history of {gradient_history_size} grads'
                                                                   f' at most,'
                                                                   f' got {basis_gradients.shape[0]}')
        # assert basis_gradients.shape[1] == num_trainable_params, \
        #     f'Expected history of {num_trainable_params} entry grads'

        assert basis_gradients.device == device, f'Expected basis gradients device {device}, got {basis_gradients.device}'
        assert basis_gradients_cpu.device == torch.device('cpu'), f'Expected basis_gradients_cpu device cpu, got {basis_gradients_cpu.device}'

        public_grads_flat=public_grads_flat.cpu()
        public_virtual_grads=public_virtual_grads.cpu()
        public_grads_list=[pg.cpu() for pg in public_grads_list]
        del public_grads_list, public_grads_flat, public_virtual_grads
        gc.collect()
        torch.cuda.empty_cache()

        # compute new subspaces for parameter gradient groups  by basis gradient groups
        offset = 0
        pca_for_group = []
        num_components_explained_variance_ratio_lists_dict = {0.5: [], 0.75: [], 0.9: [], 0.95: []}
        for i, num_param in enumerate(num_param_list):
            pub_grad:torch.Tensor = basis_gradients[:filled_history_size, offset:offset + num_param]
            offset += num_param

            num_bases = num_bases_list[i]

            pca = compute_subspace(pub_grad, num_bases, device)
            num_components_explained_variance_ratio_dict = pca[-1]
            for k, v in num_components_explained_variance_ratio_dict.items():
                assert k in num_components_explained_variance_ratio_lists_dict
                num_components_explained_variance_ratio_lists_dict[k].append(v)

            pca_for_group.append(pca)
        pbar.set_description(f'Epoch {epoch} subspace computed ! '
                             f'clips {clipping_bound}, {clipping_bound_residual} '
                             f'noise multipliers {noise_multiplier}, {noise_multiplier_residual}')
        # <<< ***GEP

        sampled_client_num = max(1, int(user_sample_rate * num_clients))
        sampled_client_indices = random.sample(range(num_clients), sampled_client_num)
        # sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]

        pbar.set_description(f'Epoch {epoch} sampled {sampled_client_indices} '
                             f'clips {clipping_bound}, {clipping_bound_residual} '
                             f'noise multipliers {noise_multiplier}, {noise_multiplier_residual}')

        clients_accuracies = []
        clients_losses = []
        client_model = cifar10NetGPkernel().to(device)
        for idx, (client_trainloader, client_testloader) in enumerate(
                zip(sampled_clients_train_loaders, sampled_clients_test_loaders)):
            pbar.set_description(f'Epoch {epoch} Client in Iter {idx + 1} Client ID {sampled_client_indices[idx]} '
                                 f'clips {clipping_bound}, {clipping_bound_residual} '
                                 f'noise multipliers {noise_multiplier}, {noise_multiplier_residual}')

            # client_model = cifar10Net()
            client_model.load_state_dict(global_model.state_dict())
            local_model_state_dict, loss, GPs[idx] = local_update(client_model, client_trainloader, idx, GPs)
            clients_losses.append(loss)
            GPs[idx].tree = None
            if to_eval:
                accuracy = test(client_model, client_testloader, client_trainloader, idx, GPs)
                clients_accuracies.append(accuracy)

            # get client grads
            for n, p in local_model_state_dict.items():
                clients_model_updates[n].append(p.data.detach().cpu() - prev_params[n])
        client_model = client_model.cpu()
        del client_model
        gc.collect()
        torch.cuda.empty_cache()

        # stack sampled clients grads
        grads_list = [torch.stack(clients_model_updates[n]) for n, p in global_model.named_parameters()]

        # flatten grads for clipping and noising
        private_grads: torch.Tensor = flatten_tensor1(grads_list)  # cpu
        del grads_list
        assert private_grads.shape == (sampled_client_num, num_trainable_params), (f'Expected a gradient row vector'
                                                                                   f' for each sampled client,'
                                                                                   f' got {private_grads.shape}')
        assert private_grads.device == torch.device(
            'cpu'), f'Expected private gradient device to be cpu, got {private_grads.device}'
        private_grads = private_grads.to(device, non_blocking=True)

        # project grads to subspace computed by public grads
        embedding_list = []
        clean_reconstruction_list = []

        offset = 0
        for i, num_param in enumerate(num_param_list):
            grad_group = private_grads[:, offset:offset + num_param]
            assert grad_group.shape == (sampled_client_num, num_param), \
                (f'Expected a gradient {num_param}-D group for each of {sampled_client_num} sampled clients '
                 f', got {grad_group.shape}')

            pca = pca_for_group[i]

            embedded_grads_group: torch.Tensor = embed_grad(grad_group, pca, device)
            assert embedded_grads_group.shape[0] == sampled_client_num, (f'Expected group embedding'
                                                                         f' with {sampled_client_num} rows.'
                                                                         f' Each for every sampled client,'
                                                                         f' got {embedded_grads_group.shape[0]} rows')
            # assert embedded_grads_group.shape == (sampled_client_num, num_bases_list[i]), \
            #     (f'Expected group embedding {num_bases_list[i]} row for {sampled_client_num} sampled clients  '
            #      f', got {embedded_grads_group.shape}')

            if torch.any(torch.isnan(embedded_grads_group)):
                raise Exception(f'NaNs in embedding: {torch.sum(torch.any(torch.isnan(embedded_grads_group)))} NaNs')
            clean_reconstruction_group: torch.Tensor = project_back_embedding(embedded_grads_group, pca, device)
            # clean_reconstruction_group = (clean_reconstruction_group * scale_transform) + translate_transform
            assert clean_reconstruction_group.shape == (sampled_client_num, num_param), \
                (f'Expected a reconstruction gradient row vector for each sampled client,'
                 f' got {clean_reconstruction_group.shape}')
            if torch.any(torch.isnan(clean_reconstruction_group)):
                raise Exception(
                    f'NaNs in reconstruction: {torch.sum(torch.any(torch.isnan(clean_reconstruction_group)))} NaNs')

            clean_reconstruction_list.append(clean_reconstruction_group)
            embedding_list.append(embedded_grads_group)

            # residual_list.append(residual_group)
            offset += num_param

        embedded_grads: torch.Tensor = torch.cat(embedding_list, dim=1)
        # assert embedded_grads.device == torch.device('cpu'), f'Expected embedded grads to be cpu, got {embedded_grads.device}'
        assert embedded_grads.device == device, f'Expected embedded grads to be cpu, got {embedded_grads.device}'
        # assert embedded_grads.shape == (sampled_client_num, num_basis_elements_actual), \
        #     (f'Expected a gradient row vector embedded'
        #      f' in a {num_basis_elements_actual}-D subspace for each sampled client, got {private_grads.shape}')

        clean_reconstruction: torch.Tensor = torch.cat(clean_reconstruction_list, dim=1)
        assert clean_reconstruction.device == device, f'Expected device to be cpu, got {clean_reconstruction.device}'
        # assert clean_reconstruction.device == torch.device('cpu'), f'Expected device to be cpu, got {clean_reconstruction.device}'
        assert clean_reconstruction.shape == (sampled_client_num, num_trainable_params), (f'Expected a '
                                                                                          f'{num_trainable_params} row vector'
                                                                                          f' for each sampled client,'
                                                                                          f' got {clean_reconstruction.shape}')

        projection_residual: torch.Tensor = private_grads - clean_reconstruction
        assert projection_residual.device == device, f'Expected device {device}, got {projection_residual.device}'
        # assert projection_residual.device == torch.device('cpu'), f'Expected device {torch.device("cpu")}, got {projection_residual.device}'

        #
        # cosine_reconstruction_clean_mat = projection_residual @ clean_reconstruction.t() / (
        #         mean_norm_of_rows(projection_residual) * mean_norm_of_rows(clean_reconstruction))
        #
        # cosine_reconstruction_clean = mean_norm_of_rows(cosine_reconstruction_clean_mat)
        # if float(cosine_reconstruction_clean) > 3 or float(cosine_reconstruction_clean) < -3:
        #     projection_residual = -projection_residual
        #
        #

        if to_eval:
            print(clients_accuracies)
            acc = sum(clients_accuracies) / len(clients_accuracies)
            if best_acc < acc:
                best_acc = acc
                if acc > args.min_acc_save:
                    save_checkpoint(path=Path(f'{args.save_model_path}/{args.exp_name}_acc_{best_acc}.pth'),
                                    epoch=epoch,
                                    # optimizer=None,
                                    acc=best_acc, best_acc=best_acc,
                                    loss=np.mean(clients_losses), model=global_model)
            if args.wandb:
                wandb.log({'Accuracy': acc, 'Best Accuracy': best_acc}, step=epoch)
            mean_acc_s.append(acc)
            print(mean_acc_s)
            acc_matrix.append(clients_accuracies)

        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = torch.tensor([
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ], device=device).reshape(-1, 1)
        if torch.any(torch.isnan(sampled_client_weights)):
            raise Exception(
                f'NaNs in sampled_client_weights: {torch.sum(torch.any(torch.isnan(sampled_client_weights)))} NaNs')

        # assert torch.sum(sampled_client_weights).item() == 1.0, (f"sampled_client_weights should sum to 1,"
        #                                           f" got {sum(sampled_client_weights)}")

        # clip grads in embedding  subspace
        embedded_grads_norms = norm_of_rows(embedded_grads)
        # embedded_grads_norms = torch.linalg.norm(embedded_grads, dim=-1)
        grad_clip_factor = torch.max(torch.ones_like(embedded_grads_norms), embedded_grads_norms / clipping_bound)
        embedded_grads_clipped = torch.div(embedded_grads, grad_clip_factor.reshape(-1, 1))
        assert embedded_grads_clipped.shape == embedded_grads.shape, \
            f'Clipping should not change shape of embedded grads, got {embedded_grads_clipped.shape}'
        if torch.any(torch.isnan(embedded_grads_clipped)):
            raise Exception(
                f'NaNs in embedded_grads_clipped: {torch.sum(torch.any(torch.isnan(embedded_grads_clipped)))} NaNs grad_clip_factor mean {grad_clip_factor.mean()}')

        # noise grads in embedding subspace
        std_grads = noise_multiplier * clipping_bound
        grad_noise = torch.normal(mean=0.0, std=noise_multiplier * clipping_bound,
                                  size=embedded_grads_clipped.shape) if std_grads > 0 else torch.zeros_like(
            embedded_grads_clipped)
        grad_noise = grad_noise.to(device)
        noised_embedded_grads = embedded_grads_clipped + grad_noise
        assert noised_embedded_grads.shape == embedded_grads.shape, \
            f'Noising should not change shape of embedded grads, got {noised_embedded_grads.shape}'

        # clip residuals
        residual_update_norms = norm_of_rows(projection_residual)
        residual_clip_factor = torch.max(torch.ones_like(residual_update_norms),
                                         residual_update_norms / clipping_bound_residual)
        residual_update_clipped = torch.div(projection_residual, residual_clip_factor.reshape(-1, 1))
        assert residual_update_clipped.shape == projection_residual.shape, \
            f'Clipping should not change shape of residuals, got {residual_update_clipped.shape}'
        if torch.any(torch.isnan(residual_update_clipped)):
            raise Exception(
                f'NaNs in residual_update_clipped: {torch.sum(torch.any(torch.isnan(residual_update_clipped)))} NaNs   residual_clip_factor mean {residual_clip_factor.mean()}')

        # noise residuals
        std_residuals = noise_multiplier_residual * clipping_bound_residual
        noise_residual = torch.normal(mean=0.0, std=noise_multiplier_residual * clipping_bound_residual,
                                      size=residual_update_clipped.shape) if std_residuals > 0 else torch.zeros_like(
            residual_update_clipped)
        noise_residual = noise_residual.to(device)
        noised_residual_update = residual_update_clipped + noise_residual
        assert noised_residual_update.shape == projection_residual.shape, \
            f'Noising should not change shape of residuals, got {noised_residual_update.shape}'

        reconstruction_list = []
        offset = 0
        for i, num_bases in enumerate(num_bases_list):
            noised_embedded_grad_group = noised_embedded_grads[:, offset:offset + num_bases]
            pca = pca_for_group[i]

            reconstruction_group = project_back_embedding(noised_embedded_grad_group, pca, device)

            assert reconstruction_group.shape == (sampled_client_num, num_param_list[i]), \
                (f'Expected a reconstructed to {num_param_list[i]}-D group gradient row vector for each sampled client,'
                 f' got {reconstruction_group.shape}')

            reconstruction_list.append(reconstruction_group)
            offset += num_bases

        reconstructed_grads = torch.cat(reconstruction_list, dim=1)
        assert reconstructed_grads.shape == (sampled_client_num, num_trainable_params), \
            f'Expected a reconstructed gradient row vector for each sampled client, got {reconstructed_grads.shape}'

        if args.wandb and (((epoch + 1) % args.log_every == 0) or ((epoch + 1) == global_epoch)):

            private_grad_norms: torch.Tensor = norm_of_rows(private_grads)

            private_grad_norms_max = private_grad_norms.max()

            private_grad_norms_min = private_grad_norms.min()

            private_grad_norms_mean = private_grad_norms.mean()

            private_grad_norms_median = private_grad_norms.median()

            # embedded_grads_norms: torch.Tensor = norm_of_rows(embedded_grads)

            embedded_grads_norms_max = float(embedded_grads_norms.max())

            embedded_grads_norms_min = float(embedded_grads_norms.min())

            embedded_grads_norms_mean = float(embedded_grads_norms.mean())

            embedded_grads_norms_median = float(embedded_grads_norms.median())

            #
            # noised_grads_norm_mean = mean_norm_of_rows(noised_embedded_grads)
            #
            residual_norm_mean = float(residual_update_norms.mean())
            residual_norm_min = float(residual_update_norms.min())
            residual_norm_max = float(residual_update_norms.max())
            residual_norm_median = float(residual_update_norms.median())
            #
            # embedded_norm_mean = float(embedded_grads_norms.mean())
            #
            # reconstructed_norm_mean = mean_norm_of_rows(reconstructed_grads)

            reconstructed_grads_norms: torch.Tensor = norm_of_rows(reconstructed_grads)
            reconstructed_grads_norms_max = float(reconstructed_grads_norms.max())
            reconstructed_grads_norms_min = float(reconstructed_grads_norms.min())
            reconstructed_grads_norms_mean = float(reconstructed_grads_norms.mean())
            reconstructed_grads_norms_median = float(reconstructed_grads_norms.median())

            reconstructed_grads_ratio = reconstructed_grads_norms / private_grad_norms
            reconstructed_grads_ratio_median = float(reconstructed_grads_ratio.median())
            residual_grads_ratio = residual_update_norms / private_grad_norms
            residual_grads_ratio_median = float(residual_grads_ratio.median())

            # sqr_reconstruction_plus_sqr_residual = norm_of_rows_squared(clean_reconstruction) + norm_of_rows_squared(
            #     projection_residual)
            # sqr_grads = norm_of_rows_squared(private_grads)

            # reconstruction_diff = torch.dist(sqr_grads, sqr_reconstruction_plus_sqr_residual)
            #
            # reconstruction_diff_mean = float(reconstruction_diff.mean())
            # reconstruction_diff_min = float(reconstruction_diff.min())
            # reconstruction_diff_max = float(reconstruction_diff.max())
            # reconstruction_diff_median = float(reconstruction_diff.median())

            cosine_reconstruction_clean_residual_clean_mat = torch.diagonal(
                projection_residual @ clean_reconstruction.t()) / (
                                                                     norm_of_rows(projection_residual) * norm_of_rows(
                                                                 clean_reconstruction))

            cosine_reconstruction_clean_residual_clean = float(torch.mean(
                cosine_reconstruction_clean_residual_clean_mat))

            cosine_reconstruction_noise_residual_noise_mat = torch.diagonal(
                noised_residual_update @ reconstructed_grads.t()) / (
                                                                     norm_of_rows(
                                                                         noised_residual_update) * norm_of_rows(
                                                                 reconstructed_grads))

            cosine_reconstruction_noise_residual_noise = float(
                torch.mean(cosine_reconstruction_noise_residual_noise_mat))

            cosine_clean_reconstruction_residual_noise_mat = torch.diagonal(
                noised_residual_update @ clean_reconstruction.t()) / (
                                                                     norm_of_rows(
                                                                         noised_residual_update) * norm_of_rows(
                                                                 clean_reconstruction))

            cosine_clean_reconstruction_residual_noise = float(
                torch.mean(cosine_clean_reconstruction_residual_noise_mat))

            cosine_residual_noise_residual_clean_mat = torch.diagonal(
                noised_residual_update @ projection_residual.t()) / (
                                                               norm_of_rows(noised_residual_update) * norm_of_rows(
                                                           projection_residual))

            cosine_residual_noise_residual_clean = float(torch.mean(cosine_residual_noise_residual_clean_mat))

            hlog_dict = {}

            # hlog_dict['pub_grad_mean_norm_mean'] = float(pub_grad_mean_norm.mean())
            # hlog_dict['pub_grad_not_used_mean_norm_mean'] = float(pub_grad_not_used_mean_norm.mean())
            # hlog_dict['pub_grads_nan_rows'] = pub_grads_nan_rows
            # hlog_dict['pub_grads_backpack_nan_rows'] = pub_grads_backpack_nan_rows

            hlog_dict['embedded_grads_norms_mean'] = embedded_grads_norms_mean
            hlog_dict['embedded_grads_norms_max'] = embedded_grads_norms_max
            hlog_dict['embedded_grads_norms_min'] = embedded_grads_norms_min
            hlog_dict['embedded_grads_norms_median'] = embedded_grads_norms_median

            hlog_dict['private_grad_norms_mean'] = private_grad_norms_mean
            hlog_dict['private_grad_norms_max'] = private_grad_norms_max
            hlog_dict['private_grad_norms_min'] = private_grad_norms_min
            hlog_dict['private_grad_norms_median'] = private_grad_norms_median
            hlog_dict.update({
                'reconstructed_grads_ratio': reconstructed_grads_ratio_median,
                'residual_grads_ratio': residual_grads_ratio_median,
                'reconstructed_grads_norms_max': reconstructed_grads_norms_max,
                'reconstructed_grads_norms_min': reconstructed_grads_norms_min,
                'reconstructed_grads_norms_median': reconstructed_grads_norms_median,
                'reconstructed_grads_norms_mean': reconstructed_grads_norms_mean,
                'residual_norm_mean': residual_norm_mean,
                'residual_norm_min': residual_norm_min,
                'residual_norm_max': residual_norm_max,
                'residual_norm_median': residual_norm_median,
                'cosine_reconstruction_clean_residual_clean': cosine_reconstruction_clean_residual_clean,
                'cosine_residual_noise_residual_clean': cosine_residual_noise_residual_clean,
                'cosine_reconstruction_noise_residual_noise': cosine_reconstruction_noise_residual_noise,
                'cosine_clean_reconstruction_residual_noise': cosine_clean_reconstruction_residual_noise})

            # hlog_dict.update({'norm_pca_components' : float(torch.sqrt(sqrd_norm_pca_components.mean(0))),
            #             'norm_pca_features': float(torch.sqrt(sqrd_norm_pca_features.mean(0)))})
            for (th, lst) in num_components_explained_variance_ratio_lists_dict.items():
                hlog_dict.update({f'explained_variance_elmes_over_{th}': sum(lst)})

            wandb.log(hlog_dict, step=epoch)

        aggregated_update = (reconstructed_grads.T @ sampled_client_weights).T.squeeze()
        assert aggregated_update.shape == torch.Size([num_trainable_params]), \
            f'Expected a single {num_trainable_params}-D gradient row vector, got {aggregated_update.shape}'
        if torch.any(torch.isnan(aggregated_update)):
            raise Exception(
                f'NaNs in aggregated_update: {torch.sum(torch.any(torch.isnan(aggregated_update)))} NaNs')
        aggregated_residuals = (noised_residual_update.T @ sampled_client_weights).T.squeeze()
        assert aggregated_residuals.shape == torch.Size([num_trainable_params]), \
            f'Expected a single {num_trainable_params}-D row vector , got {aggregated_residuals.shape}'
        if torch.any(torch.isnan(aggregated_residuals)):
            raise Exception(
                f'NaNs in aggregated_residuals: {torch.sum(torch.any(torch.isnan(aggregated_residuals)))} NaNs')
        # update global net
        new_params = {}
        offset = 0

        noise_residual = noise_residual.cpu()
        grad_noise = grad_noise.cpu()
        residual_clip_factor = residual_clip_factor.cpu()
        residual_update_clipped = residual_update_clipped.cpu()
        grad_clip_factor = grad_clip_factor.cpu()
        embedded_grads_clipped = embedded_grads_clipped.cpu()
        noised_embedded_grads = noised_embedded_grads.cpu()
        projection_residual = projection_residual.cpu()
        clean_reconstruction = clean_reconstruction.cpu()
        embedded_grads = embedded_grads.cpu()
        private_grads = private_grads.cpu()
        noised_residual_update = noised_residual_update.cpu()
        reconstructed_grads = reconstructed_grads.cpu()
        sampled_client_weights = sampled_client_weights.cpu()
        aggregated_update = aggregated_update.cpu()
        aggregated_residuals = aggregated_residuals.cpu()
        for n, p in prev_params.items():
            new_params[n] = (p +
                             aggregated_update[offset: offset + p.numel()].reshape(p.shape) +
                             aggregated_residuals[offset: offset + p.numel()].reshape(p.shape))
            offset += p.numel()
        # update new parameters of global net
        global_model.load_state_dict(new_params)

        del new_params, aggregated_update, aggregated_residuals, \
            reconstructed_grads, noised_residual_update, \
            private_grads, embedded_grads, clean_reconstruction, projection_residual, \
            embedded_grads_clipped, grad_clip_factor, noised_embedded_grads, \
            residual_update_clipped, residual_clip_factor, grad_noise, noise_residual, sampled_client_weights

        gc.collect()
        torch.cuda.empty_cache()

        # torch.cuda.memory._dump_snapshot(f"gpu_snapshots/my_snapshot_epoch_{epoch}.pickle")

        # update client models
        # for client_model in clients_models:
        #     client_model.load_state_dict(global_model.state_dict())

    char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
    ID = ''
    for ch in random.sample(char_set, 5):
        ID = f'{ID}{ch}'

    print(
        f'===============================================================\n'
        f'task_ID : '
        f'{ID}\n'
        f'main_base\n'
        f'noise_multiplier : {noise_multiplier}\n'
        f'mean accuracy : \n'
        f'{mean_acc_s}\n'
        f'acc matrix : \n'
        f'{torch.tensor(acc_matrix)}\n'
        f'===============================================================\n'
    )
if __name__ == '__main__':
    main()

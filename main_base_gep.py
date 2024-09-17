import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from emg_utils import get_dataloaders
# from opacus import PrivacyEngine
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_noise_multiplier
from tqdm.auto import trange
import copy
import sys
import random
import wandb

# >>>  ***GEP
from gep_utils import (compute_subspace, embed_grad, flatten_tensor,
                       project_back_embedding, add_new_gradients_to_history)
# <<< ***GEP


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
# >>>  ***GEP
num_public_clients = args.num_public_clients
num_basis_elements = args.basis_size
gradient_history_size = args.history_size
# <<< ***GEP
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def local_update(model, dataloader):
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    # model = model.to('cpu')
    return model


def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0

    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)

    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy


def main():
    best_acc = 0.0
    mean_acc_s = []
    acc_matrix = []
    if dataset == 'MNIST':
        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
                                 clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]
        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
    elif dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)
        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()
    # elif dataset == 'FEMNIST':
    #     clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
    #     clients_models = [femnistNet() for _ in range(num_clients)]
    #     global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)
        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    elif dataset == 'putEMG':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_dataloaders()
        clients_models = [EMGModel(num_features=24 * 8, num_classes=8, use_softmax=True) for _ in range(num_clients)]
        global_model = EMGModel(num_features=24 * 8, num_classes=8, use_softmax=True)
    else:
        print('undifined dataset')
        assert 1 == 0

    global_model.to(device)
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())
        client_model.to(device)

    noise_multiplier = 0
    if not args.no_noise:
        noise_multiplier = compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size,
                                                    client_data_sizes)
        #noise_multiplier = 3.029
    # print('noise multiplier', noise_multiplier)

    # >>> ***GEP
    public_clients_loaders = clients_train_loaders[:num_public_clients]
    public_clients_models = clients_models[:num_public_clients]
    history_gradient_per_layer = [None for _ in global_model.parameters()]
    # <<< ***GEP

    pbar = trange(global_epoch)
    for epoch in pbar:
        to_eval = ((epoch + 1) > args.eval_after and (epoch + 1) % args.eval_every == 0) or (epoch + 1) == global_epoch

        # >>>  ***GEP

        # get public clients gradients for current global model state
        public_clients_model_updates = []
        for idx, (public_client_model, public_client_loader) in enumerate(
                zip(public_clients_models, public_clients_loaders)):
            public_client_model_backup = copy.deepcopy(public_client_model)
            local_model = local_update(public_client_model, public_client_loader)
            public_client_update = [param.data - global_weight for param, global_weight in
                                    zip(public_client_model.parameters(), global_model.parameters())]
            public_clients_model_updates.append(public_client_update)

            clients_models[idx] = public_client_model_backup  # do not update public models during pca update

        # compute basis for subspace spanned by public gradients
        pca_per_layer = []
        for i, p in enumerate(global_model.parameters()):
            layer_updates = [public_client_update[i] for public_client_update in public_clients_model_updates]
            flattened_layer_update = flatten_tensor(layer_updates)

            # update gradient history
            basis_gradients = history_gradient_per_layer[i]
            basis_gradients = add_new_gradients_to_history(flattened_layer_update, basis_gradients,
                                                           gradient_history_size)
            history_gradient_per_layer[i] = basis_gradients

            # compute new subspace basis
            pca = compute_subspace(basis_gradients, num_basis_elements)
            pca_per_layer.append(pca)

        # <<< ***GEP

        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        clients_model_updates = []
        clients_accuracies = []
        for idx, (client_model, client_trainloader, client_testloader) in enumerate(
                zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
            pbar.set_description(f'Epoch {epoch} Client in Iter {idx + 1} Client ID {sampled_client_indices[idx]} noise multiplier {noise_multiplier}')
            local_model = local_update(client_model, client_trainloader)
            client_update = [param.data - global_weight for param, global_weight in
                             zip(client_model.parameters(), global_model.parameters())]
            clients_model_updates.append(client_update)
            if to_eval:
                accuracy = test(client_model, client_testloader)
                clients_accuracies.append(accuracy)

        if to_eval:
            print(clients_accuracies)
            acc = sum(clients_accuracies) / len(clients_accuracies)
            best_acc = max(acc, best_acc)
            wandb.log({'Accuracy': acc, 'Best Accuracy': best_acc})
            mean_acc_s.append(acc)
            print(mean_acc_s)
            acc_matrix.append(clients_accuracies)

        acc_matrix.append(clients_accuracies)
        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = [
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ]

        # >>> ***GEP embed clients updates in subspace spanned by public clients
        embedded_clients_model_updates = [[] for _ in range(len(sampled_client_indices))]
        for i, p in enumerate(global_model.parameters()):
            layer_updates = [client_update[i] for client_update in clients_model_updates]
            flattened_layer_update = flatten_tensor(layer_updates)
            embedded_update = embed_grad(flattened_layer_update, pca_per_layer[i])
            for j, sampled_update in enumerate(embedded_clients_model_updates):
                sampled_update.append(embedded_update[j])

        clients_model_updates = embedded_clients_model_updates
        # <<< ***GEP

        clipped_updates = []
        for idx, client_update in enumerate(clients_model_updates):
            if not args.no_clip:
                norm = torch.sqrt(sum([torch.sum(param ** 2) for param in client_update]))
                clip_rate = max(1, (norm / clipping_bound))
                clipped_update = [(param / clip_rate) for param in client_update]
            else:
                clipped_update = client_update
            clipped_updates.append(clipped_update)
        noisy_updates = []
        for clipped_update in clipped_updates:
            noise_stddev = torch.sqrt(torch.tensor((clipping_bound ** 2) * (noise_multiplier ** 2) / num_clients))
            noise = [torch.randn_like(param) * noise_stddev for param in clipped_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_update, noise)]
            noisy_updates.append(noisy_update)

        # >>>> ***GEP project back the noisy embeddings
        noisy_updates = [[project_back_embedding(layer_update, pca, device).reshape(param.shape)
                          for (layer_update, pca, param) in
                          zip(client_update, pca_per_layer, global_model.parameters())]
                         for client_update in noisy_updates]
        # <<<< ***GEP

        aggregated_update = [
            torch.sum(
                torch.stack(
                    [
                        noisy_update[param_index] * sampled_client_weights[idx]
                        for idx, noisy_update in enumerate(noisy_updates)
                    ]
                ),
                dim=0,
            )
            for param_index in range(len(noisy_updates[0]))
        ]
        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)
        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())
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

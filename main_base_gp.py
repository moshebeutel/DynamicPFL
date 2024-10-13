import os
import random
import sys
import torch.optim as optim
import wandb
from tqdm.auto import trange
from data import *
from emg_utils import get_dataloaders
from gp_utils import build_tree
from net import *
from options import parse_args
from pFedGP.pFedGP.Learner import pFedGPFullLearner
from utils import compute_noise_multiplier

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
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


def local_update(model, dataloader, cid, GPs):
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # build tree at each step
    GPs[cid], label_map, _, _ = build_tree(model, dataloader, cid, GPs)
    GPs[cid].train()

    for _ in range(local_epoch):
        optimizer.zero_grad()
        for k, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            # forward prop
            pred = model(data)

            X = torch.cat((X, pred), dim=0) if k > 0 else pred
            Y = torch.cat((Y, labels), dim=0) if k > 0 else labels

        offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                     device=Y.device)

        loss = GPs[cid](X, offset_labels, to_print=args.eval_every)
        # loss *= args.loss_scaler

        # propagate loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
        optimizer.step()
    GPs[cid].tree = None
    return model.to('cpu'), GPs[cid]


def test(client_model, client_testloader, client_trainloader, cid, GPs):
    client_model.eval()
    client_model = client_model.to(device)

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

    client_model = client_model.to('cpu')

    return accuracy


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
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients, args.batch_size)
        clients_models = [cifar10NetGPkernel() for _ in range(num_clients)]
        global_model = cifar10NetGPkernel()
        classes_per_client = 10
    # elif dataset == 'FEMNIST':
    #     clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
    #     clients_models = [femnistNet() for _ in range(num_clients)]
    #     global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients, args.batch_size)
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
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())

    noise_multiplier = 0
    if not args.no_noise:
        noise_multiplier = compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size,
                                                    client_data_sizes)
        # noise_multiplier = 3.029
    print('noise multiplier', noise_multiplier)
    assert classes_per_client > 0, f'classes per client not defined'
    GPs = torch.nn.ModuleList([])
    for _ in range(num_clients):
        GPs.append(pFedGPFullLearner(args, classes_per_client))

    pbar = trange(global_epoch)
    for epoch in pbar:
        to_eval = ((epoch + 1) > args.eval_after and (epoch + 1) % args.eval_every == 0) or (epoch + 1) == global_epoch
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]

        clients_model_updates = []
        clients_accuracies = []
        for idx, (client_model, client_trainloader, client_testloader, cid) in enumerate(
                zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders,
                    sampled_client_indices)):

            pbar.set_description(f'Epoch {epoch} Client in Iter {idx + 1} Client ID {sampled_client_indices[idx]} noise multiplier {noise_multiplier}')

            local_model, GPs[cid] = local_update(client_model, client_trainloader, cid, GPs)
            client_update = [param.data - global_weight for param, global_weight in
                             zip(client_model.parameters(), global_model.parameters())]
            clients_model_updates.append(client_update)
            if to_eval:
                accuracy = test(client_model, client_testloader, client_trainloader, cid, GPs)
                clients_accuracies.append(accuracy)

        if to_eval:
            print(clients_accuracies)
            acc = sum(clients_accuracies) / len(clients_accuracies)
            best_acc = max(acc, best_acc)
            if args.wandb:
                wandb.log({'Accuracy': acc, 'Best Accuracy': best_acc})
            mean_acc_s.append(acc)
            print(mean_acc_s)
            acc_matrix.append(clients_accuracies)

        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = [
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ]
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

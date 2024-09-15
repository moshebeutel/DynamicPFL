import torch


@torch.no_grad()
def build_tree(net, loader, cid, GPs):
    """
    Build GP tree per client
    :return: List of GPs
    """
    device = next(net.parameters()).device
    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    GPs[cid].build_base_tree(X, offset_labels)  # build tree
    return GPs[cid], label_map, offset_labels, X


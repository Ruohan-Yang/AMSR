import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn import metrics
from typing import Any, Optional, Tuple

class GCN(nn.Module):
    def __init__(self, feature_dims, out_dims, hidden_dims):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_dims, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dims, out_dims)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class Model_Net(nn.Module):
    def __init__(self, dim, layer_number, gcn_data):
        super(Model_Net, self).__init__()

        self.node_dim = dim
        self.edge_dim = dim * 2
        self.layer_number = layer_number
        self.gcn_data = gcn_data

        for i in range(self.layer_number):
            gcn = GCN(feature_dims=1, out_dims=self.node_dim, hidden_dims=64)
            setattr(self, 'gcn%i' % i, gcn)

        self.g_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.edge_dim))

        self.weight_softmax = nn.Sequential(
            nn.Linear(in_features=3+self.edge_dim, out_features=2),
            nn.Softmax(dim=1))

        self.d_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=self.layer_number))

        self.p_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=2))

    def forward(self, leftnode, rightnode, layer):

        for i in range(self.layer_number):
            layer_embed = eval('self.gcn' + str(i))(self.gcn_data[i]).cuda()
            setattr(self, 'layer%i' % i, layer_embed)

        layer_names = ['self.layer' + str(i) for i in layer.cpu().numpy().tolist()]
        specific_embed = torch.Tensor().cuda()
        for (l, i, j) in zip(layer_names, leftnode, rightnode):
            temp = torch.cat((eval(l)[i], eval(l)[j]), dim=0).cuda()
            temp = torch.unsqueeze(temp, dim=0)
            specific_embed = torch.cat((specific_embed, temp), dim=0)

        common_embed = self.g_mlp(specific_embed)
        d_input = grad_reverse(common_embed, coeff=1)
        discriminant_out = self.d_mlp(d_input)

        weight_tensor = torch.cat((layer.unsqueeze(1), leftnode.unsqueeze(1), rightnode.unsqueeze(1), specific_embed), dim=1)
        w_out = self.weight_softmax(weight_tensor)
        w0 = w_out[:, 0].unsqueeze(1)
        w1 = w_out[:, 1].unsqueeze(1)
        p_input = torch.add(specific_embed * w0, common_embed * w1)
        prediction_out = self.p_mlp(p_input)

        return prediction_out, discriminant_out

    def loss(self, data, device):
        left_node = data[:, 0].to(device)
        right_node = data[:, 1].to(device)
        layer = data[:, 2].to(device)
        p_label = data[:, 3].to(device)
        p_out, d_out = self.forward(left_node, right_node, layer)
        criterion = nn.CrossEntropyLoss()
        p_loss = criterion(p_out, p_label)
        d_loss = criterion(d_out, layer)
        whole_loss = p_loss + d_loss
        return whole_loss

    def metrics_eval(self, eval_data, device):
        scores = []
        labels = []
        preds = []
        for data in eval_data:
            data = data[0]
            left_node = data[:, 0].to(device)
            right_node = data[:, 1].to(device)
            target_layer = data[:, 2].to(device)
            link_label = data[:, 3].to(device)
            output, _ = self.forward(left_node, right_node, target_layer)
            output = F.softmax(output, dim=1)
            _, argmax = torch.max(output, 1)
            scores += list(output[:, 1].cpu().detach().numpy())
            labels += list(link_label.cpu().detach().numpy())
            preds += list(argmax.cpu().detach().numpy())

        acc = metrics.accuracy_score(labels, preds)
        pre = metrics.precision_score(labels, preds, average='weighted')
        f1 = metrics.f1_score(labels, preds, average='weighted')
        auc = metrics.roc_auc_score(labels, scores, average=None)

        return acc, pre, f1, auc

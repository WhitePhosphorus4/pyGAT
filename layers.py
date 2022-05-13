import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_node, dropout, alpha, concat=True, iskernel=False, sigma=1):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # leakyrelu的激活斜率
        self.concat = concat    # 用于判断是否只有一个attention head
        self.iskernel = iskernel    # 用于判断是否使用核注意力
        self.sigma = sigma     # 核注意力的sigma
        self.num_node = num_node    # 数据集节点数量

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))    # 建立一个权重，用于对特征数F进行线性变换
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.bias = nn.Parameter(torch.empty(size=(num_node,num_node)))  # 偏置
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        if not iskernel:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))    # 计算α，输入是上一层两个输出的拼接，输出是eij
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
        else:
            self.bias = nn.Parameter(torch.empty(size=(num_node,num_node)))  # 偏置
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)   # 激活

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)   # 计算注意力权重

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # 对于邻接矩阵中的元素，如果大于0，则说明有新的邻接边出现，那么则使用新计算出的权值，否则表示没有变连接，使用一个默认值来表示
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)   # 做一次激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        if not self.iskernel:
            e = self._self_attentional_mechanism(Wh)
        else:
            e = self._kernel_attentional_mechanism(Wh)
            
        # broadcast add
        return self.leakyrelu(e)

    def _self_attentional_mechanism(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def _kernel_attentional_mechanism(self, Wh, kernel="gaussian", cyn=0.5, pow=3, beta=2):
        # kernel attentional mechanism
        
        if kernel == "polynomial":
            e = torch.matmul(Wh, Wh.T)
            c = torch.ones_like(e) * cyn
            e = e + c                                                                                      
            # e = e + self.bias
            e.pow(pow)
        elif kernel == "sigmoid":
            e = torch.matmul(Wh, Wh.T)
            c = torch.ones_like(e) * cyn
            e = beta * e + c
            # e = beta * e + self.bias
            e = F.tanh(e)
        elif kernel == "gaussian":
            # e = torch.ones_like(Wh)
            e = self._gaussian_kernel_mechanism(Wh)
        elif kernel == "ones":
            x = torch.matmul(Wh, Wh.T)
            e = torch.ones_like(x)
        else:
            raise NotImplementedError

        return e
    
    def _gaussian_kernel_mechanism(self, Wh, gama=10):
        # TODO: 实现高斯核运算
        D2 = torch.sum(Wh.pow(2), axis=1) + torch.sum(Wh.pow(2), axis=1).T - 2 * torch.matmul(Wh, Wh.T)
        # D2 = torch.matmul(Wh, Wh.T) + torch.matmul(Wh, Wh.T).T - 2 * torch.matmul(Wh, Wh.T)
        e = torch.exp(-D2/(2*self.sigma**2))

        # D2 = torch.sum(Wh.pow(2), axis=1).T - torch.matmul(Wh, Wh.T)
        # e = torch.exp(-gama*D2/(self.sigma**2))
        return e
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 




class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size() 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args=None, output_channels=9):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(12, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.c1 = Point_Classifier(args)
        self.c2 = Point_Classifier(args)

    def forward(self, x,adaptation=False,constant=1,mmd=False):
        # Neighbor Embedding Module
        xyz = x.permute(0, 2, 1) # 64,18,12
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x))) #64,64,18
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # 64,64,18
        x = x.permute(0, 2, 1)
        # npoint = 18 if nframe=6, npoint = 30 if nframe = 10
        new_xyz, new_feature = sample_and_group(npoint=18, radius=0.15, nsample=3, xyz=xyz, points=x) # new_feature(B,Nout18,k3,2*64)
        feature_0 = self.gather_local_0(new_feature) # 2*LBR + Maxpool # 64,128,18
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=18, radius=0.2, nsample=3, xyz=new_xyz, points=feature)  # 64,18,3,128
        feature_1 = self.gather_local_1(new_feature) # 2*LBR + Maxpool  64,256,18
 
        x = self.pt_last(feature_1) # encoder:  LBR + LBR + 4*SA + concat # 64,1024,18
        x = torch.cat([x, feature_1], dim=1) # 64,1280,18.  Here is different from PCT paper
        x = self.conv_fuse(x)  # LBR1024  64,1024,18
        hidden_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)# mid feature  64,1024
        
        if adaptation == True: 
            hidden_feature = grad_reverse(hidden_feature,constant) 
        
        y1 = self.c1(hidden_feature)
        y2 = self.c2(hidden_feature)
        return y1,y2
        

class Point_Classifier(nn.Module):
    def __init__(self,args,output_channels=9):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        
    def forward(self,x):
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

# Grad Reversal torch1.8
class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()
        self.lambda_

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_ ,= ctx.saved_variables
        # grad_input = grad_output.clone()
        return - lambda_ * grad_output,None

def grad_reverse(x, lambd=1.0):
    # return GradReverse(lambd)(x)
    return grl_func.apply(x,torch.tensor(lambd))

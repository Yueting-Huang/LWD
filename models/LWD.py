# 将兰氏距离和余弦距离结合成双分支
# 为不同层赋予不同的权重，效果差别不大
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import math
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def downsample(image, resize=False):
    if resize and min(image.shape[2], image.shape[3]) > 224:
        image = transforms.functional.resize(image,224)
    return image

def prepare_image224(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>224:
        image = transforms.functional.resize(image,(224,224))
    image = transforms.ToTensor()(image)
    # print("image_size",image.shape)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

# [B,C,H,W]->[B*H*W]
def CanberraDistance(X, Y):  
    B,C,H,W = X.shape
    X=X.permute(0,2,3,1).contiguous().view(-1,C)
    Y=Y.permute(0,2,3,1).contiguous().view(-1,C)
    # 计算绝对差值  
    diff = torch.abs(X - Y)  
    # 计算X和Y的绝对值和  
    sum_abs = torch.abs(X) + torch.abs(Y)  
    
    # 避免除以零的情况，利用torch.where替换  
    # 使用where()来处理分母为0的情况，确保输出不会出现NaN  
    distances = torch.where(sum_abs == 0, torch.zeros_like(sum_abs), diff / sum_abs)  
    
    # 计算每行的和返回  
    return distances.mean(dim=1)

# 计算注意力 输入X:[B,C],输出注意矩阵:[B]
class conv_Attention(nn.Module):
    def __init__(self, inchannels=512):
        super().__init__()
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inchannels // 8, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self,X):
        X= self.conv_attent(X)
        # print("con",X.shape)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=X.dtype, device=X.device).view(1, 1, 3, 3)  
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=X.dtype, device=X.device).view(1, 1, 3, 3)  
        
        # 计算水平和垂直梯度  
        grad_x = F.conv2d(X, sobel_x, padding=0)  # 对单通道图像应用卷积  
        grad_y = F.conv2d(X, sobel_y, padding=0)  # 对单通道图像应用卷积  
        grad_x = F.pad(grad_x, (1, 1, 1, 1), mode='constant', value=0) 
        grad_y = F.pad(grad_y, (1, 1, 1, 1), mode='constant', value=0) 
        # 计算梯度幅值（可以使用不同的方法，这里使用绝对值之和作为简单示例）  
        grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)
        # grad_magnitude = F.softmax(grad_magnitude, dim=0) 
        return grad_magnitude
    
# 计算注意力 输入X:[B,C],输出注意矩阵:[B]
class Chan_Squeeze(nn.Module):
    def __init__(self, inchannels=512, outchannels=512):
        super().__init__()
        self.conv_squeeze =  nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1),
            nn.ReLU()
        )
    def forward(self,X):
        out = self.conv_squeeze(X)
        return out

class ResNet50(torch.nn.Module):
    def __init__(self, requires_grad=False, resize=False):
        super(ResNet50, self).__init__()
        self.chns = [64, 256, 512, 1024, 2048]
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        ResNet50_pretrained_features = models.resnet50(pretrained=True)._modules
        ResNet50_pretrained_features = list(ResNet50_pretrained_features.values())

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), ResNet50_pretrained_features[x])
        for x in range(4, 5):
            self.stage2.add_module(str(x), ResNet50_pretrained_features[x])
        for x in range(5, 6):
            self.stage3.add_module(str(x), ResNet50_pretrained_features[x])
        for x in range(6, 7):
            self.stage4.add_module(str(x), ResNet50_pretrained_features[x])
        for x in range(7, 8):
            self.stage5.add_module(str(x), ResNet50_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]  #

        return outs

    def forward(self, x):
        if self.resize:
            x = downsample(x, resize=True)
        feats_x = self.get_features(x)

        return feats_x

def cosine_similarity_matrix(A, epsilon=1e-6):  
    # A 的形状是 [B, N, C]  
    
    # 计算每个向量的 L2-norm  
    norms = torch.norm(A, p=2, dim=2, keepdim=True)  # 形状 [B, N, 1]  
    normalized_tensor = A / (norms+epsilon)
    # 计算内积矩阵  
    # dot_products = torch.bmm(A, A.transpose(1, 2))  # 形状 [B, N, N]  
    
    # 使用广播 Calculate cosine similarity  
    cosine_similarity_matrices = torch.bmm(normalized_tensor, normalized_tensor.transpose(1, 2)) 
    
    # 处理模为0的情况  
    # similarity_matrices[norms.squeeze() == 0] = 0  # 或者可以用 np.nan 或自定义值  
    # print(cosine_similarity_matrices)
    return cosine_similarity_matrices  



class LWD_M(torch.nn.Module):
    def __init__(self):
        super(LWD_M, self).__init__()
        self.feature_extractor = ResNet50()
        self.chns = [64, 256, 512, 1024, 2048]
        self.win = [4,4,2,1,1]
        self.squeeze_num = [32, 64, 128, 256, 512]
        self.avgpool2d=nn.AvgPool2d(kernel_size=(2,2),stride=(2,2),padding=0)
        self.maxpool2d=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=0)

        self.Chan1 = Chan_Squeeze(inchannels=self.chns[0], outchannels=self.squeeze_num[0])
        self.Chan2 = Chan_Squeeze(inchannels=self.chns[1], outchannels=self.squeeze_num[1])
        self.Chan3 = Chan_Squeeze(inchannels=self.chns[2], outchannels=self.squeeze_num[2])
        self.Chan4 = Chan_Squeeze(inchannels=self.chns[3], outchannels=self.squeeze_num[3])
        self.Chan5 = Chan_Squeeze(inchannels=self.chns[4], outchannels=self.squeeze_num[4])

        self.Atten1 = conv_Attention(inchannels=self.chns[0])
        self.Atten2 = conv_Attention(inchannels=self.chns[1])
        self.Atten3 = conv_Attention(inchannels=self.chns[2])
        self.Atten4 = conv_Attention(inchannels=self.chns[3])
        self.Atten5 = conv_Attention(inchannels=self.chns[4])
        self.w_LWD = nn.Parameter(torch.ones(5))
        self.w_MD = nn.Parameter(torch.ones(5))

        # B C H W -> 池化窗口[B, C, H/win*W/win] 划窗口[B*win_num win_size C]
    def window_partition(self, x, window):
        B, C, H, W = x.shape
        # print("x",x.shape)
        x_pool = self.fea_pool(x,int(math.log2(window))).reshape(B, C, -1)
        x_win = x.reshape(B,C,H//window,window,W//window,window)
        x_win = x_win.permute(0,2,4,3,5,1).contiguous().view(-1,window*window,C)
        # print("x_win",x_win.shape)
        # print("x________________",x.shape)
        return x_win, x_pool
    
    def layer_MD(self, x, y, win):
        # B,C,H,W -> B,C,N
        B,C,H,W = x.shape
        if(win>1):
            x_win, x = self.window_partition(x,win)
            y_win, y = self.window_partition(y,win)
            x_win_d = cosine_similarity_matrix(x_win)
            y_win_d = cosine_similarity_matrix(y_win)
            x_d = cosine_similarity_matrix(x)
            y_d = cosine_similarity_matrix(y)
            # print("x_md",x_md.shape)
            # mse_win = torch.mean((x_win_d - y_win_d) ** 2, dim=(1, 2))
            # mse_win = torch.mean(mse_win.reshape(B,-1),dim=1)
            # mse_d = torch.mean((x_d - y_d) ** 2, dim=(1, 2))
           
            mse_win = torch.sum((x_win_d - y_win_d) ** 2, dim=(1, 2))
            mse_win = torch.sum(mse_win.reshape(B,-1),dim=1)
            mse_d = torch.sum((x_d - y_d) ** 2, dim=(1, 2))

            # print("mse_win",mse_win.shape)
            # print("mse_d",mse_d.shape)
            # print("__________________________")
            # print(mse_d)
            return mse_win+mse_d
        else:
            x_d = cosine_similarity_matrix(x.reshape(B,C,-1))
            y_d = cosine_similarity_matrix(y.reshape(B,C,-1))
            # mse_d = torch.mean((x_d - y_d) ** 2, dim=(1, 2))
            mse_d = torch.sum((x_d - y_d) ** 2, dim=(1, 2))
            return mse_d
        

    def layers_MD(self, x, y, win):
        scores = []
        for i in range(5):
            score = self.layer_MD(x[i],y[i], win[i])
            # scores.append(score+torch.log(score + 1))
            scores.append(score) 
        return scores
    def forward_MD(self, x, y, win):
        B = x[0].shape[0]
        X_squeeze = []
        Y_squeeze = []

        X_squeeze.append(self.Chan1(x[1]))
        X_squeeze.append(self.Chan2(x[2]))
        X_squeeze.append(self.Chan3(x[3]))
        X_squeeze.append(self.Chan4(x[4]))
        X_squeeze.append(self.Chan5(x[5]))

        Y_squeeze.append(self.Chan1(y[1]))
        Y_squeeze.append(self.Chan2(y[2]))
        Y_squeeze.append(self.Chan3(y[3]))
        Y_squeeze.append(self.Chan4(y[4]))
        Y_squeeze.append(self.Chan5(y[5]))
        
        # for i, fea in enumerate(X_squeeze):
        #     print("压缩维度后特征大小",i,fea.shape)

    #     # list [5]
        MD_score = self.layers_MD(X_squeeze, Y_squeeze, win)
    #     print(MD_score)

        return MD_score
    def forward_LWD(self, x, y):
        B, C, H, W = x[1].shape
        # feats可以作为梯度注意力提取，再划分窗口
        # [B, C, H, W]->[B, 1, H, W]->[B, H*W]
        # x=self.Pool(x,2)
        # y=self.Pool(y,2)

        attn1 = self.Atten1(x[1]).permute(0,2,3,1).contiguous().view(-1)      
        attn2 = self.Atten2(x[2]).permute(0,2,3,1).contiguous().view(-1)
        attn3 = self.Atten3(x[3]).permute(0,2,3,1).contiguous().view(-1)
        attn4 = self.Atten4(x[4]).permute(0,2,3,1).contiguous().view(-1)
        attn5 = self.Atten5(x[5]).permute(0,2,3,1).contiguous().view(-1)

        CD1 = CanberraDistance(x[1], y[1])
        CD2 = CanberraDistance(x[2], y[2])
        CD3 = CanberraDistance(x[3], y[3])
        CD4 = CanberraDistance(x[4], y[4])
        CD5 = CanberraDistance(x[5], y[5])

        # print("attn1",attn1.shape)
        # print("attn2",attn2.shape)
        # print("attn3",attn3.shape)
        # print("attn4",attn4.shape)
        # print("attn5",attn5.shape)
        # print("CD1",CD1.shape)
        # print("CD2",CD2.shape)
        # print("CD3",CD3.shape)
        # print("CD4",CD4.shape)
        # print("CD5",CD5.shape)

        score1 = torch.sum((attn1 * CD1).reshape(B,-1), dim=1)
        score2 = torch.sum((attn2 * CD2).reshape(B,-1), dim=1)
        score3 = torch.sum((attn3 * CD3).reshape(B,-1), dim=1)
        score4 = torch.sum((attn4 * CD4).reshape(B,-1), dim=1)
        score5 = torch.sum((attn5 * CD5).reshape(B,-1), dim=1)
        
        score = []
        # for i, fea in enumerate(score):
        #     print("分数尺寸",i,fea.shape)
        score.append(score1)
        score.append(score2)
        score.append(score3)
        score.append(score4)
        score.append(score5)
        # score.append(score1+torch.log(score1 + 1))
        # score.append(score2+torch.log(score2 + 1))
        # score.append(score3+torch.log(score3 + 1))
        # score.append(score4+torch.log(score4 + 1))
        # score.append(score5+torch.log(score5 + 1))

        return score
    def Pool(self,x,i):
        x_pool = []
        x_pool.append(x[0])
        x_pool.append(self.fea_pool(x[1],i))
        x_pool.append(self.fea_pool(x[2],i))
        x_pool.append(self.fea_pool(x[3],i))
        x_pool.append(self.fea_pool(x[4],i))
        x_pool.append(self.fea_pool(x[5],i))
        return x_pool
    def fea_pool(self,x,ratio):
        for i in range(ratio):
            x=self.avgpool2d(x)+self.maxpool2d(x)
        return x

    def forward(self, x, y, as_loss=True):
        # assert x.shape == y.shape
        # device = x.device
        b, p, c, h, w = x.shape
        x = x.view(b*p, c, h, w)
        y = y.view(b*p, c, h, w)
        
        # print("x_shape",x.shape)

        if as_loss:
            feats0 = self.feature_extractor(x)
            feats1 = self.feature_extractor(y)
        else:
            with torch.no_grad():
                feats0 = self.feature_extractor(x)
                feats1 = self.feature_extractor(y)

        # for i, fea in enumerate(feats0):
        #     print("特征大小",i,fea.shape)
        X_pool= self.Pool(feats0, 0)
        Y_pool= self.Pool(feats1, 0)
        # for i, fea in enumerate(X_pool):
        #     print("池化后特征大小",i,fea.shape)

        MD_score = self.forward_MD(X_pool, Y_pool, self.win)
        LWD_score = self.forward_LWD(X_pool, Y_pool)
        # print(sum(LWD_score))
        # print(sum(MD_score))
        ww_LWD = []
        ww_MD = []
        ww_LWD.append(torch.exp(self.w_LWD[0]) / torch.sum(torch.exp(self.w_LWD)))
        ww_LWD.append(torch.exp(self.w_LWD[1]) / torch.sum(torch.exp(self.w_LWD)))
        ww_LWD.append(torch.exp(self.w_LWD[2]) / torch.sum(torch.exp(self.w_LWD)))
        ww_LWD.append(torch.exp(self.w_LWD[3]) / torch.sum(torch.exp(self.w_LWD)))
        ww_LWD.append(torch.exp(self.w_LWD[4]) / torch.sum(torch.exp(self.w_LWD)))

        ww_MD.append(torch.exp(self.w_MD[0]) / torch.sum(torch.exp(self.w_MD)))
        ww_MD.append(torch.exp(self.w_MD[1]) / torch.sum(torch.exp(self.w_MD)))
        ww_MD.append(torch.exp(self.w_MD[2]) / torch.sum(torch.exp(self.w_MD)))
        ww_MD.append(torch.exp(self.w_MD[3]) / torch.sum(torch.exp(self.w_MD)))
        ww_MD.append(torch.exp(self.w_MD[4]) / torch.sum(torch.exp(self.w_MD)))

        for i in range(len(ww_LWD)):
            LWD_score[i]=ww_LWD[i]*LWD_score[i]
            MD_score[i]=ww_MD[i]*MD_score[i]

        final_score = sum(LWD_score)+sum(MD_score)
        # print("final_score",final_score)
        final_score = torch.mean(final_score.view(b,p),dim=1)
        # print(final_score)
        return 1-torch.log(final_score+1)
        # if as_loss == True:
        #     return final_score
        # else:
        #     with torch.no_grad():
        #         return torch.log(final_score+1)

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from thop import profile

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='imgs/I47.png')
    parser.add_argument('--dist', type=str, default='imgs/I47_03_05.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image224(Image.open(args.ref).convert("RGB"), resize=True).to(device)
    dist = prepare_image224(Image.open(args.dist).convert("RGB"), resize=True).to(device)

    model = LWD_M().to(device)
    model = model.eval()

    X = torch.randn(1, 3, 224, 224)
    Y = torch.randn(1, 3, 224, 224)
    score = model(X, Y, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 12.8507

    # -- coding: utf-8 --

    # flops, params = profile(model, (ref,dist))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

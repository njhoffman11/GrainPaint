import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class time_embeddings(nn.Module):
    def __init__(self, L, n=10000):
        super().__init__()
        self.L = L
        self.d = self.L // 2
        self.n = n

    def forward(self, t: torch.Tensor):
        # Compute the positional encodings once in log space.
        # PE(k, 2i) = sin(k / n^(2i/d))
        # PE(k, 2i+1) = cos(k / n^(2i/d))
        # L: length of the sequence
        # k: position of object in sequence from 0 to L/2
        # d: output dimension of the positional encoding
        # i: dimension of the positional encoding
        # n: scaling factor

        # Compute the positional encodings once in log space.
        # column position
        col_id = torch.arange(self.d, dtype=torch.float, device=t.device)
        # e.g: cos( k/n^(i/d) ) = cos(k / exp( -ln(n) i / d) )
        pe = t.unsqueeze(-1) * torch.exp(-math.log(self.n) * col_id / self.d).unsqueeze(0)
        # concatenate the sin and cos functions
        pe = torch.cat((torch.sin(pe), torch.cos(pe)), dim=-1)
        
        return pe


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, time_dim=32, activation="relu"):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=k_size,
                                stride=stride, padding=padding)

        self.conv3d_2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        
        # self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()

        self.time_resizer = nn.Sequential(nn.Linear(time_dim, in_channels), self.activation)

    def forward(self, x, t):
        x = self.conv3d(x)
        # x = self.conv3d(x)
        x = self.activation(x)
        t = self.time_resizer(t)
        t = t.unsqueeze(2).unsqueeze(3).unsqueeze(4) # (batch_size, out_channels, 1, 1, 1)
        x = x + t
        x = self.conv3d_2(x)
        x = self.activation(x)
        return x


# class FinalConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1)
#         super(ConvBlock, self).__init__()
#         self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
#                                 stride=stride, padding=padding)
#         self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

#     def forward(self, x):
#         x = self.batch_norm(self.conv3d(x))
#         # x = self.conv3d(x)
#         x = F.relu(x)
#         t = self.time_resizer(t)
#         t = t.unsqueeze(2).unsqueeze(3).unsqueeze(4) # (batch_size, out_channels, 1, 1, 1)
#         x = x + t
#         return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2, time_dim=32, activation="relu"):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 4
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        self.time_dim = time_dim  

        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels, time_dim=self.time_dim, activation=activation)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels, time_dim=self.time_dim, activation=activation)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x, t):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x, t)
                # print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                # print(k, x.shape)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4, time_dim=32, activation="relu"):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 4
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()
        self.time_dim = time_dim

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2, time_dim=self.time_dim, activation=activation)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2, time_dim=self.time_dim, activation=activation)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels, time_dim=self.time_dim, activation=activation)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, t, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x, t)
            else:
                x = op(x, t)
        return x


if __name__ == "__main__":
    # x has shape of (batch_size, channels, depth, height, width)
    torch.cuda.empty_cache()
    B = 2
    time_emb_dim = 32
    x_test = torch.randn(B, 1, 16, 16, 16)
    x_test = x_test.cuda()
    print("The shape of input: ", x_test.shape)
    t_test = torch.randint(0, 100, (B,)).long()
    t_test = t_test
    # Linear time_dim
    time_embedder = time_embeddings(L=time_emb_dim)
    time_net = nn.Sequential(
        time_embedder,
        nn.Linear(in_features=time_emb_dim, out_features=32),
        nn.ReLU(),
    )
    t = time_net(t_test).cuda()
    print("The shape of time embedding: ", t.shape)


    encoder = EncoderBlock(in_channels=1, time_dim=time_emb_dim)
    encoder.cuda()
    print(encoder)
    x_test, h = encoder(x_test, t)

    db = DecoderBlock(out_channels=1, time_dim=time_emb_dim)
    db.cuda()
    x_test = db(x_test, t, h)
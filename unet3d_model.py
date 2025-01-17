import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from building_components import EncoderBlock, DecoderBlock, time_embeddings


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="tanh", time_dim=32, block_activation="relu", time_activation="relu"):
        super(UnetModel, self).__init__()
        self.time_embedder = time_embeddings(L=time_dim)
        if time_activation == "relu":
            self.time_activation = nn.ReLU()
        elif time_activation == "leaky_relu":
            self.time_activation = nn.LeakyReLU()
        elif time_activation == "gelu":
            self.time_activation = nn.GELU()
        elif time_activation == "selu":
            self.time_activation = nn.SELU()
        elif time_activation == "tanh":
            self.time_activation = nn.Tanh()
        elif time_activation == "elu":
            self.time_activation = nn.ELU()
        self.time_net = nn.Sequential(
            self.time_embedder,
            nn.Linear(in_features=time_dim, out_features=time_dim),
            self.time_activation,
            nn.Linear(in_features=time_dim, out_features=time_dim),
            self.time_activation,
        )
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth, time_dim=time_dim, activation=block_activation)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth, time_dim=time_dim, activation=block_activation)
        if final_activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_activation == "relu":
            self.final_act = nn.ReLU()
        elif final_activation == "tanh":
            self.final_act = nn.Tanh()
        elif final_activation == "gelu":
            self.final_act = nn.GELU()

    def forward(self, x, t):
        t = self.time_net(t)
        x, downsampling_features = self.encoder(x, t)
        x = self.decoder(x, t, downsampling_features)
        x = self.final_act(x)
        return x
    
    def randomize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
    

if __name__ == "__main__":
    B = 2
    time_emb_dim = 32
    x_test = torch.randn(B, 1, 96, 96, 96)
    x_test = x_test.cuda()
    print("The shape of input: ", x_test.shape)
    t_test = torch.randint(0, 100, (B,)).long()
    inputs = torch.randn(B, 1, 96, 96, 96)
    print("The shape of inputs: ", inputs.shape)
    model = UnetModel(in_channels=1, out_channels=1, model_depth=4, final_activation="relu", time_dim=time_emb_dim)
    inputs = inputs.cuda()
    t_test = t_test.cuda()
    model.cuda()
    x = model(inputs, t_test)
    print(model)
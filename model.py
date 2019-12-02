hop_size = 768
n_fft = 1024


# Define model 

import torch.nn as nn
import torch
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv0 = nn.utils.weight_norm(nn.Conv2d(1, 1, kernel_size = 3, padding = 1), name = "weight")
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size = 3, padding = 1), name = "weight")
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(8, 16, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv4 = nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.Tconv1 = nn.utils.weight_norm(nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv2 = nn.utils.weight_norm(nn.ConvTranspose2d(32 * 2, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv3 = nn.utils.weight_norm(nn.ConvTranspose2d(16 * 2, 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        # self.Tconv4 = nn.utils.weight_norm(nn.ConvTranspose2d(8 * 2 , 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv4 = nn.utils.weight_norm(nn.Conv2d(8 * 2 , 1, kernel_size = 3, padding = 1), name = "weight")
        # self.gru1 = nn.GRU(1024, hidden_size = 512, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        # self.gru2 = nn.GRU(768, hidden_size = 384, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        # self.gru3 = nn.GRU(512, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        x_s0 = self.leaky_relu(self.conv0(x))
        print(x_s0.shape)
        x_s1 = self.leaky_relu(self.conv1(x_s0))
        print(x_s1.shape)
        x_s2 = self.leaky_relu(self.conv2(x_s1))
        print(x_s2.shape)
        x_s3 = self.leaky_relu(self.conv3(x_s2))
        print(x_s3.shape)
        x = self.conv4(x_s3)
        print(x.shape)
        x = self.leaky_relu(x)

        # Decoder
        x = self.Tconv1(x)
        
        x = self.leaky_relu(x)

        print(x.shape)
        # Add skip connection gru
        x = torch.cat((x, x_s3), dim = 1)

        x = self.Tconv2(x)
        x = self.leaky_relu(x)
        print(x.shape)
        # Add skip connection gru
        x = torch.cat((x, x_s2), dim = 1)
        x = self.Tconv3(x)
        x = self.leaky_relu(x)
        print(x.shape)
        # Add skip connection gru
        x = torch.cat((x, x_s1), dim = 1)
        x = self.Tconv4(x)
        x = self.leaky_relu(x)
        print(x.shape)
        x = self.leaky_relu(self.Tconv5(x))
        return x
class VocalModel(nn.Module):
    def __init__(self):
        super(VocalModel, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(513, 1024, kernel_size = 3, padding = 1), name = "weight")
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(1024, 768, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(768, 512, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv4 = nn.utils.weight_norm(nn.Conv1d(512, 256, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.Tconv1 = nn.utils.weight_norm(nn.ConvTranspose1d(256, 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv2 = nn.utils.weight_norm(nn.ConvTranspose1d(512 * 2, 768, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv3 = nn.utils.weight_norm(nn.ConvTranspose1d(768 * 2, 1024, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv4 = nn.utils.weight_norm(nn.Conv1d(1024 * 2 , 513, kernel_size = 3, stride = 1, padding = 1), name = "weight")
        self.gru1 = nn.GRU(1024, hidden_size = 512, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        self.gru2 = nn.GRU(768, hidden_size = 384, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        self.gru3 = nn.GRU(512, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional=True, dropout = 0.3)
        self.leaky_relu = nn.LeakyReLU()
        # self.sigmoid = nn.Sigmoid()
        # nn.LSTM()
        
        
        
    def forward(self, x):
        # Encoder

        x_s1 = self.leaky_relu(self.conv1(x))
        
        x_s2 = self.leaky_relu(self.conv2(x_s1))
        x_s3 = self.leaky_relu(self.conv3(x_s2))
        x = self.conv4(x_s3)
        x = self.leaky_relu(x)
        # x = self.conv4()
        
        # Do GRU skip connection 1
        x_s1 = x_s1.permute(0, 2, 1)
        x_s1 = self.gru1(x_s1)[0]
        x_s1 = x_s1.permute(0, 2, 1)
        
        # Do GRU skip connection 2
        x_s2 = x_s2.permute(0, 2, 1)
        x_s2 = self.gru2(x_s2)[0]
        x_s2 = x_s2.permute(0, 2, 1)

                # Do GRU skip connection 3

        x_s3 = x_s3.permute(0, 2, 1)
        x_s3 = self.gru3(x_s3)[0]
        x_s3 = x_s3.permute(0, 2, 1)
        
        # Decoder
        x = self.Tconv1(x)
        
        x = self.leaky_relu(x)

        
        # Add skip connection gru
        x = torch.cat((x, x_s3), dim = 1)

        x = self.Tconv2(x)
        x = self.leaky_relu(x)
        
        # Add skip connection gru
        x = torch.cat((x, x_s2), dim = 1)
        x = self.Tconv3(x)
        x = self.leaky_relu(x)
        
        # Add skip connection gru
        x = torch.cat((x, x_s1), dim = 1)
        x = self.Tconv4(x)
        x = self.leaky_relu(x)
        return x

def load_model(path):
    model = VocalModel()
    model.cuda()
    # print(model)
    model.load_state_dict(torch.load(path))
    return model
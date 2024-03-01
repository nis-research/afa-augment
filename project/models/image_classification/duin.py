import torch


class DualNorm2d(torch.nn.Module):
    def __init__(self, num_features, norm_layer=torch.nn.BatchNorm2d):
        super(DualNorm2d, self).__init__()
        self.ns = torch.nn.ModuleList([norm_layer(num_features), norm_layer(num_features)])
        self.num_features = num_features
        self.ignore_model_profiling = True

        self.n_choices = ['M', 'A']
        self.route = 'M'  # route images to main BN or aux BN

    def forward(self, x):
        idx = self.n_choices.index(self.route)
        y = self.ns[idx](x)
        return y

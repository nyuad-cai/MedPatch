import torch
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=384, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layernorm(x)
        return self.fc(x)

class DenseClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseClassifier, self).__init__()
        self.dense_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense_layer(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, args):
        super(Classifier, self).__init__()
        self.args = args
        num_classes = args.num_classes
        
        if args.classifier == "mlp":
            self.classifier = MLPClassifier(input_dim=input_dim, output_dim=num_classes)
        elif args.classifier == "dense":
            self.classifier = DenseClassifier(input_dim=input_dim, output_dim=num_classes)
        else:
            raise ValueError(f"Unsupported classifier type: {args.classifier}")

    def forward(self, x):
        return self.classifier(x)


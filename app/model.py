import torch
import torch.nn as nn
import torchvision


class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()

        self.densenet = torchvision.models.densenet201(weights=None)

        for param in self.densenet.parameters():
            param.requires_grad = False
        

        self.densenet.classifier = nn.Sequential(
            # A Linear layer reducing dimensions from 1920 to 256.
            nn.Linear(1920, 256),
            # A ReLU activation function.
            nn.ReLU(),
            # A Dropout layer with a 50% probability, for regularization.
            nn.Dropout(p=0.5),
            # The final Linear layer that outputs scores for our 2 classes.
            nn.Linear(256, num_classes)
        )



    def forward(self, x):
        x = self.densenet(x)
        return x


def load_model(model_path: str, num_classes: int):

    model = DenseNet(num_classes)
    original_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in original_state_dict.items():
        # Add the "densenet." prefix to each key
        new_key = "densenet." + key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model

from ood_detector import OODDetector
import torch
import torch.nn as nn
from pytorch_ood.loss import DeepSVDDLoss
import numpy as np

"""
    This is the ood detector class we used for the simulation and real studies.
    The Deep SVDD code was adopted from https://github.com/kkirchheim/pytorch-ood, 
    see the following citation:

    @InProceedings{kirchheim2022pytorch,
    author    = {Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
    title     = {PyTorch-OOD: A Library for Out-of-Distribution Detection Based on PyTorch},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {4351-4360}
    }
"""

class Model(nn.Module):
    """ """

    def __init__(self, ndim=2, state_size=5):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 128, bias=False)
        self.layer2 = nn.Linear(128, ndim, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x).relu()
        x = self.layer2(x)
        return x
    
class SvddOODDetector(OODDetector):
    """
        OOD Detector using SVDD.
    """
    def __init__(self, state_size, model_path, center, radius, ndim=2):
        super().__init__(state_size)
        self.center = center
        self.radius = radius
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(ndim=ndim, state_size=state_size)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.model.eval()
        self.criterion = DeepSVDDLoss(n_dim=ndim, center=center).to("cuda" if torch.cuda.is_available() else "cpu")
    
    def detect(self, state):
        """
            Returns the ood label and score for the state.
            Label is 1 if in distribution, -1 if out of distribution.
        """
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        # if we only have one state:
        try:
            ood_score = self.criterion.distance(self.model(state)).item()
            if ood_score - self.radius**2 > 0:
                # out of distribution, score should be high
                return -1, ood_score
            else:
                # in distribution, score should be low
                return 1, ood_score
        except:
            # remove first dimension of state
            state = state.squeeze(0)
            ood_scores = self.criterion.distance(self.model(state))
            ood_scores = ood_scores - self.radius**2
            ood_scores = ood_scores.cpu().detach().numpy()
            ood_labels = np.ones(ood_scores.shape)
            ood_labels[ood_scores > 0] = -1
            return ood_labels, ood_scores
    

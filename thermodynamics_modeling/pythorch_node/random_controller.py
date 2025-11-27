import torch
import random

DISCRETE_DIM = [2,3] # According to the paper, AHU__active is the discrete variable with the most
                     # unique values (3)

class RandomController:
    """
    Random controller for Neural ODE simulator.
    """

    def __init__(self, scale=0.5):
        self.scale = scale

    def modifyControlsSeq(self, controls_seq):    
        """
        Modify the control sequence.
        controls_seq: [1, H, d_u]
        """
        # print(type(controls_seq))
        _, H, d_u = controls_seq.shape

        # Compute min and max values
        min_vals = controls_seq.min(dim=1).values   # shape [1, d_u]
        max_vals = controls_seq.max(dim=1).values

        # New control sequence
        new_controls = torch.zeros_like(controls_seq)

        for i in range(d_u):

            vals = torch.unique(controls_seq[0,:,i])
            length = len(vals)

            # Discrete values 
            if length == DISCRETE_DIM[0] or length == DISCRETE_DIM[1]:
                idx = torch.randint(0, length, (H, 1), device=vals.device)
                rand_values = vals[idx]
            else:
                # Continuous values
                low = min_vals[0, i].item()
                high = max_vals[0, i].item()
                rand_values = low + (high - low) * torch.rand((H, 1), device=controls_seq.device)

            new_controls[0, :, i] = rand_values[:, 0]
        
        return new_controls
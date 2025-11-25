import torch
import random

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
        print(type(controls_seq))
        _, H, d_u = controls_seq.shape

        # Compute min and max values
        min_vals = controls_seq.min(dim=1).values   # shape [1, d_u]
        max_vals = controls_seq.max(dim=1).values  

        # New control sequence
        new_controls = torch.zeros_like(controls_seq)

        for i in range(d_u):
            low = min_vals[0, i].item()
            high = max_vals[0, i].item()

            # Booleans
            if low == 0 and high == 1:
                rand_values = torch.randint(0, 2, (H, 1), device=controls_seq.device)
            else:
                # Continuous values
                rand_values = low + (high - low) * torch.rand((H, 1), device=controls_seq.device)

            new_controls[0, :, i] = rand_values[:, 0]

        return new_controls
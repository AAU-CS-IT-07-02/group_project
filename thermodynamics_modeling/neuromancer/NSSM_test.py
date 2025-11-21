import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neuromancer import BasicLogger
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable
from neuromancer.trainer import Trainer
import os

# Import data loader from NODE.py
from NODE import get_splits, config


# ------------------------------------------------------------------------------
# 1. Model Definition
# ------------------------------------------------------------------------------
class SSM(nn.Module):
    """
    Neural State Space Model (NSSM) architecture.
    """
    def __init__(self, fx, fu, fd, nx, nu, nd):
        super().__init__()
        self.fx, self.fu, self.fd = fx, fu, fd
        self.in_features, self.out_features = nx + nu + nd, nx

    def forward(self, x, u, d):
        return self.fx(x) + self.fu(u) + self.fd(d)

# ------------------------------------------------------------------------------
# 2. Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset_split", "train_data.csv")

    if not os.path.exists(csv_path):
        print(f"Warning: File not found at {csv_path}")

    # Load Data
    H = 16  # Prediction horizon for testing
    train_loader, dev_loader, test_data, dt_sec, stats = get_splits(
        csv_path, dt_minutes=5, H=H, batch_size=64
    )

    # Determine dimensions
    ny = test_data['Y'].shape[-1]
    nu = test_data['U'].shape[-1]
    nd = test_data['D'].shape[-1]
    nx = ny

    # Initialize Networks
    n_hidden = 40
    fx = blocks.MLP(nx, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[n_hidden, n_hidden])
    fu = blocks.MLP(nu, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[n_hidden, n_hidden])
    fd = blocks.MLP(nd, nx, bias=True, linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU, hsizes=[n_hidden, n_hidden])

    ssm_module = SSM(fx, fu, fd, nx, nu, nd)

    # Build System Graph
    # Inputs: 'xn' (initial state), 'U' (controls), 'D' (disturbances)
    model_node = Node(ssm_module, ['xn', 'U', 'D'], ['xn'], name='NSSM')
    dynamics_model = System([model_node], name='system', nsteps=H)

    # Define Loss
    y = variable("Y")
    yhat = variable('xn')[:, :-1, :] # Align predictions with ground truth

    reference_loss = 5. * (yhat == y)^2
    reference_loss.name = "ref_loss"

    problem = Problem([dynamics_model], PenaltyLoss([reference_loss], []))

    # Trainer Configuration
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.003)
    logger = BasicLogger(args=None, savedir=config["outdir"], verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=50,
        epochs=50,
        train_metric="train_loss",
        dev_metric="dev_loss",
        eval_metric="dev_loss",
        logger=logger
    )

    # Training
    print("Starting training...")
    best_model = trainer.train()

    # --------------------------------------------------------------------------
    # 3. Evaluation and Plotting
    # --------------------------------------------------------------------------
    print("Generating evaluation plots...")

    problem.load_state_dict(best_model)

    # Prepare Test Data Batch
    test_batch = {}
    test_batch['name'] = 'test'

    for key, val in test_data.items():
        if key in ['xn', 'Y', 'U', 'D']:
            if torch.is_tensor(val):
                test_batch[key] = val.detach().clone().type(torch.float32)
            elif hasattr(val, "__len__"):
                test_batch[key] = torch.tensor(val, dtype=torch.float32)

    # Run Inference
    with torch.no_grad():
        output = problem(test_batch)

    # Determine output keys
    pred_key = 'test_xn' if 'test_xn' in output else 'xn'

    if pred_key in output:
        pred_traj = output[pred_key].cpu().numpy()
    else:
        pred_traj = output['xn'].cpu().numpy()

    true_traj = test_batch['Y'].cpu().numpy()

    # Plot first sample window for the first room
    sample_idx = 0
    room_idx = 0

    plt.figure(figsize=(12, 6))
    plt.plot(true_traj[sample_idx, :, room_idx],
             label='True Temperature', color='black', linewidth=2)
    plt.plot(pred_traj[sample_idx, :, room_idx],
             label='NSSM Prediction', color='cyan', linestyle='--', linewidth=2)

    plt.title(f"Model Verification: Room {room_idx} (Sample Window {sample_idx})")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(script_dir, "results_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
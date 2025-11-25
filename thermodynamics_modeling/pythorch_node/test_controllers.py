"""
Quick test for controller implementations.
"""

import torch
from controllers import get_controller, PerRoomPIDController, DeadbandController, NoopController, RandomController


def test_noop():
    """Test Noop controller returns unchanged controls."""
    ctrl = NoopController()
    
    controls_seq = torch.randn(1, 48, 18)
    y0 = torch.randn(1, 6)
    y_mean = torch.zeros(6)
    y_std = torch.ones(6)
    c_mean = torch.zeros(18)
    c_std = torch.ones(18)
    
    result = ctrl.apply(controls_seq, y0, y_mean, y_std, c_mean, c_std)
    assert torch.allclose(result, controls_seq), "Noop should return identical controls"
    print("✓ Noop controller test passed")


def test_pid_shape():
    """Test PID controller returns correct shape."""
    ctrl = PerRoomPIDController(setpoint_celsius=25.0)
    
    H, d_u, d_y = 48, 18, 6
    controls_seq = torch.randn(1, H, d_u)
    y0 = torch.randn(1, d_y)
    
    # Create realistic normalization stats
    y_mean = torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])  # Room temps around 20°C
    y_std = torch.tensor([2.0] * 6)  # Std dev of 2°C
    
    c_mean = torch.randn(d_u)
    c_std = torch.ones(d_u) * 0.3
    
    result = ctrl.apply(controls_seq, y0, y_mean, y_std, c_mean, c_std)
    
    assert result.shape == controls_seq.shape, f"Shape mismatch: {result.shape} != {controls_seq.shape}"
    assert not torch.isnan(result).any(), "NaN detected in output"
    print("✓ PID controller shape test passed")


def test_pid_operation():
    """Test PID responds to temperature error."""
    ctrl = PerRoomPIDController(setpoint_celsius=25.0, K_p=0.1)
    
    H, d_u, d_y = 48, 18, 6
    controls_seq = torch.zeros(1, H, d_u)  # Start with zero controls
    
    # y0 far below setpoint (20°C vs 25°C target)
    y0 = torch.zeros(1, d_y)  # Normalized: 0 means 20°C if mean=20, std=2
    
    y_mean = torch.tensor([20.0] * 6)
    y_std = torch.tensor([2.0] * 6)
    c_mean = torch.zeros(d_u)
    c_std = torch.ones(d_u)
    
    result = ctrl.apply(controls_seq, y0, y_mean, y_std, c_mean, c_std)
    
    # First room's radiator valve (index 0) should increase
    valve_change = result[0, 0, 0].item() - controls_seq[0, 0, 0].item()
    assert valve_change > 0, f"PID should increase heating valve when temp too low; got {valve_change}"
    print(f"✓ PID operation test passed (valve adjustment: +{valve_change:.4f})")


def test_deadband():
    """Test Deadband controller."""
    ctrl = DeadbandController(setpoint_celsius=25.0, deadband_celsius=0.5)
    
    H, d_u, d_y = 48, 18, 6
    controls_seq = torch.ones(1, H, d_u) * 0.5  # Mid-range controls
    
    # y0 below setpoint + deadband (i.e., < 24.5°C)
    y0 = torch.zeros(1, d_y)  # Normalized: 0 means 20°C if mean=20, std=2
    
    y_mean = torch.tensor([20.0] * 6)
    y_std = torch.tensor([2.0] * 6)
    c_mean = torch.zeros(d_u)
    c_std = torch.ones(d_u)
    
    result = ctrl.apply(controls_seq, y0, y_mean, y_std, c_mean, c_std)
    
    # First room's radiator valve should be maxed out (1.0 physical = normalized ~3.3 with std=1, mean=0)
    # But clamp is in physical space, so physical valve should be 1.0
    valve_physical = result[0, 0, 0].item() * c_std[0].item() + c_mean[0].item()
    assert valve_physical >= 0.99, f"Deadband should max heating valve when too cold; got {valve_physical}"
    print(f"✓ Deadband controller test passed (valve physical value: {valve_physical:.4f})")


def test_get_controller():
    """Test factory function."""
    c1 = get_controller("Noop")
    assert isinstance(c1, NoopController)
    
    c2 = get_controller("PID", setpoint=25.0)
    assert isinstance(c2, PerRoomPIDController)
    
    c3 = get_controller("Deadband", setpoint=22.0, deadband=1.0)
    assert isinstance(c3, DeadbandController)
    
    c4 = get_controller(None)
    assert isinstance(c4, NoopController)
    
    print("✓ Factory function test passed")


def test_random():
    """Test Random controller perturbs controls."""
    ctrl = RandomController(noise_std=0.1, seed=42)
    
    H, d_u, d_y = 48, 18, 6
    controls_seq = torch.zeros(1, H, d_u)  # Start with zero controls
    y0 = torch.randn(1, d_y)
    
    y_mean = torch.tensor([20.0] * 6)
    y_std = torch.tensor([2.0] * 6)
    c_mean = torch.zeros(d_u)
    c_std = torch.ones(d_u)
    
    result = ctrl.apply(controls_seq, y0, y_mean, y_std, c_mean, c_std)
    
    # Result should be different from input (noise added)
    assert not torch.allclose(result, controls_seq, atol=1e-5), "Random controller should perturb controls"
    
    # Result should be clamped to [0, 1] in physical space
    for i in range(d_u):
        controls_phys = result[0, :, i] * c_std[i] + c_mean[i]
        assert (controls_phys >= -0.01).all() and (controls_phys <= 1.01).all(), \
            f"Controls should be clamped to [0, 1]; got range [{controls_phys.min():.3f}, {controls_phys.max():.3f}]"
    
    assert not torch.isnan(result).any(), "NaN detected in output"
    print("✓ Random controller test passed")


if __name__ == "__main__":
    test_noop()
    test_pid_shape()
    test_pid_operation()
    test_deadband()
    test_get_controller()
    test_random()
    print("\n✓ All controller tests passed!")

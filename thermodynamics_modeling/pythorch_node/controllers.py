"""
Minimal controller implementations for building thermodynamics model.
All controllers work in denormalized (physical) space.
"""

import torch


class PerRoomPIDController:
    """
    Independent PID controller for each room's all actuators.
    
    - Operates in denormalized (physical) space: room temps in °C, setpoint in °C.
    - Each room controls: radiator valve, damper position, AHU on/off.
    - Also adjusts global setpoints (heating water temp, supply air temp).
    - Conservative tuning: K_p=0.02, K_i=0.001, K_d=0.005.
    
    Actuator indices per room in observations (3 per room):
        - Room 0: index 0 (valve), 1 (damper), 2 (AHU)
        - Room 1: index 3 (valve), 4 (damper), 5 (AHU)
        - ...
        - Room 5: index 15 (valve), 16 (damper), 17 (AHU)
    
    Global setpoints:
        - Index 18: Heating:Control__setpoint_water_temperature__supply
        - Index 19: Ventilation:Sensor__air_temperature__supply
    """
    
    def __init__(self, setpoint_celsius=25.0, K_p=0.02, K_i=0.001, K_d=0.005):
        """
        Args:
            setpoint_celsius: target room temperature in °C (same for all 6 rooms)
            K_p, K_i, K_d: conservative PID gains
        """
        self.setpoint = setpoint_celsius
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        
        # Per-room state
        self.integral_errors = [0.0] * 6
        self.prev_errors = [0.0] * 6
        
        # Global state for aggregate control signals
        self.integral_error_global = 0.0
        self.prev_error_global = 0.0
    
    def apply(self, controls_seq, y0, y_mean, y_std, c_mean, c_std):
        """
        Apply PID control to all actuators (radiator valves, dampers, AHU, global setpoints).
        
        Args:
            controls_seq: [1, H, d_u] normalized control sequence
            y0: [1, d_y=6] normalized initial room temps
            y_mean, y_std: [d_y] normalization stats for room temps
            c_mean, c_std: [d_u] normalization stats for controls
        
        Returns:
            controls_seq_modified: [1, H, d_u] modified control sequence (normalized)
        """
        # Ensure all inputs are float32 to match model dtype
        controls_seq = controls_seq.float()
        y0 = y0.float()
        y_mean = y_mean.float()
        y_std = y_std.float()
        c_mean = c_mean.float()
        c_std = c_std.float()
        
        controls_modified = controls_seq.clone()
        
        # Denormalize y0 to physical space (°C)
        y0_denorm = y0 * y_std.unsqueeze(0) + y_mean.unsqueeze(0)  # [1, 6]
        
        # Compute aggregate error (average across all rooms)
        errors_all = self.setpoint - y0_denorm[0, :].cpu().numpy()
        error_global = float(errors_all.mean())
        
        # Update global PID state
        P_global = self.K_p * error_global
        self.integral_error_global += error_global
        I_global = self.K_i * self.integral_error_global
        D_global = self.K_d * (error_global - self.prev_error_global)
        self.prev_error_global = error_global
        
        pid_signal_global_physical = P_global + I_global + D_global
        
        # PID update per room
        for room_idx in range(6):
            # Current room temp in °C
            y_current = y0_denorm[0, room_idx].item()
            
            # Error in physical space
            error = self.setpoint - y_current
            
            # PID terms (per-room)
            P = self.K_p * error
            self.integral_errors[room_idx] += error
            I = self.K_i * self.integral_errors[room_idx]
            D = self.K_d * (error - self.prev_errors[room_idx])
            self.prev_errors[room_idx] = error
            
            pid_signal_physical = P + I + D
            
            # Actuator indices for this room
            valve_idx = 3 * room_idx
            damper_idx = valve_idx + 1
            ahu_idx = valve_idx + 2
            
            # 1) RADIATOR VALVE: increase when temp too low
            pid_signal_norm = pid_signal_physical / c_std[valve_idx].item()
            controls_modified[0, :, valve_idx] += pid_signal_norm
            valve_phys = controls_modified[0, :, valve_idx] * c_std[valve_idx] + c_mean[valve_idx]
            valve_phys = torch.clamp(valve_phys, 0.0, 1.0)
            controls_modified[0, :, valve_idx] = (valve_phys - c_mean[valve_idx]) / c_std[valve_idx]
            
            # 2) DAMPER POSITION: modulate proportionally to error
            # Increase damper (more air) when heating needed (positive error)
            damper_signal = 0.5 * pid_signal_physical  # Scale down to be gentler
            pid_signal_norm_damper = damper_signal / c_std[damper_idx].item()
            controls_modified[0, :, damper_idx] += pid_signal_norm_damper
            damper_phys = controls_modified[0, :, damper_idx] * c_std[damper_idx] + c_mean[damper_idx]
            damper_phys = torch.clamp(damper_phys, 0.0, 1.0)
            controls_modified[0, :, damper_idx] = (damper_phys - c_mean[damper_idx]) / c_std[damper_idx]
            
            # 3) AHU ACTIVE: turn on if error > threshold, off otherwise
            ahu_threshold = 1.0  # 1°C below setpoint triggers AHU
            ahu_command_phys = 1.0 if error > ahu_threshold else 0.0
            ahu_command_norm = (ahu_command_phys - c_mean[ahu_idx].item()) / c_std[ahu_idx].item()
            controls_modified[0, :, ahu_idx] = ahu_command_norm
        
        # 4) GLOBAL SETPOINTS: adjust both heating water temp and supply air temp based on aggregate error
        # Index 18: Heating water setpoint (typically 40–70°C, let's modulate ±5°C from baseline)
        if c_std.shape[0] > 18:
            heating_setpoint_signal = pid_signal_global_physical * 0.1  # Scale down
            pid_norm_heating = heating_setpoint_signal / c_std[18].item()
            controls_modified[0, :, 18] += pid_norm_heating
            heating_phys = controls_modified[0, :, 18] * c_std[18] + c_mean[18]
            heating_phys = torch.clamp(heating_phys, 35.0, 75.0)  # Reasonable bounds for water temp
            controls_modified[0, :, 18] = (heating_phys - c_mean[18]) / c_std[18]
        
        # Index 19: Supply air temperature (typically 15–30°C)
        if c_std.shape[0] > 19:
            air_setpoint_signal = pid_signal_global_physical * 0.05  # Even gentler for air temp
            pid_norm_air = air_setpoint_signal / c_std[19].item()
            controls_modified[0, :, 19] += pid_norm_air
            air_phys = controls_modified[0, :, 19] * c_std[19] + c_mean[19]
            air_phys = torch.clamp(air_phys, 12.0, 32.0)  # Reasonable bounds
            controls_modified[0, :, 19] = (air_phys - c_mean[19]) / c_std[19]
        
        return controls_modified
    
    def reset(self):
        """Reset PID state (integral and derivative memory)."""
        self.integral_errors = [0.0] * 6
        self.prev_errors = [0.0] * 6
        self.integral_error_global = 0.0
        self.prev_error_global = 0.0


class NoopController:
    """
    No-op controller: returns controls unchanged.
    Useful as baseline for comparison.
    """
    
    def apply(self, controls_seq, y0, y_mean, y_std, c_mean, c_std):
        """Return controls unchanged."""
        # Ensure float32 to match model dtype
        return controls_seq.float().clone()
    
    def reset(self):
        """No-op."""
        pass


class DeadbandController:
    """
    Simple on/off thermostat with hysteresis (deadband).
    
    - If room temp < setpoint - deadband: radiator valve = max (1.0 normalized)
    - If room temp > setpoint + deadband: radiator valve = min (0.0 normalized)
    - Otherwise: hold current valve position
    """
    
    def __init__(self, setpoint_celsius=25.0, deadband_celsius=0.5):
        """
        Args:
            setpoint_celsius: target room temperature in °C
            deadband_celsius: ±band around setpoint for on/off switching
        """
        self.setpoint = setpoint_celsius
        self.deadband = deadband_celsius
    
    def apply(self, controls_seq, y0, y_mean, y_std, c_mean, c_std):
        """
        Apply deadband control to radiator valves.
        
        Args:
            controls_seq: [1, H, d_u] normalized control sequence
            y0: [1, d_y=6] normalized initial room temps
            y_mean, y_std: [d_y] normalization stats
            c_mean, c_std: [d_u] normalization stats
        
        Returns:
            controls_seq_modified: [1, H, d_u] modified control sequence
        """
        # Ensure all inputs are float32 to match model dtype
        controls_seq = controls_seq.float()
        y0 = y0.float()
        y_mean = y_mean.float()
        y_std = y_std.float()
        c_mean = c_mean.float()
        c_std = c_std.float()
        
        controls_modified = controls_seq.clone()
        
        # Denormalize y0 to physical space
        y0_denorm = y0 * y_std.unsqueeze(0) + y_mean.unsqueeze(0)  # [1, 6]
        
        for room_idx in range(6):
            y_current = y0_denorm[0, room_idx].item()
            valve_idx = 3 * room_idx
            
            if y_current < self.setpoint - self.deadband:
                # Too cold: heat fully (1.0 in physical space)
                valve_norm = (1.0 - c_mean[valve_idx].item()) / c_std[valve_idx].item()
            elif y_current > self.setpoint + self.deadband:
                # Too warm: stop heating (0.0 in physical space)
                valve_norm = (0.0 - c_mean[valve_idx].item()) / c_std[valve_idx].item()
            else:
                # In deadband: keep current valve position
                valve_norm = controls_modified[0, 0, valve_idx]
            
            # Apply to all steps
            controls_modified[0, :, valve_idx] = valve_norm
        
        return controls_modified
    
    def reset(self):
        """No state to reset."""
        pass


class RandomController:
    """
    Baseline random controller: perturbs all control signals with aggressive Gaussian noise.
    
    Useful for:
    - Comparing against deterministic controllers
    - Checking if model predictions are actually sensitive to control changes
    - Validating that non-random control strategies (PID, Deadband) make physical sense
    
    Strategy:
    - Add aggressive Gaussian noise N(0, sigma) to each actuator in normalized space
    - Clamp resulting values to valid ranges [0, 1] in physical space
    - Different noise per timestep for wilder behavior (not same for whole window)
    """
    
    def __init__(self, noise_std=0.5, seed=None):
        """
        Args:
            noise_std: standard deviation of Gaussian noise in normalized space (default 0.5 for wild exploration)
            seed: optional random seed for reproducibility
        """
        self.noise_std = noise_std
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
    
    def apply(self, controls_seq, y0, y_mean, y_std, c_mean, c_std):
        """
        Apply aggressive random noise to all actuators at every timestep.
        
        Args:
            controls_seq: [1, H, d_u] normalized control sequence
            y0: [1, d_y=6] normalized initial room temps (unused)
            y_mean, y_std: [d_y] normalization stats (unused)
            c_mean, c_std: [d_u] normalization stats for controls
        
        Returns:
            controls_seq_perturbed: [1, H, d_u] perturbed control sequence (normalized)
        """
        # Ensure float32
        controls_seq = controls_seq.float()
        c_mean = c_mean.float()
        c_std = c_std.float()
        
        controls_perturbed = controls_seq.clone()
        B, H, d_u = controls_seq.shape
        
        # Generate different random noise for EVERY timestep (wilder behavior)
        noise = torch.randn(B, H, d_u, device=controls_seq.device) * self.noise_std
        
        # Add noise in normalized space
        controls_perturbed += noise
        
        # Clamp to valid ranges [0, 1] in physical space, then re-normalize
        for i in range(d_u):
            # Convert to physical space
            controls_phys = controls_perturbed[:, :, i] * c_std[i] + c_mean[i]
            # Clamp
            controls_phys = torch.clamp(controls_phys, 0.0, 1.0)
            # Re-normalize
            controls_perturbed[:, :, i] = (controls_phys - c_mean[i]) / c_std[i]
        
        return controls_perturbed
    
    def reset(self):
        """No persistent state to reset."""
        pass


def get_controller(controller_name, **kwargs):
    """
    Factory function to instantiate controllers by name.
    
    Args:
        controller_name: 'PID', 'Deadband', 'Random', 'Noop', or None
        **kwargs: passed to controller __init__
    
    Returns:
        controller instance
    """
    if controller_name is None or controller_name == "Noop":
        return NoopController()
    elif controller_name == "PID":
        return PerRoomPIDController(
            setpoint_celsius=kwargs.get("setpoint", 25.0),
            K_p=kwargs.get("K_p", 0.02),
            K_i=kwargs.get("K_i", 0.001),
            K_d=kwargs.get("K_d", 0.005)
        )
    elif controller_name == "Deadband":
        return DeadbandController(
            setpoint_celsius=kwargs.get("setpoint", 25.0),
            deadband_celsius=kwargs.get("deadband", 0.5)
        )
    elif controller_name == "Random":
        return RandomController(
            noise_std=kwargs.get("noise_std", 0.1),
            seed=kwargs.get("seed", None)
        )
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

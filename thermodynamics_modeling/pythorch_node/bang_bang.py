import torch

def bang_bang_control(y0, controls_seq, scalers, controller_map, setpoint=21.0):
    """
    Deadband Controller with FIXED Thresholds (20.5 - 22.5).
    This version produced the graph you liked.
    """
    device = y0.device
    
    y_mean = scalers["y_mean"].to(device)
    y_std = scalers["y_std"].to(device)
    c_mean = scalers["c_mean"].to(device)
    c_std = scalers["c_std"].to(device)

    y0_real = y0 * y_std + y_mean 

    # --- HARDCODED THRESHOLDS (The "Good Graph" Settings) ---
    HEAT_START_TEMP = 21.4
    COOL_START_TEMP = 21.6
    
    # Actuator Settings
    RAD_ON = 1.0    
    RAD_OFF = 0.0
    
    DAMP_OPEN = 1.0  
    DAMP_SHUT = 0.0
    
    AHU_ON = 1.0
    AHU_OFF = 14.0

    for item in controller_map:
        room_idx = item['room_idx']
        rad_idx = item['rad_idx']
        damp_idx = item['damp_idx']
        ahu_idx = item['ahu_idx']
        
        current_temp = y0_real[0, room_idx].item()

        # Initialize to IDLE
        act_rad = RAD_OFF
        act_damp = DAMP_SHUT
        act_ahu = AHU_OFF

        if current_temp <= HEAT_START_TEMP:
            # Too Cold -> Heat
            act_rad = RAD_ON
            act_damp = DAMP_SHUT
            act_ahu = AHU_ON
            
        elif current_temp >= COOL_START_TEMP:
            # Too Hot -> Cool
            act_rad = RAD_OFF
            act_damp = DAMP_OPEN
            act_ahu = AHU_OFF
            
        else:
            # Comfort Zone -> Idle
            act_rad = RAD_OFF
            act_damp = DAMP_SHUT
            act_ahu = AHU_OFF

        # Apply
        if rad_idx is not None:
            controls_seq[:, :, rad_idx] = (act_rad - c_mean[rad_idx]) / c_std[rad_idx]
        if damp_idx is not None:
            controls_seq[:, :, damp_idx] = (act_damp - c_mean[damp_idx]) / c_std[damp_idx]
        if ahu_idx is not None:
            controls_seq[:, :, ahu_idx] = (act_ahu - c_mean[ahu_idx]) / c_std[ahu_idx]

    return controls_seq

def random_control(controls_seq):
    return controls_seq + torch.randn_like(controls_seq) * 0.1
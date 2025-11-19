import torch
import os

path = r"c:\software\AAU\group_project_and_dataset\group_project\thermodynamics_modeling\neuromancer\out_300\best_model_state_dict.pth"

if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        sd = torch.load(path, map_location='cpu')
        print("Keys in state_dict:")
        for k in sd.keys():
            print(k)
        
        # Also check shapes of a few keys to confirm dimensions
        print("\nShapes:")
        for k in list(sd.keys())[:5]:
            print(f"{k}: {sd[k].shape}")
            
    except Exception as e:
        print(f"Error loading: {e}")

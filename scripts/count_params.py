
import argparse
import os
import sys
import torch
import yaml
from prettytable import PrettyTable

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils import get_root_logger
from basicsr.archs import build_network
from basicsr.models.uncertainty_mapping import (
    LearnableUncertaintyMapping,
    SpatialUncertaintyMapping,
    ContentAwareSpatialUncertaintyMapping
)

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num_params):
    """Formats the parameter count into a readable string (e.g., 20.5M)."""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f} M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f} K"
    else:
        return str(num_params)

def main():
    parser = argparse.ArgumentParser(description='Calculate Model Parameters')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    # Load configuration options
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    print(f"\nAnalyzing Model Parameters for config: {args.opt}\n")
    table = PrettyTable(["Component", "Type", "Total Params", "Trainable Params"])
    table.align = "l"

    total_project_params = 0
    total_project_trainable = 0

    # 1. Analyze Generator (UNet/SwinUNet)
    if 'network_g' in opt:
        try:
            net_g = build_network(opt['network_g'])
            total = count_parameters(net_g)
            trainable = count_trainable_parameters(net_g)
            
            table.add_row([
                "Generator (Net G)", 
                opt['network_g']['type'] if 'type' in opt['network_g'] else opt['network_g'].get('target', 'Unknown').split('.')[-1],
                format_params(total), 
                format_params(trainable)
            ])
            total_project_params += total
            total_project_trainable += trainable
        except Exception as e:
            print(f"Error building network_g: {e}")

    # 2. Analyze MSE Network (if exists)
    if 'network_mse' in opt:
        try:
            net_mse = build_network(opt['network_mse'])
            total = count_parameters(net_mse)
            # Usually MSE net is frozen during stage 2, check config or logic, but here we just count static
            trainable = count_trainable_parameters(net_mse) 
            
            table.add_row([
                "MSE Network", 
                opt['network_mse'].get('target', 'Unknown').split('.')[-1],
                format_params(total), 
                format_params(trainable)
            ])
            total_project_params += total
            # Note: Whether it's trainable depends on the training script logic, but here we report model's attribute
            # In stage 2 usually it is frozen, so we might want to manually set trainable to 0 if we knew logic.
            # But strict 'model.parameters()' counting is safer for general purpose.
        except Exception as e:
            print(f"Error building network_mse: {e}")

    # 3. Analyze Uncertainty Mapper
    if 'uncertainty_mapping' in opt:
        um_opt = opt['uncertainty_mapping']
        if um_opt.get('use_learnable', False):
            try:
                mapper_type = um_opt.get('type', 'simple')
                if mapper_type == 'content_aware':
                    mapper = ContentAwareSpatialUncertaintyMapping(
                        in_channels=3, 
                        hidden_channels=um_opt.get('hidden_channels', 64),
                        num_heads=um_opt.get('num_heads', 4),
                        semantic_dim=768 if um_opt.get('semantic_model') == 'clip' else 384, # Approximate defaults
                        window_size=um_opt.get('window_size', 8),
                        use_pretrained_semantic=um_opt.get('use_pretrained_semantic', False),
                        semantic_model_name=um_opt.get('semantic_model', 'clip')
                    )
                elif mapper_type == 'spatial':
                     mapper = SpatialUncertaintyMapping(
                        hidden_dim=um_opt.get('hidden_dim', 64),
                        num_layers=um_opt.get('num_layers', 3)
                    )
                else:
                    input_dim = 1 if not um_opt.get('content_aware', False) else 4
                    mapper = LearnableUncertaintyMapping(input_dim=input_dim)
                
                total = count_parameters(mapper)
                trainable = count_trainable_parameters(mapper)
                
                table.add_row([
                    "Uncertainty Mapper", 
                    mapper_type,
                    format_params(total), 
                    format_params(trainable)
                ])
                total_project_params += total
                total_project_trainable += trainable
            except Exception as e:
                print(f"Error building uncertainty_mapper: {e}")

    print(table)
    print(f"\nTotal Project Parameters: {format_params(total_project_params)}")
    # We don't sum trainable here because some parts might be frozen in 'train.py' but show as trainable here
    print(f"Total Model Size (approx FP32): {total_project_params * 4 / (1024**2):.2f} MB")

if __name__ == '__main__':
    main()

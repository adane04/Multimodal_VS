import argparse
from pathlib import Path
import pprint
import json
from typing import Dict, Any

save_dir = Path('results_all')

class Config(object):
    def __init__(self, **kwargs):
        """Configuration class with multimodal support"""
        # Set all kwargs as class attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize paths
        self.set_dataset_dir(self.video_type)
        
        # Validate multimodal parameters
        self.validate_multimodal_config()

    def validate_multimodal_config(self):
        """Ensure all required multimodal parameters exist"""
        required_params = [
            'video_input_size',
            'text_input_size',
            'max_video_len',
            'max_text_len'
        ]
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Missing required multimodal parameter: {param}")

    def set_dataset_dir(self, video_type: str = 'tvsum'):
        """Initialize all path directories with multimodal support"""
        base_path = save_dir.joinpath(video_type, f'split_{self.split_index}')
        
        self.log_dir = base_path.joinpath('logs')
        self.score_dir = base_path.joinpath('scores')
        self.save_dir = base_path.joinpath('models')
        
        # Multimodal-specific paths
        self.feature_dir = base_path.joinpath('features')
        self.video_feature_path = self.feature_dir.joinpath('video_features.h5')
        self.text_feature_path = self.feature_dir.joinpath('text_features.npy')
        self.splits_path = self.feature_dir.joinpath('splits.json')

    def __repr__(self):
        """Pretty-print configurations with sections"""
        config_str = "Multimodal Configurations:\n"
        sections = {
            'Mode': ['mode', 'verbose', 'video_type'],
            'Model Architecture': [
                'video_input_size', 'text_input_size', 
                'hidden_size', 'num_layers', 'fusion_heads'
            ],
            'Sequence Handling': ['max_video_len', 'max_text_len'],
            'Training': [
                'n_epochs', 'batch_size', 'lr', 
                'discriminator_lr', 'summary_rate'
            ],
            'Paths': [
                'log_dir', 'score_dir', 'save_dir',
                'video_feature_path', 'text_feature_path'
            ]
        }

        for section, params in sections.items():
            config_str += f"\n[{section}]\n"
            for param in params:
                if hasattr(self, param):
                    config_str += f"{param:>20}: {getattr(self, param)}\n"
        
        return config_str

def str2bool(v: str) -> bool:
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(parse: bool = False, **optional_kwargs) -> Config:
    """Get configurations with multimodal support"""
    
    # Load default parameters
    with open('configs/default_params.json', 'r') as f:
        default_params: Dict[str, Any] = json.load(f)

    parser = argparse.ArgumentParser()

    # ==================== Mode Parameters ====================
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument('--mode', type=str, 
                          default=default_params.get('mode', 'train'))
    mode_group.add_argument('--verbose', type=str2bool, 
                          default=default_params.get('verbose', True))
    mode_group.add_argument('--video_type', type=str,
                          default=default_params.get('video_type', 'summe'))

    # ================== Model Architecture ===================
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--video_input_size', type=int,
                           default=default_params.get('video_input_size', 1024))
    model_group.add_argument('--text_input_size', type=int,
                           default=default_params.get('text_input_size', 768))
    model_group.add_argument('--hidden_size', type=int,
                           default=default_params.get('hidden_size', 512))
    model_group.add_argument('--num_layers', type=int,
                           default=default_params.get('num_layers', 2))
    model_group.add_argument('--num_heads', type=int,
                           default=default_params.get('num_heads', 8))
    model_group.add_argument('--fusion_heads', type=int,
                           default=default_params.get('fusion_heads', 8))
    model_group.add_argument('--summary_rate', type=float,
                           default=default_params.get('summary_rate', 0.3))

    # ================== Sequence Handling ====================
    seq_group = parser.add_argument_group('Sequence Handling')
    seq_group.add_argument('--max_video_len', type=int,
                         default=default_params.get('max_video_len', 300))
    seq_group.add_argument('--max_text_len', type=int,
                         default=default_params.get('max_text_len', 100))

    # =================== Training Parameters =================
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--n_epochs', type=int,
                           default=default_params.get('n_epochs', 50))
    train_group.add_argument('--batch_size', type=int,
                           default=default_params.get('batch_size', 4))
    train_group.add_argument('--clip', type=float,
                           default=default_params.get('clip', 5.0))
    train_group.add_argument('--lr', type=float,
                           default=default_params.get('lr', 1e-4))
    train_group.add_argument('--discriminator_lr', type=float,
                           default=default_params.get('discriminator_lr', 1e-5))
    train_group.add_argument('--split_index', type=int,
                           default=default_params.get('split_index', 0))
    train_group.add_argument('--discriminator_slow_start', type=int,
                           default=default_params.get('discriminator_slow_start', 1))

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Convert to dictionary and update with optional kwargs
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
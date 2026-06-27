import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import torch
import torch.nn as nn
from thop import profile
from src.core import YAMLConfig
import time

#torch.cuda.set_device(2)

def main(args):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # Load the model's state dictionary
    cfg.model.load_state_dict(state)

    # Check if GPU is available, if yes, use GPU, otherwise fallback to CPU
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Deploy the model (convert to evaluation mode and any other deployment-specific configurations)
    model_deploy = cfg.model.deploy()
    model_deploy = model_deploy.to(device)  # Move the model to GPU if available
    model_deploy.eval()  # Ensure the model is in evaluation mode

    # Optional: Load the postprocessor if needed (not used for FLOPs calculation)
    postprocessor = cfg.postprocessor.deploy()
    print(f"Postprocessor deploy mode: {postprocessor.deploy_mode}")

    class WrappedModel(nn.Module):
        def __init__(self, model, device):
            super().__init__()
            self.model = model
            self.device = device
            self.orig_target_sizes = torch.tensor([[640, 640]], device=device)  # Initialize on the correct device

        def forward(self, images):
            return self.model(images, None,  self.orig_target_sizes)

    wrapped_model = WrappedModel(model_deploy, device)

    print("Calculating FLOPs and Parameters...")
    batch_size = 1  # Increase batch size for better GPU utilization
    data = torch.rand(batch_size, 3, 640, 640).to(device)  # Move the input data to GPU
    orig_target_sizes = torch.tensor([[640, 640]]).to(device)

    try:
        flops, params = profile(wrapped_model, inputs=(data,))
        print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Total Parameters: {params / 1e6:.2f} M")
    except Exception as e:
        print(f"Error during FLOPs calculation with thop: {e}")
        flops = None
        params = None

    # Simulate inference and calculate runtime
    print("Simulating Inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = wrapped_model(data)
    end_time = time.time()
    
    # Calculate FPS (Frames Per Second)
    inference_time = end_time - start_time
    fps = batch_size / inference_time if inference_time > 0 else 0  # FPS calculation for batch size
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/rtdetr/r50_c2f_p2_i1_q1.yml', help='Path to the config file.')
    parser.add_argument('--resume', '-r', type=str, default='/home/data/USERS/wyr/DA-RTDETR/output/city2foggy/r50_p1_i1_c3/checkpoint.pth', help='Path to the checkpoint file.')
    parser.add_argument('--check', action='store_true', default=False, help='Check the ONNX model.')
    parser.add_argument('--simplify', action='store_true', default=False, help='Simplify the ONNX model.')

    args = parser.parse_args()

    main(args)
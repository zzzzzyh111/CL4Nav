import os
import shutil

import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_onnx(model, input_tensor, save_dir, filename='rgb_cl_simulation.onnx'):
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert models to eval mode
    model.eval()
    onnx_path = os.path.join(save_dir, filename)

    # Export as ONNX file
    torch.onnx.export(
        model,                        # Model
        input_tensor,                 # A sample input to define the size and type of input to the model
        onnx_path,                    # Path of the saved file
        export_params=True,           # Whether to export model parameters
        do_constant_folding=True,     # Whether to perform constant folding optimization
        input_names=['rgb_input'],        # Input name
        output_names=['feature_output'],      # Output name
        dynamic_axes={'rgb_input': {0: 'batch_size'}, 'feature_output': {0: 'batch_size'}}  # Dynamic batch size
    )
    print(f"ONNX model saved at {onnx_path}")


def save_config_file(model_checkpoints_folder, args):
    try:
        print(f"Attempting to save config file in folder: {model_checkpoints_folder}")

        if not os.path.exists(model_checkpoints_folder):
            print(f"Folder {model_checkpoints_folder} does not exist, creating it.")
            os.makedirs(model_checkpoints_folder)
        else:
            print(f"Folder {model_checkpoints_folder} already exists.")

        # Save Configuration File
        config_file_path = os.path.join(model_checkpoints_folder, 'config.yml')
        with open(config_file_path, 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

        # Check if the file was saved successfully
        if os.path.exists(config_file_path):
            print(f"Config file saved successfully at {config_file_path}")
        else:
            print(f"Failed to save config file at {config_file_path}")
    except Exception as e:
        print(f"Error while saving config file: {str(e)}")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Script to convert FaceSwap modeo to a ONNX model
import numpy as np
import onnx
import torch


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    
    # Dummy target image
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float32) # Switched to float32 to avoid old numpy deprecation warnings
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    # Dummy source embedding
    embedding = torch.randn(1, 512).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()
    
    # Convert mode to ONNX
    torch.onnx.export(
        net, 
        (img, embedding), 
        output, 
        input_names=["target", "source"], 
        output_names=["output"],
        keep_initializers_as_inputs=False, 
        verbose=False, 
        opset_version=opset
    )
    
    model = onnx.load(output)
    graph = model.graph
    
    # Set dynamic batch sizes for BOTH inputs
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    graph.input[1].type.tensor_type.shape.dim[0].dim_param = 'None'
    
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)


if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='FaceSwap PyTorch to onnx')
    parser.add_argument('--input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file_path = args.input
    if os.path.isdir(input_file_path):
        input_file_path = os.path.join(input_file_path, "faceswap_model.pt")
    assert os.path.exists(input_file_path)
    assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "faceswap_model.onnx")
    convert_onnx(backbone_onnx, input_file_path, args.output, simplify=args.simplify)

import numpy as np
import onnx
import torch


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()
    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
def convert_onnx_main(input, output, network, simplify=False):
    import os
    import argparse
    from backbones import get_model

    
    backbone_onnx = get_model(network, dropout=0.0, fp16=False, num_features=512)
    convert_onnx(backbone_onnx, input, output, simplify=simplify)

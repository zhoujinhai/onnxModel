import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

    def forward(self, xyz, center):
        """
        :param xyz: B * N * S * C == 1 * 2 * 4 * 3
        :param center: B * N * C == 1 * 2 * 3
        :return:
        """
        res = xyz - center.view(1, 2, 1, 3)   # .repeat([1, 1, 4, 1])

        return res



if __name__ == "__main__":

    net = TestNet()

    inputs = torch.randn((1, 2, 4, 3))

    center = torch.tensor([[[2.2264, 0.1646, 0.3770], [-0.6141, 0.1071, 2.1032]]])
    print(center.shape)
    out = net(inputs, center)
    print("**** torch out ******: ", out.shape)
    onnx_path = "./test.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (inputs, center),
                      onnx_path,
                      verbose=True,
                      input_names=["points", "center"],
                      output_names=["res"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      # dynamic_axes={
                      #     "points": {1: "b", 2: "c", 3: "n"},
                      #     "res": {0: "b", 1: "n"}
                      # }
                      )

    print("onnx model has exported!")

    # inference by onnx
    import onnxruntime
    import onnx
    # check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    so1 = onnxruntime.SessionOptions()
    available_providers = onnxruntime.get_available_providers()

    net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    out = net_session.run(None, {"points": inputs.numpy(), "center": center.numpy()})
    print("----onnx runtime out----: ", out)

    import cv2
    net = cv2.dnn.readNetFromONNX(onnx_path)
    # get layer info
    layer_names = net.getLayerNames()

    for name in layer_names:
        id = net.getLayerId(name)
        layer = net.getLayer(id)
        print("layer id : %d, type : %s, name: %s" % (id, layer.type, layer.name))

    print("cv2 load model is OK!")
    print("start set input")
    print("inputs shape: ", inputs.shape)
    net.setInput(inputs.numpy().astype(np.float32), name="points")
    net.setInput(center.numpy().astype(np.float32), name="center")
    print("set input Done")
    cv_res = net.forward()
    print("$$$$$cv res$$$$: ", cv_res.shape, cv_res.dtype, type(cv_res))

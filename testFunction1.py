import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(6, 64, 1)

    def forward(self, xyz):
        xyz = xyz.squeeze(0)    # xyz is B * C * N
        l0_points = xyz
        l0_xyz = xyz
        
        xyz = l0_xyz.permute(0, 2, 1)
        points = l0_points.permute(0, 2, 1)

        B, N, C = xyz.shape
        grouped_xyz = xyz.view(B, 1, N, C)
        print("grouped_xyz shape: ", grouped_xyz.shape, " points.shape: ", points.shape)
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, C)], dim=-1)
        new_points = new_points.permute(0, 3, 2, 1)
        print(new_points.shape)
        new_points = self.conv(new_points)
        return new_points


if __name__ == "__main__":
    net = TestNet()
    inputs = torch.randn((1, 3, 150))
    inputs = inputs.unsqueeze(0)

    out = net(inputs)
    print("**** torch out ******: ", out.shape)
    onnx_path = "./test1.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (inputs, ),
                      onnx_path,
                      verbose=True,
                      input_names=["points"],
                      output_names=["res"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                          "points": {1: "b", 2: "c", 3: "n"},
                          "res": {0: "b", 1: "n"}
                      }
                      )

    print("onnx model has exported!")

    import cv2
    net = cv2.dnn.readNetFromONNX(onnx_path)
    # 获取各层信息
    layer_names = net.getLayerNames()

    for name in layer_names:
        id = net.getLayerId(name)
        layer = net.getLayer(id)
        print("layer id : %d, type : %s, name: %s" % (id, layer.type, layer.name))

    print("cv2 load model is OK!")
    print("start set input")
    print("inputs shape: ", inputs.shape)
    net.setInput(inputs.numpy().astype(np.float32), name="points")
    # net.setInput(center.numpy().astype(np.float32), name="center")
    print("set input Done")
    cv_res = net.forward()
    print("$$$$$cv res$$$$: ", cv_res.shape, cv_res.dtype, type(cv_res))

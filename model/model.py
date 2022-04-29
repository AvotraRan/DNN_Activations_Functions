import torch.nn as nn
from activation import reLU_f,tanh_f,swish, mish
import torch.nn.functional as F



class DNN(nn.Module):
  """Simple convolutional network."""

  def __init__(self, image_side_size, num_classes, act_fc, use_pytorch_ac = False, in_channels=1):
    super().__init__()
    AC = {"tanh" : tanh_f, "swish": swish,"relu":reLU_f, "mish": mish}
    AC_pytorch = {"relu":F.relu, "tanh" : F.tanh, "swish": F.silu, "mish": F.mish}
    self.conv1 = nn.Conv2d(in_channels,image_side_size, 5,2,3)
    self.conv2 = nn.Conv2d(image_side_size, image_side_size,3,2,3)
    self.conv3 = nn.Conv2d(image_side_size,2*image_side_size, 3,1,2)
    self.conv4 = nn.Conv2d(2*image_side_size,2*image_side_size,3,1,1)
    self.conv5 = nn.Conv2d(2*image_side_size,1,3,1,0)
    self.linear = nn.Linear((image_side_size-18)*(image_side_size-18), num_classes)
    self.activation_functions = AC_pytorch[act_fc] if use_pytorch_ac else AC[act_fc]


  def forward(self, x):
    x = self.conv1(x)
    x = self.activation_functions(x)
    x = self.conv2(x)
    x = self.activation_functions(x)
    x = self.conv3(x)
    x = self.activation_functions(x)
    x = self.conv4(x)
    x = self.activation_functions(x)
    x = self.conv5(x)
    x = self.activation_functions(x)
    x = self.linear(x.view(x.size(0), -1))
    return x

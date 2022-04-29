# The aims of this project is to make a comparison of activation functions

# A comparison of Activations Functions  Tanh, Relu, Swish and Mish on Deep Neural Network

## Clone 

```
git clone https://github.com/HabibMbow94/DNN_Activations_Functions.git 
```
## Create virtual environment ##

```
$ python3 -m venv ENV_NAME
```
## Activate your environment ##

```
$ source ENV_NAME/bin/activate
```

## Requirement installations ##
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
## Running models  ##

```
$ python3 main.py --activation mish --num_epochs 30 -- torch_implement False  
```
## Example of running models ##
```
$ python3 main.py -ac relu -n 30 -i False  
```

```
$ python3 main.py -ac swish -n 30 -i True 
```
# Colab link:
[Click here.](https://colab.research.google.com/drive/1sz9kNF_jA0RTw9HlhTKheNwwWddRBr7u?usp=sharing)

# Related Papers #


* <a href= 'https://arxiv.org/pdf/1803.08375.pdf'> ReLU </a>
* <a href= 'https://hal.archives-ouvertes.fr/hal-03265059v2/document '> ReLU'(0) </a>
* <a href= 'https://en.wikipedia.org/wiki/Swish_function '> Swish  </a>
* <a href= 'https://www.bmvc2020-conference.com/assets/papers/0928.pdf'> Mish</a>

#  Structures of the Networks  #
* AC = {"relu":reLU_f, "tanh" : tanh_f, "swish": swish, "mish": mish}
* AC_pytorch = {"relu": relu, "tanh" : F.tanh, "swish": F.silu, "mish": F.mish}
* activation_functions = AC_pytorch[act_fc] if use_pytorch_ac else AC[act_fc]
* conv1 = nn.Conv2d(in_channels,image_side_size, 5,2,3)
* activation_functions(x)
* conv2 = nn.Conv2d(image_side_size, image_side_size,3,2,3)
* activation_functions(x)
* max2=nn.MaxPool2d(9)
* conv3 = nn.Conv2d(image_side_size,2*image_side_size, 3,1,2)
* activation_functions(x)
* conv4 = nn.Conv2d(2*image_side_size,2*image_side_size,3,1,1)
* activation_functions(x)
* conv5 = nn.Conv2d(2*image_side_size,1,3,1,0)
* activation_functions(x)
* linear = nn.Linear((image_side_size-18)*(image_side_size-18), num_classes)



# Contributors
<img src="https://avatars.githubusercontent.com/u/98966847?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/72751041?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/98966969?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/99017712?v=4" width="100" height="100">
------|-----|------|------
[Fenosoa Randrianjatovo](https://github.com/FenosoaRandrianjatovo) | [Habib Mbow](https://github.com/HabibMbow94) | [Santatriniaina Avotra Randrianambinina](https://github.com/AvotraRan) | [Heritiana Daniel Andriasolofo](https://github.com/heritiana-aimsammi-sn2022)
 
 

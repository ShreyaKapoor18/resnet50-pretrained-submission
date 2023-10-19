from model_tools.check_submission import check_models
import torch
import torchvision.models as models
from os.path import join
import os
import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import torchvision

"""
Template module for a base model submission to brain-score
"""
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['resnet50-meshes-lt-100-original-pretrained']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    
    """
    assert name == 'resnet50-meshes-lt-100-original-pretrained'
    model = torchvision.models.resnet50(pretrained=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = join(os.path.dirname(os.path.abspath(__file__)), 'resnet50-meshes_lt_100_textures-scratch-epoch=00-val_loss=5.24.pt')
    print(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_2 = {}
    for k,v in checkpoint.keys():
            k2 = k2.split('model')[-1]
    checkpoint_2[k]=v            
    model.load_state_dict(checkpoint_2)
        
    wrapper = PytorchWrapper(identifier='resnet50-meshes-lt-100-original-pretrained', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    """
    Convolutional layer 1: "features.0"
    Batch Normalization 1: "features.1"
    ReLU Activation 1
    Convolutional layer 2: "features.3"
    Batch Normalization 2: "features.4"
    ReLU Activation 2
    Convolutional layer 3: "features.7" (This is "features.2" in 0-based indexing)
    Batch Normalization 3: "features.8"
    ReLU Activation 3
    Max pooling layer: "features.9 """
    assert name=='resnet50-meshes-lt-100-original-pretrained'
    return ["layer1.1relu", 'layer2', 'maxpool', 'layer3.0.downsample', 'avegpool', 'fc']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """
    @article{DBLP:journals/corr/HeZRS15,
  author       = {Kaiming He and
                  Xiangyu Zhang and
                  Shaoqing Ren and
                  Jian Sun},
  title        = {Deep Residual Learning for Image Recognition},
  journal      = {CoRR},
  volume       = {abs/1512.03385},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.03385},
  eprinttype    = {arXiv},
  eprint       = {1512.03385},
  timestamp    = {Wed, 25 Jan 2023 11:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
 """


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
    #get_model('resnet50-meshes-lt-100-original')

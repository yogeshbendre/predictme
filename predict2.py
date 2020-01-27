import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import json

v3 = models.inception_v3(pretrained=True)
v3.aux_logits = False
labels = {}

#labels = json.load(open('imagenet1000_clsidx_to_labels.txt','r'))
with open('imagenet1000_clsidx_to_labels.txt') as f:
  labels = [line.strip() for line in f.readlines()]

def predict_class(filename):

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        v3.to('cuda')
    v3.eval()
    with torch.no_grad():
      output = v3(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    #print(torch.nn.functional.softmax(output[0], dim=0))
    print(output.shape)
    a, index = torch.max(output,1)
    print(a)
    print(index) 
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    lb = labels[index[0]].split(': ')[1].split("'")[1]
    print(lb, percentage[index[0]].item())
    return (lb)

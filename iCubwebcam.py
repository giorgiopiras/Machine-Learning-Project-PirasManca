import cv2
import torchvision.models as models
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms


def initialize_model_inc(num_classes, use_pretrained=True):
    model_ft = models.inception_v3(pretrained=use_pretrained)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299
    return model_ft, input_size


def initialize_model_alex(num_classes, use_pretrained=True):
    """ Alexnet
    """
    model_ft = models.alexnet(pretrained=use_pretrained)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


n_classes = 20
labels = ['bodylotion', 'book', 'cellphone', ' flower', 'glass', 'hairbrush', 'hairclip', 'mouse', 'mug', 'ovenglove',
          'pencilcase', 'perfume', 'remote', 'ringbinder', 'soapdispencer', 'sodabottle', 'sprayer', 'squeezer',
          'sunglasses', 'wallet']
net, in_size = initialize_model_alex(n_classes)
params = torch.load('iCubModel++')
net.load_state_dict(params)
net.eval()

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fr = Image.fromarray(image)
    fr.save('frame.jpg')

    input_image = Image.open('frame.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(in_size),
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        net.to('cuda')

    with torch.no_grad():
        output = net(input_batch)

    d = ImageDraw.Draw(input_image)
    font = ImageFont.truetype("arial.ttf", 25)
    d.text((20, 20), str(labels[output[0].argmax().item()]), fill=(255, 255, 0), font=font)
    input_image.save('frame.jpg')

    print(labels[output[0].argmax().item()])

    # Display the resulting frame
    frame = cv2.imread('frame.jpg')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

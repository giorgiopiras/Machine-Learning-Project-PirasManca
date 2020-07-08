import cv2
import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms

net = models.alexnet(pretrained=True)
net.eval()

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    fr = Image.fromarray(frame)
    fr.save('frame.jpg')

    input_image = Image.open('frame.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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

    print(output[0].argmax())

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

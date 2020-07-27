import matplotlib
import numpy as np
from torchvision import models, transforms
import torch
from torch import nn
from PIL import Image
import requests
import io
from secml.data import CDataset
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMeanStd
from secml.array import CArray
from secml.figure import CFigure
from secml.adv.attacks import CAttackEvasionPGDLS


def initialize_model_alex(num_classes, use_pretrained=True):
    """ Alexnet
    """
    model_ft = models.alexnet(pretrained=use_pretrained)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


n_classes = 20
net, in_size = initialize_model_alex(n_classes)
params = torch.load('iCubModel++')
net.load_state_dict(params)
net.eval()

# Random seed
torch.manual_seed(0)

criterion = nn.CrossEntropyLoss()
optimizer = None  # the network is pretrained

# imagenet normalization
normalizer = CNormalizerMeanStd(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

# wrap the model, including the normalizer
clf = CClassifierPyTorch(model=net,
                         loss=criterion,
                         optimizer=optimizer,
                         epochs=10,
                         batch_size=1,
                         input_shape=(3, 224, 224),
                         softmax_outputs=False,
                         preprocess=normalizer,
                         random_state=0,
                         pretrained=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
# img_path = input("Insert image path:")
img_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Wheel_mouse.JPG/1200px-Wheel_mouse.JPG'
r = requests.get(img_path)
img = Image.open(io.BytesIO(r.content))

# apply transform from torchvision
img_t = transform(img)

# convert to CArray
batch_t = torch.unsqueeze(img_t, 0).view(-1)
batch_c = CArray(batch_t.numpy())

# prediction for the given image
preds = clf.predict(batch_c)
labels = ['bodylotion', 'book', 'cellphone', ' flower', 'glass', 'hairbrush', 'hairclip', 'mouse', 'mug', 'ovenglove',
          'pencilcase', 'perfume', 'remote', 'ringbinder', 'soapdispencer', 'sodabottle', 'sprayer', 'squeezer',
          'sunglasses', 'wallet']
label = preds.item()
predicted_label = labels[label]
print(predicted_label)
matplotlib.use('TkAgg')
fig = CFigure()
fig.sp.imshow(img)
fig.sp.xticks([])
fig.sp.yticks([])
fig.sp.title(predicted_label)
fig.show()

# Attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 5  # Maximum perturbation
lb, ub = 0.0, 1.0  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = 0  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.01,
    'eta_min': 2.0,
    'max_iter': 100,
    'eps': 1e-3
}

pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf,
                                    surrogate_classifier=clf,
                                    surrogate_data=CDataset(batch_c, label),
                                    distance=noise_type,
                                    dmax=dmax,
                                    solver_params=solver_params,
                                    y_target=y_target,
                                    lb=lb, ub=ub)

print("Attack started...")
eva_y_pred, _, eva_adv_ds, _ = pgd_ls_attack.run(batch_c, label)
print("Attack complete!")

adv_label = labels[clf.predict(eva_adv_ds.X).item()]
print(adv_label)
start_img = batch_c
eva_img = eva_adv_ds.X

# normalize perturbation for visualization
diff_img = start_img - eva_img
diff_img -= diff_img.min()
diff_img /= diff_img.max()

start_img = np.transpose(start_img.tondarray().reshape((3, 224, 224)), (1, 2, 0))
diff_img = np.transpose(diff_img.tondarray().reshape((3, 224, 224)), (1, 2, 0))
eva_img = np.transpose(eva_img.tondarray().reshape((3, 224, 224)), (1, 2, 0))

fig = CFigure(width=15, height=5)
fig.subplot(1, 3, 1)
fig.sp.imshow(start_img)
fig.sp.title(predicted_label)
fig.sp.xticks([])
fig.sp.yticks([])

fig.subplot(1, 3, 2)
fig.sp.imshow(diff_img)
fig.sp.title("amplified perturbation")
fig.sp.xticks([])
fig.sp.yticks([])

fig.subplot(1, 3, 3)
fig.sp.imshow(eva_img)
fig.sp.title(adv_label)
fig.sp.xticks([])
fig.sp.yticks([])

fig.show()

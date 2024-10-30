import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image

from image_preprocessing import preprocess_image


class CaptchaModel(nn.Module):
    def __init__(self, num_classes=62):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.convs(torch.randn(1, 1, 60, 40))
        self.fc1 = nn.Linear(64 * 7 * 10, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 7 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def predict_symbol(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()
        if predicted_class < 10:
            return chr(predicted_class + ord('0'))
        elif predicted_class < 36:
            return chr(predicted_class - 10 + ord('A'))
        else:
            return chr(predicted_class - 36 + ord('a'))


if __name__ == "__main__":
    model = CaptchaModel(num_classes=62)
    model.load_state_dict(torch.load('captcha_model.pth'))
    model.eval()
    for i in range(1, 6):
        image_path = f'test_image_{i}.png'
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensors = [transform(Image.fromarray(i)) for i in preprocess_image(image_path)]
        predictions = []
        for tensor in image_tensors:
            predictions.append(predict_symbol(model, tensor))
        print(''.join(predictions))

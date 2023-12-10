# Streamlit code
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import fastai
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# rf_loaded = pickle.load(open("model.h5","rb"), map_location = 'cpu')
rf_loaded =  CPU_Unpickler(open("model.h5","rb")).load()

# rf_loaded = torch.load('model.h5',map_location=torch.device('cpu'))

def transform_multiply(mul):
    def fn(arr):
        arr = arr * mul
        return arr
    return fn

def plot_image(img_batch, figsize=(8,3), cmap=None, title=None):
    if len(img_batch.shape)==3:
        img_batch = np.expand_dims(img_batch, axis=0)
    N = len(img_batch)
    fig = plt.figure(figsize=figsize)
    for i in range(N):
        img = img_batch[i]
#         img = np.transpose(img, [1,0,2])
        plt.subplot(1,N,i+1)
        plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(f"{title}")
    plt.show()

# Function to preprocess the image for model input
def preprocess_image(image):
    grayscale_image = Image.fromarray(np.uint8(image))
    grayscale_image = grayscale_image.convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(grayscale_image).unsqueeze(0)
    return img_tensor.to(device)

rf_loaded.eval()

def rgb2lab(rgb):
    if len(rgb.shape)==4:
        arr = []
        for img in rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            arr.append(img)
        arr = np.array(arr)
    else:
        arr = cv2.cvtColor(rgb, cv2.COLOR_LAB2RGB)
    return arr

def calculate_l1_loss(predicted_image, target_image):
    # Convert both images to LAB color space
    predicted_lab = rgb2lab(predicted_image)
    target_lab = rgb2lab(target_image)
    
    # Extract AB channels
    predicted_ab = predicted_lab[:, :, 1:]  # Extract AB channels from predicted image
    target_ab = target_lab[:, :, 1:]  # Extract AB channels from target image
    
    # Calculate L1 loss between AB values
    l1_loss = np.mean(np.abs(predicted_ab - target_ab))
    
    return l1_loss

def autocolorize(image):
    preprocessed_img = preprocess_image(image)
    
    # Ensure model is also on the same device
    rf_loaded.to(device)
    
    # Perform model inference
    with torch.no_grad():
        colored_image = rf_loaded(preprocessed_img)
    
    
    # Move the output tensor to CPU for post-processing
    colored_image = colored_image.cpu().squeeze(0).numpy()
    colored_image = np.transpose(colored_image, (1, 2, 0))
    

    # Post-processing to convert the output to RGB
    preprocessed_img_cpu = preprocessed_img.cpu().squeeze(0).numpy()
    preprocessed_img_cpu = np.transpose(preprocessed_img_cpu, (1, 2, 0))

    colored_image = transform_multiply(255.0)(colored_image)
    
    preprocessed_img_cpu = transform_multiply(255.0)(preprocessed_img_cpu)
    
    lab_pred = np.concatenate((preprocessed_img_cpu, colored_image), axis=2)
    
    rgb_pred = cv2.cvtColor((lab_pred.astype("uint8")), cv2.COLOR_LAB2RGB)
    # print(lab_pred)
    
    # Plotting the resulting RGB image
    # plot_image(rgb_pred, figsize=(10,10), title="RGB Actual")
    data = Image.fromarray(rgb_pred) 
    return data

st.title('Black & White Image Autocolorization')

uploaded_file = st.file_uploader("Choose a black & white image to colorize", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image',  width=300)

    if st.button('Autocolorize'):
        # Perform autocolorization
        colored_image = autocolorize(image)
        st.image(colored_image, caption='Autocolorized Image',  width=300)
        
        
st.title('L1 Checker')

uploaded_file = st.file_uploader("Choose a colored image. Our model will produce the black and white version of it and color it so you can see how accuracte it is to the real image.", type=["jpg", "png", "jpeg"])
colored_image = Image.fromarray(np.zeros((5, 5)))  # Replace (5, 5) with the desired shape

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', width=300)

    if st.button('Check L1 Score'):
        # Perform autocolorization
        colored_image = autocolorize(image)
        st.image(colored_image, caption='Autocolorized Image', width=300)

        # Calculate L1 loss with a newly uploaded image
        uploaded_image = Image.open(uploaded_file)  # Load the newly uploaded image again
        uploaded_image = np.array(uploaded_image)    # Convert to numpy array

        # Resize both images to ensure they have the same dimensions
        colored_image_resized = colored_image.resize(uploaded_image.shape[1::-1])

        # Calculate L1 loss
        l1_loss = calculate_l1_loss(np.array(colored_image_resized), uploaded_image)
        st.write(f"L1 Loss: {l1_loss/(256*256)}")

# model
# This model was inspired by the following kaggle notebook: https://www.kaggle.com/code/rajeevctrl/imagecolorization-labcolorspace-fastai/notebook

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet

def build_fastai_model(in_channels=1, out_channels=2, image_shape=(224, 224)):
    model_body = create_body(resnet34, n_in=in_channels, cut=-2)
    model = DynamicUnet(encoder=model_body, n_out=out_channels, img_size=image_shape)
    return model

# Normalization code
input_transforms = [
    transform_expand_dim(axis=2),
    to_channel_first,
    transform_normalize(106.09226837631537, 71.83302103485308)
]

output_transforms = [
    to_channel_first,
    transform_normalize(132.98195538604114, 16.854253771660943)
]
def transform_expand_dim(axis):
    def fn(arr):
        arr = np.expand_dims(arr, axis=axis)
        return arr
    return fn

def transform_multiply(mul):
    def fn(arr):
        arr = arr * mul
        return arr
    return fn

def transform_divide(div):
    def fn(arr):
        arr = arr / div
        return arr
    return fn

def transform_normalize(mean, std):
    def normalize(x):
        return (x - mean) / std
    return normalize

def transform_normalize_min_max(min_val, max_val):
    def normalize_min_max(x):
        return (x - min_val) / (max_val - min_val)
    return normalize_min_max

def model_parameters_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

# Initial data loading code

def display(img):
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(img)
    plt.show()

def display_colored(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def rgb_image(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.zeros(shape)
    img[:,:,0] = l[:,:]
    img[:,:,1:]= ab
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def rgb_image_A(l, ab):
    shape = (l.shape[0],l.shape[1],3)
    img = np.ones(shape)

    img[:,:,1]= ab[:,:,0]
    img[:,:,0] = np.ones((224,224)) * 128

    print("The minimum A is", min(ab[:,:,0].flatten()))
    print("The maximum A is", max(ab[:,:,0].flatten()))
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def rgb_image_B(l, ab):
    shape = (l.shape[0],l.shape[1],3)

    img = np.zeros(shape)
    img[:,:,2]= ab[:,:,1]
    img[:,:,0] = np.ones((224,224)) * 128

    print("The minimum B is", min(ab[:,:,1].flatten()))
    print("The maximum B is", max(ab[:,:,1].flatten()))
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

class LABImageDataset(Dataset):
    def __init__(self, L_data, ab_data):
        self.L_data = L_data
        self.ab_data = ab_data

    def __len__(self):
        return len(self.L_data)

    def __getitem__(self, idx):
        L = self.L_data[idx]
        ab = self.ab_data[idx]
        return L, ab

def get_data_loader(batch_size = 64):


    # npArray_ab1 = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/ab/ab/ab1.npy")
    # npArray_ab2 = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/ab/ab/ab2.npy")
    # npArray_ab3 = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/ab/ab/ab3.npy")
    # npArray_l = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/l/gray_scale.npy")
    # npArray_ab3 = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/ab/ab/ab3.npy")
    # npArray_l = np.load("/content/drive/MyDrive/Image Colorization/Image Colorization/l/gray_scale.npy")

    npArray_l = np.load("gray_scale.npy")
    npArray_ab1 = np.load("ab/ab1.npy")
    npArray_ab2 = np.load("ab/ab2.npy")
    npArray_ab3 = np.load("ab/ab3.npy")

    # npArray_l = npArray_l[:10000]
    npArray_ab = np.concatenate((npArray_ab1, npArray_ab2, npArray_ab3))

    # print(npArray_l[0])
    # print(npArray_ab[0])
    # print(npArray_ab[0][:][:][1:].shape)
    # display(npArray_l[0])
    # display_colored(rgb_image(npArray_l[0], npArray_ab[0]))

    # npArray_ab = npArray_ab[:1000]
    # npArray_ab = npArray_ab.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    # npArray_l = npArray_l[:1000]
    # new_shape = (1000, 224, 224,1)
    # npArray_l = npArray_l.reshape(new_shape)

    lab_dataset = LABImageDataset(npArray_l, npArray_ab)
    print(npArray_ab.shape)
    # print(npArray_ab1.shape)
    # print(npArray_ab2.shape)
    # print(npArray_ab3.shape)
    print(npArray_l.shape)
    relevant_indices = np.arange(1, 25000)
    # relevant_indices = np.arange(1, 5000)
    #relevant_indices = np.arange(1, 10000)

    print(lab_dataset)

    np.random.seed(1000)

    np.random.shuffle(relevant_indices)

    split = int(len(relevant_indices) * 0.8)

    split_in_val_test = split + int(len(relevant_indices) * 0.1)

    train_indices = relevant_indices[:split]
    val_indices = relevant_indices[split: split_in_val_test]
    test_indices = relevant_indices[split_in_val_test:]

    # np.random.shuffle(train_indices)
    # np.random.shuffle(test_indices)
    # np.random.shuffle(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = train_sampler)
    val_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = val_sampler)
    test_loader = DataLoader(lab_dataset, batch_size = batch_size, sampler = test_sampler)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_data_loader(64)

for images, labels in train_loader:
   display(images[:][0])
   display_colored(rgb_image(images[:][0], labels[:][0]))
   display_colored(rgb_image_A(images[:][0], labels[:][0]))
   display_colored(rgb_image_B(images[:][0], labels[:][0]))
   break
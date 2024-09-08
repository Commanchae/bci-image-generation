import PIL.Image
import pygame
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import numpy as np
import time
### Transforms  ###
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale()
])

to_pil_transform = transforms.ToPILImage()

### Functions ###
def resize_image(img, resize=(128, 128)):
    img = img.resize(resize)
    mode, size, data = img.mode, img.size, img.tobytes()
    return mode, size, data

def get_pygame_image(mode, size, data):
    return pygame.image.fromstring(data, size, mode)
    
def get_next_image(iterator):
    image_tensor, output = next(iterator)

    image_PIL = to_pil_transform(image_tensor[0]).convert('RGB')
    image_mode, image_size, image_data = resize_image(image_PIL)
    pygame_image = get_pygame_image(image_mode, image_size, image_data)

    return pygame_image, image_tensor

def padtrim_sample(samples: np.ndarray, sampling_frequency: int, duration: float) -> np.ndarray:
        # Get the number of channels and timesteps.
        C, T = samples.shape

        # Calculate the desired length.
        target_length = int(sampling_frequency * duration)

        # If the sample is shorter than the desired length, pad it.
        if T < target_length:
            pad_width = ((0, 0), (0, target_length - T))
            samples = np.pad(samples, pad_width, 'constant')
        else:
            # Else, if the sample is longer than the desired length, trim it.
            samples = samples[:, :target_length]
        
        # Return the padded/trimmed sample.
        return samples

def draw_crosshair(window):
    WHITE = (255, 255, 255)
    crosshair_length = 20
    crosshair_thickness = 2
    crosshair_color = WHITE
    center_x, center_y = WIDTH // 2, HEIGHT // 2

    # Draw horizontal line
    pygame.draw.line(screen, crosshair_color, 
                     (center_x - crosshair_length // 2, center_y), 
                     (center_x + crosshair_length // 2, center_y), 
                     crosshair_thickness)

    # Draw vertical line
    pygame.draw.line(screen, crosshair_color, 
                     (center_x, center_y - crosshair_length // 2), 
                     (center_x, center_y + crosshair_length // 2), 
                     crosshair_thickness)
    
### Data ###
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
cifar_iterator = iter(train_dataloader)

### Pygame Variables ###
WIDTH, HEIGHT = 600, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

### Loop Variables ###
images_to_show = 10
images_shown = 0
loop_duration = 5
show_duration = 2
delay_until_recording_duration = 1
recording_duration = loop_duration - (show_duration + delay_until_recording_duration)

load_new_image = True
current_image = None
start_ticks = 0
samples, targets = [], []
sample = []
current_image = {
    'image': None,
    'tensor': None
}

### BCI Variables ###
SAMPLING_FREQUENCY = 256
i = 0
while images_shown < images_to_show and running:

    clock.tick(256)
    screen.fill((0, 0, 0))
    draw_crosshair(screen)
    i += 1
    # maybe i can check if 1/256 seconds have passed since last iteration.
    if current_image['image'] is None:
        current_image['image'], current_image['tensor'] = get_next_image(cifar_iterator)
        start_ticks = time.perf_counter()
        sample = []

    if current_image['image'] is not None:
        seconds = (time.perf_counter()- start_ticks)
        if seconds <= show_duration:
            screen.blit(current_image['image'], (236, 236))

        elif seconds >= show_duration + delay_until_recording_duration and seconds < loop_duration:

            # Appends random for now, but will pull from EEG stream in the future.
            sample.append(torch.rand(size=(5,)))

            #sample_tmp, _ = inlet.pull_sample()
            #sample.append(sample_tmp)
            pass
        elif seconds >= loop_duration:
            # Save the training data.   
            sample = np.array(sample) # Shape: [duration*sampling_frequency, 5] (5 channels from Muse 2). Example shape: [512, 5]
            sample = sample.T # Shape: [5, duration*sampling_frequency]. Example shape: [5, 512]
            sample = np.ascontiguousarray(sample) # Converts array to contiguous array for faster processing.
            sample = padtrim_sample(samples=sample, sampling_frequency=SAMPLING_FREQUENCY, duration=recording_duration)

            targets.append(current_image['tensor'])
            samples.append(sample)

            # Reset.
            current_image['image'], current_image['tensor'] = None, None
            images_shown += 1

           

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(len(samples))
            running = False
    pygame.display.update()
# screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
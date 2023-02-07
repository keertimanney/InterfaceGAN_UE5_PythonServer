import os.path
import io
import IPython.display
import numpy as np
import cv2
import PIL.Image

import torch

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate


def build_generator(model_name):
  """Builds the generator by model name."""
  gan_type = MODEL_POOL[model_name]['gan_type']
  if gan_type == 'pggan':
    generator = PGGANGenerator(model_name)
  elif gan_type == 'stylegan':
    generator = StyleGANGenerator(model_name)
  return generator


def sample_codes(generator, num, latent_space_type='Z', seed=0):
  """Samples latent codes randomly."""
  np.random.seed(seed)
  codes = generator.easy_sample(num)
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
    codes = generator.get_value(generator.model.mapping(codes))
  return codes


def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  print('we are here')
  PIL.Image.fromarray(fused_image).save('samplekeerti.jpeg')
  # dont show image but save it
  # image = PIL.Image.fromarray(fused_image)
  # image.show() # This is where we show the image...
  # Save the image somewhere over here... at locatoin X
  # Return on webcall the location of the image
  # im_data = data.getvalue()
  # disp = IPython.display.display(IPython.display.Image(im_data))
  # im_data.save('sample.jpeg')
  return 'samplekeerti.jpeg'

#@title { display-mode: "form", run: "auto" }
model_name = "stylegan_ffhq" #@param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
latent_space_type = "W" #@param ['Z', 'W']

generator = build_generator(model_name)

ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
boundaries = {}
for i, attr_name in enumerate(ATTRS):
  boundary_name = f'{model_name}_{attr_name}'
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
  else:
    boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')

#@title { display-mode: "form", run: "auto" }

def handle_input(noise_seed, age, gender, eyeglasses):
    num_samples = 3 #@param {type:"slider", min:1, max:8, step:1}
    # noise_seed = 0 #@param {type:"slider", min:0, max:1000, step:1}

    latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        synthesis_kwargs = {'latent_space_type': 'W'}
    else:
        synthesis_kwargs = {}

    images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
    # imshow(images, col=num_samples)

    #@title { display-mode: "form", run: "auto" }

    # age = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    # eyeglasses = 0 #@param {type:"slider", min:-2.9, max:3.0, step:0.1}
    # gender = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    pose = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    smile = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}

    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(ATTRS):
        new_codes += boundaries[attr_name] * eval(attr_name)


    from IPython.display import display, Image

    new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
    display(new_images, col=num_samples)
    # returning the path here instead of showing the image
    return imshow(new_images, col=num_samples)

from aiohttp import web

async def hello(request):
    print(dir(request))
    # print(request.params)
    # noise_seed, age, eyeglasses, gender,
    noise_seed = request.rel_url.query['noise_seed']
    age = request.rel_url.query['age']
    gender = request.rel_url.query['gender']
    eyeglasses = request.rel_url.query['eyeglasses']
    path = handle_input(int(noise_seed), int(age), int(gender), int(eyeglasses))
    return web.Response(text = f'Hello world {noise_seed} {age} {gender} {eyeglasses} {path}')

async def hello_world(request):
  return web.Response(text = f'hello world')

app = web.Application()
app.add_routes([web.get('/hello_world', hello_world)])
app.add_routes([web.get('/keerti', hello)])
web.run_app(app)
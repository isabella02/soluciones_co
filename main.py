#librerias necesarias 
import autograd.numpy as np  
from autograd import grad
import matplotlib.pyplot as plt
import time
import skimage
import skimage.io as sio

#La siguiente función ejecuta la propagación del método del espectro angular(técnica para modelar la propagación de un campo de ondas) dado un frente de onda inicial y sus dimensiones, longitud de onda de la luz y distancia de propagación. 

def asm_prop(wavefront, length=32.e-3, \
wavelength=550.e-9, distance=10.e-3):
  # compara si las dimensiones del frente de la onda son =2 o =3
  if len(wavefront.shape) == 2: 
   dim_x, dim_y = wavefront.shape
  elif len(wavefront.shape) == 3:
   number_samples, dim_x, dim_y = wavefront.shape
  else: 
    print('only 2D wavefronts or array of 2D wavefronts supported')
  assert dim_x == dim_y, 'wavefront should be square'# muestra una excepcion cuando no son iguales
  # luego se hacen las respectivas operaciones y calculos y al final se arroja el nuevo frente de onda
  px = length / dim_x
  l2 = (1/wavelength)**2
  fx = np.linspace(-1/(2*px), 1/(2*px)-1/(dim_x *px), dim_x)
  fxx, fyy = np.meshgrid(fx,fx)
  q = l2-fxx**2-fyy**2
  q[q<0] = 0.0
  h = np.fft.fftshift(np.exp(1.j * 2 * np.pi * distance * np.sqrt(q)))
  fd_wavefront = np.fft.fft2(np.fft.fftshift(wavefront))
  if len(wavefront.shape) == 3:
    fd_new_wavefront = h[np.newaxis,:,:] * fd_wavefront
    New_wavefront = np.fft.ifftshift(np.fft.ifft2(\
    fd_new_wavefront))[:,:dim_x,:dim_x]
  else:
    fd_new_wavefront = h * fd_wavefront
    new_wavefront = np.fft.ifftshift(np.fft.ifft2(\
    fd_new_wavefront))[:dim_x,:dim_x]
  return new_wavefront

#con esta funcion propagaremos nuestro haz a través de una serie de imágenes de objetos de fase espaciadas uniformemente(como un haz de luz atravez de varias capas de vidrio) usando autograd

# y Recogeremos la luz en el equivalente de un sensor de imagen.

def onn_layer(wavefront, phase_objects, d=100.e-3):
  for ii in range(len(phase_objects)):
    wavefront = asm_prop(wavefront * phase_objects[ii], distance=d)
  return wavefront

#La clave para entrenar un modelo en Autograd es definir una función que devuelva una pérdida escalar. Luego, esta función de pérdida puede integrarse en la función de graduación de Autograd para calcular gradientes.La funcion es la siguiente, con parametros definidos

def get_loss(wavefront, y_tgt, phase_objects, d=100.e-3):
  img = np.abs(onn_layer(wavefront, phase_objects, d=d))**2
  mse_loss = np.mean( (img - y_tgt)**2 + np.abs(img-y_tgt) )
  return mse_loss

#se halla el gradiente con el resulado escalar obtenido

get_grad = grad(get_loss, argnum=2)

#leemos la imagen de destino y configuremos el frente de onda de entrada. 
# target image
tgt_img = sio.imread('./smiley1.png')[:, :, 0]
y_tgt = 1.0 * tgt_img / np.max(tgt_img)
# configurar el frente de onda de entrada (una onda plana de 16 mm de apertura)
dim = 128
side_length = 32.e-3
aperture = 8.e-3
wavelength = 550.e-9
k0 = 2*np.pi / wavelength
px = side_length / dim
x = np.linspace(-side_length/2, side_length/2-px, dim)
xx, yy = np.meshgrid(x,x)
rr = np.sqrt(xx**2 + yy**2)
wavefront = np.zeros((dim,dim)) * np.exp(1.j*k0*0.0)
wavefront[rr <= aperture] = 1.0

#A continuación, se define la tasa de aprendizaje, la distancia de propagación y los parámetros del modelo. 
lr = 1e-3
dist = 50.e-3
phase_objects = [np.exp(1.j * np.zeros((128,128))) \
 for aa in range(32)]
losses = [] 

#Llamamos a la función de gradiente que definimos anteriormente (que es una transformación de función de la función que escribimos para calcular la pérdida) y aplicamos los gradientes resultantes a los parámetros de nuestro modelo. 

for step in range(128):
  my_grad = get_grad(wavefront, y_tgt, phase_objects, d=dist)
  for params, grads in zip(phase_objects, my_grad):
    params -= lr * np.exp( -1.j * np.angle(grads))
  loss = get_loss(wavefront, y_tgt, phase_objects,d=dist)
  losses.append(loss)
  img = np.abs(onn_layer(wavefront, phase_objects))**2
  print('loss at step {} = {:.2e}, lr={:.3e}'.format(step, loss, lr))
  fig = plt.figure(figsize=(12,7))
  plt.imshow(img / 2.0, cmap='jet')
  #el código guardará una serie de cifras que representan la salida de la red óptica a medida que se acerca cada vez más a la imagen de destino
  plt.savefig('./smiley_img{}.png'.format(step))
  plt.close(fig)

fig = plt.figure(figsize=(7,4))
plt.plot(losses, lw=3)
plt.savefig('./smiley_losses.png')




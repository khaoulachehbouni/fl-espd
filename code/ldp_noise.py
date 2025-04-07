import pickle
import pandas as pd
import numpy as np
from random import gauss




def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5

    return [x/mag for x in vec]






def add_noise(embeddings,noisy_embeddings,epsilon):
  shape = 768 #bert dimensionality
  scale = 1 / epsilon
  for embedding in embeddings:
    l = np.random.gamma(shape, scale)
    v = make_rand_vector(768)
    noise_to_add = l*np.asarray(v)
    noisy_array = embedding + noise_to_add
    noisy_embeddings.append(noisy_array)
  return noisy_embeddings





with open('all_embeddings.pkl', 'rb') as e:
  all_embeddings = pickle.load(e)

embeddings_eps20 = []
embeddings_eps20 = add_noise(all_embeddings,embeddings_eps20,20)

with open('embeddings_eps20.pkl', 'wb') as d:
  pickle.dump(embeddings_eps20, d)
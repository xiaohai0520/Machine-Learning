#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

imgs = np.load("outs.npz")
plt.figure()
plt.subplot(221)
plt.imshow(imgs["input"])
plt.title("Input Image")
plt.subplot(222)
plt.imshow(imgs["content"])
plt.title("Content Image")
plt.subplot(223)
plt.imshow(imgs["style"])
plt.title("Style Image")
plt.subplot(224)
plt.imshow(imgs["output"])
plt.title("Output Image")
plt.savefig("dancing.jpg", bbox_inch="tight")
plt.show()

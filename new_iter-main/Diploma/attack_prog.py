from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import foolbox
from art.defences.detector.evasion import EvasionDetector
from art.defences.detector.poison import PoisonFilteringDefence

model = load_model('diploma_model.keras')
fmodel = foolbox.models.TensorFlowModel(model,bounds=(0,1), device=None, preprocessing=None)
model.summary()

classes = ['lawn-mower', 'rocket', 'car', 'tank', 'tractor', 'beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium', 'fish', 'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates' 'apples', 'tomato','mushrooms', 'oranges', 'pears', 'sweet', 'peppers', 'clock', 'computer', 'keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'raccoon']
img_name = 'car1.jpg'
img = image.load_img(img_name, target_size=(32, 32)) 
plt.imshow(img)
plt.show()

print(fmodel)

img = np.array(img)
img = img.astype('float32')
img /= 255
model.predict(img.reshape(1,32,32,3))
img = np.expand_dims(img, axis=0)
predict = model.predict(img)
predict = np.argmax(predict, axis=1)
print(classes[predict[0]])

attackDeepFool = foolbox.attacks.L2DeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss='crossentropy')
adversalDeepFool = attackDeepFool(img.reshape(32,32,3), 0)
probsDeepFool = model.predict(attackDeepFool.reshape(1,10))
print(probsDeepFool)
print(np.argmax(probsDeepFool))

attackGBA = foolbox.attacks.GaussianBlurAttack()
adversalGBA = attackGBA(img.reshape(32,32,3), 0)
probsGBA = model.predict(attackGBA.reshape(1,10))
print(probsGBA)
print(np.argmax(probsGBA))

attackLFG = foolbox.attacks.LinfFastGradientAttack(random_start=False)
adversalLFG = attackLFG(img.reshape(32,32,3), 0)
probsLFG = model.predict(adversalLFG.reshape(1,10))
print(probsLFG)
print(np.argmax(probsLFG))

attackBoundary = foolbox.attacks.BoundaryAttack(steps=25000, spherical_step=0.01, source_step=0.01, source_step_convergance=1e-07, step_adaptation=1.5, update_stats_every_k=10)
adversalLBoundary = attackBoundary(img.reshape(32,32,3), 0)
probsBoundary = model.predict(adversalLBoundary.reshape(1,10))
print(probsBoundary)
print(np.argmax(probsBoundary))

attackSAPN = foolbox.attacks.SaltAndPepperNoiseAttack(steps=1000, across_channels=True, channel_axis=None)
adversalSAPN = attackSAPN(img.reshape(32,32,3), 0)
probsSAPN = model.predict(adversalSAPN.reshape(1,10))
print(probsSAPN)
print(np.argmax(probsSAPN))

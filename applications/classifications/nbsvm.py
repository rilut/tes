from keras.layers import Input, Embedding, Activation, Lambda, Flatten
from keras.models import Model
import keras.backend as K

nf = 900
ny = 20

inp = Input(shape=(nf,))
w_ = Embedding(nf + 1, 1,  embeddings_initializer='uniform', mask_zero=True)(inp)
r_ = Embedding(nf + 1, ny, embeddings_initializer='zeros')(inp)

# out = ((w + 0.4) * r / 10)
out = Lambda(lambda (w, r): K.sum((w + 0.4) * r / 10, axis=1))([w_, r_])
out = Activation('softmax')(out)

model = Model(inp, out)
model.summary()

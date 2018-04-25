from keras.models import Model
from keras.applications import inception_v3, inception_resnet_v2
from keras.layers import Dense, ELU, Dropout, BatchNormalization
filename = "finalEnsemble/inception_v4_difftop_832.h5"
model = inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(256,256,3), pooling='avg')
x = model.output
x = Dense(512)(x)
x = BatchNormalization(name="lol")(x)
x = ELU()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(18, activation='softmax')(x)

new_model = Model(inputs=model.input, outputs=pred)
new_model.load_weights(filename)
new_model.save(filename)

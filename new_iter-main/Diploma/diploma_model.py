import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist
from art.defences.detector.evasion import EvasionDetector, BinaryInputDetector
from art.defences.detector.poison import PoisonFilteringDefence, SpectralSignatureDefense

tf.random.set_seed(1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class Model():

    def __init__(self):
        pass

    def get_model():
        inputs = keras.Input(shape=(32, 32, 3), name="img")
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        block_1_output = layers.MaxPooling2D(3)(x)

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name="toy_resnet")

        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        return model

    
model_sample = Model.get_model()

model_sample.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.2)

save_model = model_sample.save('diploma_model.keras')

print(model_sample.evaluate(x_test, y_test))

def sec_model(x_train, y_train):

    evasion_base = EvasionDetector(x = x_train, batch_size = 64)
    eb_result = evasion_base()
    print(eb_result)
    evasion_input_detector = BinaryInputDetector(x = x_train, batch_size= 64)
    eid_result = evasion_input_detector()
    print(eid_result)
    poison_base = PoisonFilteringDefence(x_train=x_train, y_train=y_train)
    pb_result = poison_base()
    print(pb_result)
    poison_signature_detector =  SpectralSignatureDefense()
    psd_result = poison_signature_detector(x_train=x_train, y_train=y_train, expected_pp_poison= 0.33, batch_size= 128, eps_multiplier = 1.5)
    print(psd_result)


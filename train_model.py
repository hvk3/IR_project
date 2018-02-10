import data_generator
import models

if __name__ == "__main__":
    trainDatagen, valDatagen = data_generator.get_no_sent2vec_gen()
    model = models.no_sent2vec_training_model()
    batchSize = 16
    model.fit_generator(datagen
        ,steps_per_epoch=1000 // batchSize
        ,epochs=20
        ,validation_data=valGen
        ,validation_steps=800 // batchSize)

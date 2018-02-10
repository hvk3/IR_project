from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adadelta


def no_sent2vec_training_model(EMBEDDING_SIZE, NUM_COMMENTS):
    # Metadata based inputs
    input_metadata = Input(shape=(6,))
    dense1_metadata = Dense(64, activation='relu')(input_metadata)
    dense2_metadata = Dense(32, activation='relu')(dense1_metadata)

    # Raw video based features
    input_audio = Input(shape=(128,))
    dense1_audio = Dense(512, activation='relu')(input_audio)
    input_video = Input(shape=(1024,))
    dense1_video = Dense(2048, activation='relu')(input_video)

    # Combine A/V layers
    dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
    dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)

    # Text-based features
    input_description = Input(shape=(EMBEDDING_SIZE,))
    input_comments = Concatenate()([ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS)])
    input_channelName = Input(shape=(EMBEDDING_SIZE,))

    dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
    dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(input_comments)
    dense1_channelName = DenseConcatenate()(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
    input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
    dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)

    # Dense layers on all concatenated inputs to get output
    dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
    dense5_allinputs = Dense(4096,  activation='relu')(dense4_allinputs)
    dense6_allinputs = Dense(2048,  activation='relu')(dense5_allinputs)
    model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

    # Define model inputs (ordered) for model
    model_inputs = [input_metadata, input_audio, input_video, input_description]
    model_inputs += input_comments
    model_inputs += [input_channelName]

    # Define loss, compile model
    model = keras.models.Model(inputs=model_inputs, outputs=model_output)
    opt = Adadelta(lr=1)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

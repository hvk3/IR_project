from keras.layers import Input, Dense, Concatenate, Dropout,\
    BatchNormalization, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adadelta, RMSprop
from keras.models import Model
import sys


def model(EMBEDDING_SIZE,
          NUM_COMMENTS,
          vocab_size=0,
          seqLength=0,
          batch_norm=False,
          dropout=False,
          use_audio=True,
          use_video=True,
          use_metadata=True,
          use_channel=True,
          use_description=True):
    # Metadata based inputs
    input_metadata = Input(shape=(6,))
    dense1_metadata = Dense(64, activation='relu')(input_metadata)
    dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
    if (batch_norm):
        dense2_metadata = BatchNormalization()(dense2_metadata)
    # Make output empty (all 0s)
    if (not use_metadata):
        dense2_metadata = Dropout(1)(dense2_metadata)

    # Raw audio/video based features
    input_audio = Input(shape=(128,))
    dense1_audio = Dense(512, activation='relu')(input_audio)
    if (batch_norm):
        dense1_audio = BatchNormalization()(dense1_audio)
    if (not use_audio):
        dense1_audio = Dropout(1)(dense1_audio)
    input_video = Input(shape=(1024,))
    dense1_video = Dense(2048, activation='relu')(input_video)
    if (batch_norm):
        dense1_video = BatchNormalization()(dense1_video)
    if (not use_video):
        dense1_video = Dropout(1)(dense1_video)

    dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
    dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
    if (batch_norm):
        dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

    # Text-based features
    input_description = Input(shape=(EMBEDDING_SIZE,))
    input_comments = [
        Input(shape=(EMBEDDING_SIZE,)) for _ in xrange(NUM_COMMENTS)
    ]
    concat_input_comments = Concatenate()(input_comments)
    input_channelName = Input(shape=(EMBEDDING_SIZE,))

    dense1_description = Dense(
        EMBEDDING_SIZE / 2,
        activation='relu'
    )(input_description)
    if (batch_norm):
        dense1_description = BatchNormalization()(dense1_description)
    if (not use_description):
        dense1_description = Dropout(1)(dense1_description)
    dense1_comments = Dense(
        NUM_COMMENTS / 2 * EMBEDDING_SIZE / 2,
        activation='relu'
    )(concat_input_comments)
    if (batch_norm):
        dense1_comments = BatchNormalization()(dense1_comments)
    if (NUM_COMMENTS == 0):
        dense1_comments = Dropout(1)(dense1_comments)
    dense1_channelName = Dense(
        EMBEDDING_SIZE / 2,
        activation='relu'
    )(input_channelName)
    if (batch_norm):
        dense1_channelName = BatchNormalization()(dense1_channelName)
    if (not use_channel):
        dense1_channelName = Dropout(1)(dense1_channelName)
    input_alltextual = Concatenate()([
        dense1_description,
        dense1_comments,
        dense1_channelName
    ])
    dense2_alltextual = Dense(
        (NUM_COMMENTS * EMBEDDING_SIZE) / 16
    )(input_alltextual)
    if (batch_norm):
        dense2_alltextual = BatchNormalization()(dense2_alltextual)

    dense4_allinputs = Concatenate()([
        dense2_metadata,
        dense3_audiovideo,
        dense2_alltextual
    ])
    dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
    if (dropout):
        dense5_allinputs = Dropout(0.5)(dense5_allinputs)
    dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
    if (dropout):
        dense6_allinputs = Dropout(0.5)(dense6_allinputs)
    model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)
    if (seqLength > 0):
        repeated = RepeatVector(seqLength)(model_output)
        decoder_lstm_1 = LSTM(
            200,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True
        )(repeated)
        # decoder_lstm_2 = LSTM(
        #     500,
        #     recurrent_dropout=0.2,
        #     return_sequences=True
        # )(decoder_lstm_1)
        model_output = TimeDistributed(
            Dense(vocab_size, activation='softmax')
        )(decoder_lstm_1)
        # )(decoder_lstm_2)

    model_inputs = [
        input_metadata,
        input_audio,
        input_video,
        input_description
    ] + input_comments + [input_channelName]
    model = Model(inputs=model_inputs, outputs=model_output)

    if (seqLength > 0):
        model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(lr=5e-4),
            metrics=['accuracy']
        )
    else:
        model.compile(loss='mean_squared_error', optimizer='adadelta')
    return model


def customTrain(model, dataGen, epochs, batch_size, valRatio=0.2):
    steps_per_epoch = 320000 / batch_size
    for _ in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        for i in range(steps_per_epoch):
            (x_train, y_train), (x_test, y_test) = dataGen.next()
            train_metrics = model.train_on_batch(x_train, y_train)
            train_loss += train_metrics[0]
            train_acc += train_metrics[1]
            val_metrics = model.test_on_batch(x_test, y_test)
            val_loss += val_metrics[0]
            val_acc += val_metrics[1]
            sys.stdout.write("%d / %d : Tr loss: %f, Tr acc: %f, Vl loss: %f, Vl acc: %f  \r" % (i+1, steps_per_epoch, train_loss/(i+1), train_acc/(i+1), val_loss/(i+1), val_acc/(i+1)))
            sys.stdout.flush()
        print("\n")



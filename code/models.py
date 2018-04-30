from keras.layers import Input, Dense, Concatenate, Dropout,\
    BatchNormalization, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
import keras.optimizers as optimizers
import numpy as np
import sys


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)

        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(a)
        return weighted_input
        # result = K.sum(weighted_input, axis=1)

        # if self.return_attention:
        #     return [result, a]
        # return result

    def compute_output_shape(self, input_shape):
        # if self.return_attention:
        #     return [(input_shape[0], input_shape[-1]),
        #             (input_shape[0], input_shape[1])]
        # else:
        #     return input_shape[0], input_shape[-1]
        return input_shape[0], input_shape[1], input_shape[-1]

# class AttentionWithContext(Layer):
#     """
#     Attention operation, with a context/query vector, for temporal data.
#     Supports Masking.
#     Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
#     "Hierarchical Attention Networks for Document Classification"
#     by using a context vector to assist the attention
#     # Input shape
#         3D tensor with shape: `(samples, steps, features)`.
#     # Output shape
#         2D tensor with shape: `(samples, features)`.

#     How to use:
#     Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
#     The dimensions are inferred based on the output shape of the RNN.

#     Note: The layer has been tested with Keras 2.0.6

#     Example:
#         model.add(LSTM(64, return_sequences=True))
#         model.add(AttentionWithContext())
#         # next add a Dense layer (for classification/regression) or whatever...
#     """

#     def __init__(self,
#                  W_regularizer=None, u_regularizer=None, b_regularizer=None,
#                  W_constraint=None, u_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):

#         self.supports_masking = True
#         self.init = initializers.get('glorot_uniform')

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.u_regularizer = regularizers.get(u_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.u_constraint = constraints.get(u_constraint)
#         self.b_constraint = constraints.get(b_constraint)

#         self.bias = bias
#         super(AttentionWithContext, self).__init__(**kwargs)

#     def build(self, input_shape):
#         assert len(input_shape) == 3

#         self.W = self.add_weight((input_shape[-1], input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         if self.bias:
#             self.b = self.add_weight((input_shape[-1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)

#         self.u = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_u'.format(self.name),
#                                  regularizer=self.u_regularizer,
#                                  constraint=self.u_constraint)

#         super(AttentionWithContext, self).build(input_shape)

#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None

#     def call(self, x, mask=None):
#         uit = dot_product(x, self.W)

#         if self.bias:
#             uit += self.b

#         uit = K.tanh(uit)
#         ait = dot_product(uit, self.u)

#         a = K.exp(ait)

#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())

#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

#         a = K.expand_dims(a)
#         weighted_input = x * a
#         # return K.sum(weighted_input, axis=1)
#         return weighted_input

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], input_shape[-1]


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
        from keras.layers import Bidirectional, Conv1D, MaxPool1D, Flatten
        repeated = RepeatVector(seqLength)(model_output)
        # conv_1 = Conv1D(filters=128, kernel_size=1, activation='relu')(repeated)
        # maxpool_1 = MaxPool1D(pool_size=2)(conv_1)
        # conv_2 = Conv1D(filters=8, kernel_size=3, activation='relu')(maxpool_1)
        # # maxpool_2 = MaxPool1D(pool_size=2)(conv_2)
        # flattened = Flatten()(maxpool_1)
        # repeated_ = RepeatVector(seqLength)(flattened)
        decoder_lstm_1 = Bidirectional(
            LSTM(
                500,
                dropout=0.8,
                recurrent_dropout=0.8,
                return_sequences=True
            )
        )(repeated)
        decoder_lstm_2 = Bidirectional(
            LSTM(
                1000,
                dropout=0.8,
                recurrent_dropout=0.8,
                return_sequences=True
            )
        )(decoder_lstm_1)
        l_dense_sent = TimeDistributed(Dense(200))(decoder_lstm_2)
        # # l_attn_sent = AttentionWithContext()(l_dense_sent)
        # # l_attn_sent = Attention()(l_dense_sent)
        model_output = TimeDistributed(
            Dense(vocab_size, activation='softmax')
        )(l_dense_sent)
        # )(l_dense_sent)
        # )(l_attn_sent)

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
            optimizer=optimizers.Adam(),
            metrics=['accuracy', 'mse']
        )
    else:
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adadelta())
    return model


def customTrain(model, dataGen, epochs, batch_size, valRatio=0.2, no_acc=False):
    steps_per_epoch = 32000 / batch_size
    for _ in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        for i in range(steps_per_epoch):
            (x_train, y_train), (x_test, y_test) = dataGen.next()
            train_metrics = model.train_on_batch(x_train, y_train)
            if no_acc:
                train_loss += train_metrics
            else:
                train_loss += train_metrics[0]
                train_acc += train_metrics[1]
            val_metrics = model.test_on_batch(x_test, y_test)
            y_predicted = model.predict_on_batch(x_train)
            #for j in xrange(y_predicted.shape[0]):
            #    for k in xrange(y_predicted.shape[1]):
                    # print 'Predicted label:', np.argmax(y_predicted[i][j])
                    # print 'Actual label:', np.argmax(y_test[i][j])
            #        sys.stdout.write('Predicted label: {}\t'.format(
            #            np.argmax(y_predicted[j][k])
            #        ))
            #        sys.stdout.write('Actual label: {}\n'.format(
            #            np.argmax(y_train[j][k])
            #        ))
                    # sys.stdout.flush()
            if no_acc:
                val_loss += val_metrics
            else:
                val_loss += val_metrics[0]
                val_acc += val_metrics[1]
            #sys.stdout.write("\n%d / %d : Tr loss: %f, Tr acc: %f, Vl loss: %f, Vl acc: %f  \n" % (
            if no_acc:
                sys.stdout.write("\r%d / %d : Tr loss: %f, Vl loss: %f  \r" % (
                i + 1,
                steps_per_epoch,
                train_loss / (i + 1),
                val_loss / (i + 1)))
            else:
                sys.stdout.write("\r%d / %d : Tr loss: %f, Tr acc: %f, Vl loss: %f, Vl acc: %f  \r" % (
                    i + 1,
                    steps_per_epoch,
                    train_loss / (i + 1),
                    train_acc / (i + 1),
                    val_loss / (i + 1),
                    val_acc / (i + 1)))
            sys.stdout.flush()
        print("\n")


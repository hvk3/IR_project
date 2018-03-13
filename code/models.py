from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adadelta
from keras.models import Model


def sent2vec_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# LSTM for generating title; ref : https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
	decoder_lstm = LSTM(EMBEDDING_SIZE, return_sequences=True, return_state=True)(model_output)
	decoder_outputs = Dense(26, activation='softmax')(decoder_lstm)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_video, input_description]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=decoder_outputs)
	# opt = Adadelta(lr=1)
	# model.compile(loss='mean_squared_error', optimizer=opt)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	return model


def no_sent2vec_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_video, input_description]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_metadata_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_audio, input_video, input_description]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_audio_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw video based features
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = dense1_video
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_video, input_description]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_video_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)

	# Combine A/V layers
	dense2_audiovideo = dense1_audio
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_description]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_description_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_comments, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_video]
	model_inputs += input_comments
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_comments_model(EMBEDDING_SIZE, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_channelName = Input(shape=(EMBEDDING_SIZE,))

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_channelName = Dense(EMBEDDING_SIZE/2, activation='relu')(input_channelName)
	if (add_batch_norm):
		dense1_channelName = BatchNormalization()(dense1_channelName)
	input_alltextual = Concatenate()([dense1_description, dense1_channelName])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_video, input_description]
	model_inputs += [input_channelName]

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


def no_channelname_model(EMBEDDING_SIZE, NUM_COMMENTS, add_batch_norm=False, add_dropout=False):
	# Metadata based inputs
	input_metadata = Input(shape=(6,))
	dense1_metadata = Dense(64, activation='relu')(input_metadata)
	dense2_metadata = Dense(32, activation='relu')(dense1_metadata)
	if (add_batch_norm):
		dense2_metadata = BatchNormalization()(dense2_metadata)

	# Raw audio/video based features
	input_audio = Input(shape=(128,))
	dense1_audio = Dense(512, activation='relu')(input_audio)
	if (add_batch_norm):
		dense1_audio = BatchNormalization()(dense1_audio)
	input_video = Input(shape=(1024,))
	dense1_video = Dense(2048, activation='relu')(input_video)
	if (add_batch_norm):
		dense1_video = BatchNormalization()(dense1_video)

	# Combine A/V layers
	dense2_audiovideo = Concatenate()([dense1_audio, dense1_video])
	dense3_audiovideo = Dense(1024, activation='relu')(dense2_audiovideo)
	if (add_batch_norm):
		dense3_audiovideo = BatchNormalization()(dense3_audiovideo)

	# Text-based features
	input_description = Input(shape=(EMBEDDING_SIZE,))
	input_comments = [ Input(shape=(EMBEDDING_SIZE,)) for _ in range(NUM_COMMENTS) ]
	concat_input_comments = Concatenate()(input_comments)

	dense1_description = Dense(EMBEDDING_SIZE/2, activation='relu')(input_description)
	if (add_batch_norm):
		dense1_description = BatchNormalization()(dense1_description)
	dense1_comments = Dense(NUM_COMMENTS/2 * EMBEDDING_SIZE/2, activation='relu')(concat_input_comments)
	if (add_batch_norm):
		dense1_comments = BatchNormalization()(dense1_comments)
	input_alltextual = Concatenate()([dense1_description, dense1_comments])
	dense2_alltextual = Dense(NUM_COMMENTS/4 * EMBEDDING_SIZE/4)(input_alltextual)
	if (add_batch_norm):
		dense2_alltextual = BatchNormalization()(dense2_alltextual)

	# Dense layers on all concatenated inputs to get output
	dense4_allinputs = Concatenate()([dense2_metadata, dense3_audiovideo, dense2_alltextual])
	dense5_allinputs = Dense(4096, activation='relu')(dense4_allinputs)
	if (add_dropout):
		dense5_allinputs = Dropout(0.5)(dense5_allinputs)
	dense6_allinputs = Dense(2048, activation='relu')(dense5_allinputs)
	if (add_dropout):
		dense6_allinputs = Dropout(0.5)(dense6_allinputs)
	model_output = Dense(EMBEDDING_SIZE)(dense6_allinputs)

	# Define model inputs (ordered) for model
	model_inputs = [input_metadata, input_audio, input_video, input_description]
	model_inputs += input_comments

	# Define loss, compile model
	model = Model(inputs=model_inputs, outputs=model_output)
	opt = Adadelta(lr=1)
	model.compile(loss='mean_squared_error', optimizer=opt)

	return model


if __name__ == "__main__":
	model = no_sent2vec_model(100, 2)
	print model.inputs

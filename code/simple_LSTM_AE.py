import argparse
from keras.layers import LSTM, RepeatVector
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam

parser = argparse.ArgumentParser(description='Simple LSTM AE module')
parser.add_argument('--timesteps', type=int, default=10, help='Number of LSTM timesteps')
parser.add_argument('--hidden_dim', type=int, default=100, help='Vector size from encoder')
parser.add_argument('input_dim', type=int, help='Input size')


def get_LSTM_AE(input_dim, hidden_dim, timesteps):
	model_input = LSTM(input_dim=(timesteps, input_dim), dropout=0.5, units=hidden_dim)
	encoder = RepeatVector(timesteps)(model_input)
	decoder = LSTM(units=timesteps, return_sequences=True)(encoder)
	model = Model(inputs=model_input, outputs=decoder)
	model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
	return model


if __name__ == '__main__':
	args = parser.parse_args()
	model = get_LSTM_AE(args.input_dim, args.hidden_dim, args.timesteps)

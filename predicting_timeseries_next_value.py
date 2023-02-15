import numpy as np
import tensorflow as tf
import pandas as pd
import statistics as stat

newl = '\n'
validation_set_length = 0.
time_step_length = 10
feature_count = 2

dataset_df = pd.read_csv("C:\\Users\\andri\\Documents\\Programming\\klines_all_3 (1).csv",
                         names= ["symbols", "datetime", "close", "volume"])
trading_pairs = set(dataset_df["symbols"]) ^ {"BUSDUSDT", "PAXUSDT", "TUSDUSDT", "USDCUSDT", "EURUSDT", "GBPUSDT", "AUDUSDT", "SUSDUSDT", "COCOSUSDT"}
split_to_pairs: list[np.array] = [np.array(
    (dataset_df[dataset_df["symbols"] == pair_i]).iloc[:, 2:]
) for pair_i in trading_pairs]

case_counts_by_ccy: list[int] = [x.shape[0] for x in split_to_pairs]
print("Case distribution:\n", case_counts_by_ccy)
split_to_pairs = list(filter(
    lambda y: y.shape[0] == stat.mode(case_counts_by_ccy),
    split_to_pairs
))
print(f"Dropped {len(trading_pairs) - len(split_to_pairs)} currencies that had different amount of cases.")
dataframes_stacked: np.array = np.stack(split_to_pairs, axis= 0)

dataset_tensor: tf.TensorSpec(tf.TensorShape([None, 1, 2])) = tf.reshape(tf.convert_to_tensor(
    dataframes_stacked[0, :, :] # Set a custom integer for the first index to perform prediction on any of the currencies
), [dataframes_stacked.shape[1], 1, 2])
dataset_tensor = tf.concat([
    dataset_tensor[x: -1 * (time_step_length - x - 1), ...] if x != time_step_length - 1 else dataset_tensor[x:, ...] for x in range(time_step_length)
], axis= 1)
dataset_tensor = dataset_tensor[0: dataset_tensor.shape[0], :, :]
train_input: tf.TensorSpec(tf.TensorShape([None, 1, 2])) = dataset_tensor[
    0: int((0.95 - validation_set_length) * dataframes_stacked.shape[1]), :, :
]
train_result: tf.TensorSpec(tf.TensorShape([None, 1, 1])) = dataset_tensor[
    0: int((0.95 - validation_set_length) * dataframes_stacked.shape[1]), 0, 0
]
print(f"Train input:{newl}{train_input}{newl}Train reference:{newl}{train_result}")
validation_input: tf.TensorSpec(tf.TensorShape([None, 1, 2])) = dataset_tensor[
    int((0.95 - validation_set_length) * dataframes_stacked.shape[1]): int(0.9 * dataframes_stacked.shape[1]), :, :
]
validation_result: tf.TensorSpec(tf.TensorShape([None, 1, 1])) = dataset_tensor[
    int((0.95 - validation_set_length) * dataframes_stacked.shape[1]): int(0.9 * dataframes_stacked.shape[1]):, 0, 0
]
test_input: tf.TensorSpec(tf.TensorShape([None, 1, 2])) = dataset_tensor[
    int(0.95 * dataframes_stacked.shape[1]):, :, :
]

batch_size = 128 # I: Number of samples has to be a multiple of batch_size
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001, beta_1= 0.9, beta_2= 0.999)
initializer = tf.keras.initializers.RandomNormal(mean= 0.3, stddev= 1.5)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        input_shape= (time_step_length, feature_count),
        units= 12,
        kernel_initializer= initializer,
        activation= 'softmax',
        recurrent_activation= 'sigmoid',
        recurrent_dropout= 0,
        unroll= False,
        use_bias= True,
        bias_initializer= initializer,
        return_sequences= True
    ),
    tf.keras.layers.LSTM(
        units= 12,
        kernel_initializer= initializer,
        activation= 'relu',
        recurrent_activation= 'sigmoid',
        recurrent_dropout= 0,
        unroll= False,
        use_bias= True,
        bias_initializer= initializer
    ),
    tf.keras.layers.Dense(1, activation= 'linear')
])

model.compile(optimizer= optimizer, loss= 'mean_squared_logarithmic_error', metrics= ['mae'], run_eagerly= True)
print(f"Created model{newl}{model.summary()}")
model.fit(x= train_input , y= train_result , epochs= dataset_tensor.shape[0] // batch_size)
predictions: np.array = model.predict(x= test_input, batch_size= batch_size, steps= None, )
print(predictions)
from core import create_batch, CaimanSeriesExtensions, CaimanDataFrameExtensions
from PyQt5 import QtWidgets

# Two sets of params for testing
params_a = \
{
    "fr": 30,
    "decay_time": 0.4,
    "p": 1,
    "nb": 2,
    "rf": 15,
    "K": 4,
    "stride": 6,
    "method_init": "greedy_roi",
    "rolling_sum": True,
    "only_init": True,
    "ssub": 1,
    "tsub": 1,
    "merge_thr": 0.85,
    "min_SNR": 2.0,
    "rval_thr": 0.85,
    "use_cnn": True,
    "min_cnn_thr": 0.99,
    "cnn_lowest": 0.1
}

params_b = \
{
    "fr": 30,
    "decay_time": 0.4,
    "p": 1,
    "nb": 2,
    "rf": 15,
    "K": 4,
    "stride": 6,
    "method_init": "greedy_roi",
    "rolling_sum": True,
    "only_init": True,
    "ssub": 1,
    "tsub": 1,
    "merge_thr": 0.55,
    "min_SNR": 2.0,
    "rval_thr": 0.55,
    "use_cnn": True,
    "min_cnn_thr": 0.99,
    "cnn_lowest": 0.1
}

# Create a new empty batch
df = create_batch(path='/home/kushal/test_batch.pickle')

# Add a CNMF item
df.caiman.add_item(
    algo='cnmf',
    input_movie_path='/home/kushal/caiman_data/example_movies/demoMovie.tif',
    params=params_a
)

# Add a MCorr item
df.caiman.add_item(
    algo='cnmf',
    input_movie_path='/home/kushal/caiman_data/example_movies/demoMovie.tif',
    params=params_b
)


print(df)


# Define some callback functions
def callback_func():
    print("Finished item!")


def callback_func_2():
    print("Finished item 2!")


# Start event loop
app = QtWidgets.QApplication([])

# Start item 0 in an external process
df.iloc[0].caiman.run(callbacks_finished=[callback_func])

# External process is non-blocking
print("I can do other stuff while CNMF is running!")

# Start the other process too while the first one is still running to show proof of principle
df.iloc[1].caiman.run(callbacks_finished=[callback_func_2])
print("I started another process while the first one is still running!")

app.exec()

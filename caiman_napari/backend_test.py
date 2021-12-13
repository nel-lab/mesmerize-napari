from core import create_batch, CaimanSeriesExtensions, CaimanDataFrameExtensions
from PyQt5 import QtWidgets

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

df = create_batch(path='/home/kushal/test_batch.pickle')

df.caiman.add_item(
    algo='cnmf',
    input_movie_path='/home/kushal/caiman_data/example_movies/demoMovie.tif',
    params=params_a
)

df.caiman.add_item(
    algo='cnmf',
    input_movie_path='/home/kushal/caiman_data/example_movies/demoMovie.tif',
    params=params_b
)

print(df)


def callback_func():
    print("Finished item!")


def callback_func_2():
    print("Finished item 2!")


app = QtWidgets.QApplication([])

df.iloc[0].caiman.run(callbacks_finished=[callback_func])

print("I can do other stuff while CNMF is running!")

df.iloc[1].caiman.run(callbacks_finished=[callback_func_2])

app.exec()

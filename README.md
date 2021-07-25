# caiman-napari-prototype
Prototype `caiman` plugin for the `napari` viewer.

## CNMF demo
This plugin interfaces with CaImAn CNMF using `QProcess` and displays the spatial components using `napari.layers.shapes`. `stdout` from the ongoing `QProcess` is shown in the text area on the right.
![cnmf](./screenshots/cnmf_prototype.gif)

## Streaming demo
Napari can stream calcium imaging movies in realtime while pyqtgraph is used to show calcium traces. Please note that due to limitations of GIF images, this GIF doesn't fully represent the high level of responsiveness and interactivity during live streaming.
![streaming](./screenshots/napari_streaming.gif)

# Creating a working environment
The installation of `napari` and `caiman` in the same env is a work in progress. This order of steps worked for me on Ubuntu.

Install `napari` first and then `caiman`

Clone `napari`, install requirements, and install in editable mode:

```bash
git clone https://github.com/napari/napari.git
cd napari
pip install -r requirements.txt
pip install -e ".[all]"
```

Install `caiman` from my fork (dependency issues related to the latest versions of `h5py` and `tensorflow`, should probably make a PR after figuring it out):

```bash
git clone https://github.com/kushalkolar/CaImAn.git
cd CaImAn
pip install -r requirements.txt
```

You will see the following dependency issues after the installation:

```
hdmf 2.5.8 requires h5py<3,>=2.9, but you'll have h5py 3.3.0 which is incompatible.
hdmf 2.5.8 requires numpy<1.21,>=1.16, but you'll have numpy 1.21.0 which is incompatible.
pynwb 1.5.1 requires h5py<3,>=2.9, but you'll have h5py 3.3.0 which is incompatible.
pynwb 1.5.1 requires numpy<1.21,>=1.16, but you'll have numpy 1.21.0 which is incompatible.
networkx 2.5.1 requires decorator<5,>=4.3, but you'll have decorator 5.0.9 which is incompatible.
tensorflow 2.4.2 requires h5py~=2.10.0, but you'll have h5py 3.3.0 which is incompatible.
tensorflow 2.4.2 requires numpy~=1.19.2, but you'll have numpy 1.21.0 which is incompatible.
tensorflow 2.4.2 requires six~=1.15.0, but you'll have six 1.16.0 which is incompatible.
tensorflow 2.4.2 requires typing-extensions~=3.7.4, but you'll have typing-extensions 3.10.0.0 which is incompatible.
```

Installing `tensorflow~=2.4.0` again after the previous step seems to downgrade numpy and the other things to the right version

The latest version of `holoview` is also giving issues with `panel`, using `holoviews~=1.12.0` allows `caiman` to import but just produces a massive warning

# Mesmerize-napari

Brings Mesmerize-like batch manager functionality to the `napari` viewer. 

**Currently a work-in-progress, you are welcome to install and play around with it but the codebase is constantly
evolving on a daily basis so I would not use it for real workflows (unless you want to contribute to the development!)**



## Functionality

This provides a batch manager for performing and organizing CaImAn algorithms for calcium imaging analysis. A robust 
backend built with pandas extensions can be used directly, or through the GUI with napari.

The pandas extensions interface with CaImAn algorithms using `QProcess`

## Streaming demo
**Very redimentary at the moment**
Napari can stream calcium imaging movies in realtime while pyqtgraph is used to show calcium traces. Please note that due to limitations of GIF images, this GIF doesn't fully represent the high level of responsiveness and interactivity during live streaming.
![streaming](./screenshots/napari_streaming.gif)

# Installation
You must have git, build tools etc. installed

1. Create a new python env, you must use python3.8. 3.9+ create weird issues with h5py

2. Install caiman in this environment using kushalkolar's branch

```commandline
git clone https://github.com/kushalkolar/CaImAn.git
cd CaImAn
git checkout mcorr-basename-prefix
pip install -r requirements.txt
pip install -e .
```

3. Install `napari` and `napari-plot`

```commandline
pip install "napari[all]" && pip install "napari-plot[all]"
```

4. Clone & install `mesmerize-napari`

```commandline
https://github.com/nel-lab/mesmerize-napari.git
cd mesmerize-napari
pip install -e .
```

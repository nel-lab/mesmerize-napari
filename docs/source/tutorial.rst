Tutorial
****************

Setup
=============

If you ran the following command when installing CaImAn:

``python caimanmanager.py install --inplace``

then there should be a directory called **example_movies** in the CaImAn folder. In this folder,
there should be a file called **demoMovie.tif**. For this tutorial, we will use this movie for analysis,
though you can use an imaging movie you choose.

Activate the environment you pip installed CaImAn, Napari, Napari-1d, mesmerize-core, and mesmerize-napari
within by running the following command:

``source activate <name of environment>``

Then, in the terminal run the following:

``napari``

The napari software should launch, opening the following window:

.. figure:: ./images/tutorial/napari_base.png

Click on the **plugin** tab, and select **Mesmerize: Main Offline GUI**

.. figure:: ./images/tutorial/select_plugin.png

You should see the following when you click the mesmerize plugin

.. figure:: ./images/tutorial/main_gui_intro.png

The first thing you should do is define your parent data path. If your full data path to your input movie
is */home/arjun/CaImAn/example_movies/demoMovie.tif*, and the sub-directory */CaImAn/* is consistent across
all systems you use, then you can set the parent data path to */home/arjun/*, making the relative
data path for the movie just */CaImAn/example_movies/demoMovie.tif*. Doing so makes it easier to transfer and
access input movies and output files from CaImAn between systems. To set a parent data path, do the following:

.. figure:: ./images/tutorial/set_parent_data_path.png

**NOTE:** The text box will turn green when the parent data path is valid, and red when it is invalid.




Motion Correction (MCORR)
==============


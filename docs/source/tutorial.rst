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

.. figure:: ./images/napari_base.png


Motion Correction (MCORR)
==============


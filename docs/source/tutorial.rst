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

The first thing you should do is define your parent data path. The reason we necessitate defining a parent
data path is for the following scenario:

If you are analysing the same movies on multiple devices - let's call them Computer A and Computer B - then you
will likely be using the same file organization system to manage the movies you wish to analyse. Let's say your movies
are recorded in multiple sessions - *session_1*, *session_2* - and are stored in the directory
**example_movies**. In this case, all movies in session_1 are located in *./example_movies/session_1/*. In both
Computer A and Computer B, this file system **/example_movies/** is the same, but the path leading to the
directory **/example_movies/** will differ. To reference movies within Session 1 in Computer A,
the full path may be */home/computer_A/example_movies/session_1/*, while in Computer B it may be
*/home/computer_B/example_movies/session_1/*. To ensure the same file system **/example_movies/** can be reliably
used between both computer, we will set the parent data path of Computer A to **/home/computer_A** and the
parent data path of Computer B to **/home/computer_B/**. As a result, the relative path of each movie to the
**/example_movies/** directory is preserved without being affected by differences in the parent directory.

To set the parent data path, do the following:

.. figure:: ./images/tutorial/set_parent_data_path.png

**NOTE:** The text box will turn green when the parent data path is valid, and red when it is invalid.

Next, open a new batch by clicking the following button:

FIGURE

A window will launch allowing you to choose a directory to store the batch file (.pikle) within.
Make sure to choose a directory that comes *after* the parent directory.

FIGURE

Once you've created a new batch, select an input movie to use. In this case, we will use the
demoMovie.tif from the example_movies folder. To select a movie, click the **Input Movie** button:

FIGURE

Once you've selected a movie, you should also see the combo box below *Recent Input Movies* has
updated to contain the path to the input movie file. This will allow you to quickly set the
input movie path for each process (MCORR, CNMF(E)) you want to run. For now, the most recently
opened movie will be the input movie path used for a process you create, so no change is necessary.

Motion Correction (MCORR)
==============


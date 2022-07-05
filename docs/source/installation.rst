Installation Guide
************************

Background Information
========================
The following section provides some background information on conda environments, git commands,
and a package installation manager for python: PIP. If you've set these up already, you can skip directly to
the **Installation** section


Conda Environments
------------------
Conda environments are virtual environments used to separate projects and their respective
package dependencies. These virtual environments contain distinct python interpreters, allowing packages
installed within a given environment to be independent of the system interpreter.
To install conda, go to the following website, and install the version
needed for your operating system (OS).

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages

General Commands
~~~~~~~~~~~~~~~~
To create conda environments, run the following

``conda env create -n <name of environment> python=<desired python version>``

For example, if I wanted to create an environment named ``mesmerize``, with python version 3.9,
you can run:

``conda env create -n mesmerize python=3.9``

To see a list of existing conda environments, run the following:

``conda env list``

To activate a specific environment, run the following:

``source activate <name of environment>``

Git Commands
----------------
Git commands allow you to access, clone, modify, and update github repositories. To install
Git on your computer, go to the following website, and install the version needed for your
operating system (OS).

https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

**Note for Windows Users**

Windows powershell gives me many issues when trying to install git within the powershell terminal,
so I would recommend installing Git Bash Terminal instead, and running git commands from with Git Bash.

To download, go to the following website and follow the directions

https://www.educative.io/answers/how-to-install-git-bash-in-windows

Package Installation Manager (PIP)
------------------------------------
PIP is a package manager for python, meaning it's a tool
that allows you to install and manage libraries and dependencies.
You can use pip to install packages within a specific virtual environment
(i.e. conda environments), thus creating an isolated python interpreter
with only the libraries you need for your project.
To install pip, go to the following weibsite, and install the version needed for your operating system (OS).

https://pip.pypa.io/en/stable/installation/

General Commands
~~~~~~~~~~~~~~
To see a list of existing packages and their versions, run the following:

``pip list``

To install a the latest stable version of a specific package, run the following:

``pip install <package name>``

For example, if I need to install numpy, I can run the following:

``pip install numpy``

To install a specific version of a package run the following:

``pip install <package name>==<version #>``

For example, if I need to install version 1.22 of numpy, I can run the following:

``pip install numpy==1.22``

To install packages within a specific conda environment, just active the conda environment and then run
your pip commands

Installation
=========================

Linux OS
----------
Create a virtual environment specific to this project:

.. code-block:: bash

    conda create -n mesmerize python=3.9
    source activate mesmerize

Then navigate the directory in which you wish to install CaImAn

CaImAn
~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/flatironinstitute/CaImAn
    cd CaImAn
    pip install -r requirements.txt
    pip install -e .
    python caimanmanager.py install --inplace

Next, navigate into the directory in which you wish to install Napari

Napari
~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/napari.napari.git
    cd napari
    pip install -e ".[all]"

Next, navigate into the directory in which you wish to install napari-1d, a plotting library for Napari

Napari-1d
~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/lukasz-migas/napari-1d.git
    cd napari-1d
    pip install -e ".[all]"

Next, navigate into the directory in which you wish to install mesmerize-core and mesmerize-napari

mesmerize-core
~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/nel-lab/mesmerize-core
    pip install -e mesmerize-core/

mesmerize-napari
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/nel-lab/mesmerize-napari
    pip install -e mesmerize-napari/

Finally, we will install specific versions of some miscellaneous packages

Miscellaneous Packages
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install qtawesome
    conda install -c conda-forge h5py=2.10.0

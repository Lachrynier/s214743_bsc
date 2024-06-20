### Computational Methods for Correcting Metal-Induced Reconstruction Artifacts in X-ray CT of Historic Gold Treasure

Author: David Diamond Wang Johansen (s214743)

This GitHub repository contains code for my BSc thesis written in the fall semester of 2024. Not all notebooks and intermediary investigations that were part of the process are included.

The Vindelev dataset is currently on its way to being set up as a DOI. Contact Carsten Gundlach at cagu@fysik.dtu.dk or Hans Martin Kjer at hmkj@dtu.dk for inquiries related to the dataset.

The Python virtual environment used is conda. In the requirements.yml file, all the required channels, packages and dependencies are listed.

core.py:

Contains the major functions and classes used for plotting.py

plotting.py: 

Contains the code to generate the majority of the plots (primarily 2D reconstructions) for the report.

It is assumed that all the centre slice data with its corrections are saved under centres/ as AcquisitionData objects in pickle data format.

notebooks/:

Contains Jupyter Notebooks used for some of the processing and for the majority of the 3D plotting.
# Compressed shadow maps demo

This project contains a demo for the compressed shadow maps project based on merged multiresolution hierarchies. This structure allows the creation and usage of very large shadow maps (up to and upwards of 1 million x 1 million resolution) while using only a few MB of video memory. Compressed shadow maps are a very efficient method for creating shadows for large scale static geometry.

The details of this project can be found in the following publications: 

**[1] Scandolo, L. , Bauszat, P. and Eisemann, E. (2016), Compressed Multiresolution Hierarchies for High‐Quality Precomputed Shadows. Computer Graphics Forum, 35: 331-340. doi:10.1111/cgf.12835**

**[2] Scandolo, L. , Bauszat, P. and Eisemann, E. (2016), Merged Multiresolution Hierarchies for Shadow Map Compression. Computer Graphics Forum, 35: 383-390. doi:10.1111/cgf.13035**

Further information can be found at https://graphics.tudelft.nl/Publications-new/2016/SBE16/ and https://graphics.tudelft.nl/Publications-new/2016/SBE16a/

*The project is prepared to be run in Visual Studio 2017 or 2019 through premake. Using the premake-vs201X.bat batch file will generate the solution in the build directory. CUDA is necessary for the creation and (optionally) the evaluation of the compressed shadow maps.*

![Compressed shadow maps](https://graphics.tudelft.nl/Publications-new/2016/SBE16a/teaser.png "Compressed shadow maps")

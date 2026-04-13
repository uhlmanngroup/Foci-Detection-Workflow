# Foci Detection Workflow

## Image Processing Pipeline - Step Summary
### 1. Mask Creation (Cellpose_DAPI_Mask_Creation.ipynb)
Flowchart:

In this first step Cellpose-SAM creates masks for all the nuclei in a picture. For this the images from the DAPI channel are used. All the detected nuclei in the masks are numbered individually starting from 1 (0 would be the background).
The program saves the resulting masks with the ending *_seg.npy into the same folder where the pictures are.

Example mask created by Cellpose-SAM:

Left: DAPI image, where all the detected nuclei are numbered (red).
Right: Mask created by Cellpose-SAM from the DAPI picture on the left. 

### 2. Mask Cleanup (Cellpose_DAPI_Mask_Cleanup.ipynb)
Flowchart:

This program consists of two steps: 
1.	All the detected nuclei are checked to see if they touch the border of the picture and are then removed from the masks.
Then the program shows a depiction of the remaining sizes of detected nuclei for all the pictures.
2.	The user can then set a size threshold, the program then removes all the nuclei that are smaller than that threshold. This is useful to remove artefacts or other irregularities.
At the end all the detected nuclei in the masks are renumbered so that there are no gaps in the numbering from the removed nuclei.

In the end two masks are saved to the same folder that the masks originated from:
*_seg.npy: The final masks after the end of step two. This overwrites the original masks from the mask creation program.
*_seg_noborder.npy: The masks that were created after step one. With this file it is possible to re-do step two of this program and set a different threshold. 

Example picture output:

### 3. Foci detection (Foci_Detection_main.ipynb)
#### 3.1 Parameter Space Generation (one-time setup per data set and channel)
Flowchart:

To get results that are tailored to each data set, three key parameters go through an iteration process, where with the help of ground truth inputs from the user, a parameter space is created. These three parameters are:
-	Foci brightness threshold – The minimal brightness that a focus has to have to be detected (as a percentile of the brightest pixel in the picture)
-	Background brightness – The background brightness that the potential foci candidates are compared to (as a percentile of the brightest pixel in the picture)
-	Contrast threshold – The multiplier by which the potential foci candidates have to be brighter than the background
This process takes place over 6 steps (All steps are identical for FITC and TRITC):
1.	The user selects one to ten nuclei from one picture and adds a ground truth foci count/range. 
2.	The program then iterates through the parameter values and marks all the combinations as valid that result in the foci ranges that correspond to the ground truth for each nucleus.
Foci brightness threshold – Range 0-100th percentile, iteration in 31 steps
Background brightness – Range 0-100th percentile, iteration in 31 steps
Contrast threshold – Range from 1-10x multiplier, iteration in 15 steps
3.	Then all the valid parameter combinations are reduced by only keeping those that were valid for all the selected nuclei.
4.	From these valid parameter combinations a 3d KDE (Kernel Density Estimation) Isosurface is created so that it contains 85% of the point density. This is done to exclude any outliers.
5.	With the Isosurface as base a Delaunay triangulation is done to create a 3d body where all the points are connected to each other (also in the interior). With this body the program can determine if a point is inside or outside of it. This Delaunay triangulation is then saved together with its bounding box.
6.	For visualization a convex hull is generated from the Isosurface and the Delaunay triangulation.


Picture output example:
Step 1, nuclei overview with IDs:




## Dependencies and Acknowledgments

This project relies on the following open-source libraries:

### Core Scientific Computing
- **NumPy** (BSD-3-Clause) - Harris, C.R., et al. (2020). Nature 585: 357–362.
- **SciPy** (BSD-3-Clause) - Virtanen, P., et al. (2020). Nature Methods 17: 261–272.
- **pandas** (BSD-3-Clause) - McKinney, W. (2010). Data Structures for Statistical Computing in Python.

### Machine Learning
- **scikit-learn** (BSD-3-Clause) - Pedregosa, F., et al. (2011). Journal of Machine Learning Research 12: 2825–2830.

### Image Processing
- **scikit-image** (BSD-3-Clause) - van der Walt, S., et al. (2014). PeerJ 2:e453.
- **imageio** (BSD-2-Clause) - Klein, A., et al. (2024). imageio: Image reading and writing in Python.
- **OpenCV** (Apache-2.0) - Bradski, G. (2000). Dr. Dobb's Journal of Software Tools.
- **Pillow (PIL)** (HPND License) - Clark, A. (2015). Pillow (PIL Fork) Documentation.

### Deep Learning / Segmentation
- **Cellpose-SAM** (BSD-3-Clause) - Pachitariu, M., et al. (2025). bioRxiv. https://doi.org/10.1101/2025.04.28.651001

### Visualization
- **matplotlib** (PSF-based) - Hunter, J.D. (2007). Computing in Science & Engineering 9(3): 90–95.

### Utilities
- **tqdm** (MPL-2.0/MIT) - da Costa-Luis, C., et al. (2019). tqdm: A Fast, Extensible Progress Bar.
- **PyYAML** (MIT) - Ben-Kiki, O., et al. (2020). PyYAML: YAML parser and emitter for Python.

All libraries are used in accordance with their respective licenses.

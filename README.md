# Foci Detection Workflow

## Image Processing Pipeline - Step Summary
### 1. Mask Creation (Cellpose_DAPI_Mask_Creation.ipynb)
Flowchart:
![/Readme_pictures/1.Mask_creation_Flowchart.png]
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
- Foci brightness threshold – The minimal brightness that a focus has to have to be detected (as a percentile of the brightest pixel in the picture)
- Background brightness – The background brightness that the potential foci candidates are compared to (as a percentile of the brightest pixel in the picture)
- Contrast threshold – The multiplier by which the potential foci candidates have to be brighter than the background

This process takes place over 6 steps (All steps are identical for FITC and TRITC):
1.	The user selects one to ten nuclei from one picture and adds a ground truth foci count/range. 
2.	The program then iterates through the parameter values and marks all the combinations as valid that result in the foci ranges that correspond to the ground truth for each nucleus.
     - Foci brightness threshold – Range 0-100th percentile, iteration in 31 steps
     - Background brightness – Range 0-100th percentile, iteration in 31 steps
     - Contrast threshold – Range from 1-10x multiplier, iteration in 15 steps
3.	Then all the valid parameter combinations are reduced by only keeping those that were valid for all the selected nuclei.
4.	From these valid parameter combinations a 3d KDE (Kernel Density Estimation) Isosurface is created so that it contains 85% of the point density. This is done to exclude any outliers.
5.	With the Isosurface as base a Delaunay triangulation is done to create a 3d body where all the points are connected to each other (also in the interior). With this body the program can determine if a point is inside or outside of it. This Delaunay triangulation is then saved together with its bounding box.
6.	For visualization a convex hull is generated from the Isosurface and the Delaunay triangulation.


Picture output example:

Step 1, nuclei overview with IDs:

Step 1, selected nucleus:

Step 2, parameter iterations that are valid for one nucleus each:

Steps 4-6, visualization of the KDE Isosurface and the Delaunay triangulation:

#### 3.2 Watershed Threshold Configuration
Flowchart:

In this next step the threshold for the watershed (decides area of detected nuclei) can be set via one of two options:
-	Threshold values in the config.yaml
-	Input by the user after pixel brightness analysis
Pixel brightness analysis: 
Here all the pixels from all the nuclei (no background) in X pictures (amount can be set in the config.yaml) are rescaled to a scale from 0-100th percentile of the brightest pixel. This rescaling can be set to either local or global in the config.yaml.
     - Local rescaling\
 All the nuclei will be individually rescaled, meaning that the brightest pixel in each nucleus will be set to the 100th percentile and the rest will be scaled in relation to that. In the end all the differently rescaled values from all the nuclei will be merged together.
          - Advantages:\
  Dim nuclei will have better foci area detection.
          - Disadvantages:\
  Foci areas aren’t comparable between different nuclei. The same focus will have a bigger detected area when being in a dim nucleus compared to being in a brighter one.
     - Global rescaling\
 Here all the nuclei will be rescaled together on the same scale. The brightest pixel in the whole picture is set to the 100th percentile and the rest will be scaled in relation to that. 
          - Advantages:\
  Foci areas will be comparable between different nuclei. The same focus will have the same detected area no matter how bright the nucleus around it is.
          - Disadvantages:\
  Dim nuclei with dim foci will have a less sensitive area detection.

After the rescaling of the pixel brightness the user can set a threshold via input for FITC and TRITC.

Example output from the pixel brightness analysis:\
Note: Y-axis is logarithmic to preserve the detail in the higher brightness percentiles.

#### 3.3 Load Parameter Spaces
Flowchart:

To get parameters that are adjusted to each data set, the Delaunay triangulations (FITC + TRITC) and their bounding boxes, that were generated earlier, are loaded. 

The next few steps are used to generate 256 parameter combinations that are inside the Delaunay body.
1.	To get parameter combinations, the bounding box is filled with sobol samples (random points, but even spread). The number of sobol samples can be adjusted in the config.yaml file (2^16 recommended). 
2.	Each point is checked to see if it is inside the Delaunay body or outside and only those that are inside are kept. 
3.	From the remaining points one is chosen at random as a starting point. From there 256 points are chosen so that they are maximally far apart from each other.

These resulting 256 parameter combinations can be used to analyze a data set or they can be further reduced to improve computation time. All the random operations are reproducible, as they are connected to the random seed that can be set in the config.yaml.

Pictures to help visualize the process (not generated by the program):

Steps 1-2: Sobol sample generation (green = inside Delaunay body, red = outside Delaunay body):

Step 3, farthest point sampling (red = 256 points maximally far apart, light gray = unused sobol samples):

#### 3.4 Adaptive Calibration (optional)
After generating these 256 parameter combinations it is possible to reduce their amount by analyzing a few images with all of the parameter combinations and then picking the best performing ones:
1.	The amount of images to analyze and the desired number of parameter combinations can be set in the config.yaml.
2.	After analyzing the pictures the following metrics are saved:
     - The average number of foci detected across all parameter combinations (mean_foci)
     - The deviation from the mean for each parameter combination
     - The coefficient of variation (cv)
3.	From these metrics every parameter combination gets a score = 2*deviation + 1*cv\
A lower score is better, meaning that the parameter combinations that are close to the average detection result are best and having less variation is also better.\
Example of two parameter combinations and their metrics after analyzing 4 nuclei with all having a global average of 8 foci detected. In this case parameter combination A is more reliable than B:

With this score the first parameter combination is chosen.
4.	If the parameter combinations are reduced to more than one, the formula for the score changes a bit to increase the diversity of the parameter combinations.

With this the physical distance of the parameter combinations in the 3d grid is also factored in and because it is subtracted from the score points farther away are favored.

The diversity weight is a hardcoded value that automatically adapts to the reduced number of parameter combinations and goes from 0.3 for 2 parameter combinations up to 0.6 for 4 or more parameter combinations.

After selecting the reduced number of parameter combinations, they are then used for the rest of the images that are analyzed. They are also saved and can be loaded if the data analysis is resumed after a break.

#### 3.5 Main Processing Loop (per image)
Flowchart:

The foci detection is split into two parts: The detection of the center of the foci and the area detection (watershed segmentation). For the foci detection all the nuclei are isolated and analyzed individually.

Foci detection:
-	First the texture of all the nuclei is analyzed. This is done by computing the coefficient of variation (CV) to describe how uneven the brightness is across a nucleus. This is done to better differentiate between uniform bright nuclei (few foci expected) and spotty bright nuclei (many foci expected). 
This CV is only saved as a value right now, but there is the option in the code to apply a stricter contrast multiplier for uniformly bright nuclei (Hardcoded as an additional multiplier of 1.0).
-	Then the difference of Gaussian filter is applied to each nucleus. This reduces the background noise. The next steps are done on both the filtered and the unfiltered pictures.

Parameter combination iteration takes place here. The following steps are done with all the previously selected parameter combinations (Parameters: Foci brightness threshold, Background brightness, contrast threshold).
-	Local maxima are detected with a minimal distance of 2 pixels between them and they have to be brighter than the Foci brightness threshold (one of the iterated parameters).
-	For each of these local maxima a background area is chosen to compare it to. This area has the form of an annulus (Donut) with the local maximum in the hole in the middle. If the maximum is located in the middle of a nucleus the inner radius of the annulus is 2 pixels and the outer one is 6 pixels. If the maximum is located close to the border of a nucleus, the outer radius of the annulus grows to 12 pixels to ensure that the chosen background area is big enough (the parts of the annulus that would go outside the nucleus are cut off).
-	From this annulus a brightness value that corresponds to the background brightness (one of the iterated parameters) is chosen.
-	To get a pass in this step each detected local maximum has to be brighter than the background brightness value that was chosen from their corresponding annulus by a factor that is equal or higher than the contrast threshold (one of the iterated parameters).
-	The location of the local maxima that passed the last step are now compared on the unfiltered picture and the one with a difference of Gaussian filter applied. To be confirmed as foci, they must exist on both pictures within a distance of 2 pixels.
-	All the foci locations from different parameter combinations are accumulated for the next step, where their area is detected.

Area detection (watershed segmentation):
-	Here the watershed threshold that was set in an earlier step (3.2) is used for the distance transform to turn the picture into a binary black and white picture where all the foci are the centers of white spots on a black background. The size if the white spots depends on the watershed threshold. Only pixels that are brighter than the watershed threshold are part of the white spots.
-	From this binary picture a watershed segmentation is done where all the detected foci are the seeds. This means that each focus expands its area until it either meets another focus, or the end of the white region surrounding it.

From this foci detection process metrics are saved to two .csv files (step 3.7)

#### 3.6 Channel intensities
Flowchart: 

Additionally to the metrics from step 3.5 the channel intensities for all the channels is measured and saved into the same .csv files.

#### 3.7 Data Collection
To save all the data two .csv files are created, one to save all the data that pertains the foci themselves and one for the data pertaining the nuclei. 

Foci data:
-	Cell number, to identify which nucleus this focus belongs to
-	Channel (FITC/TRITC)
-	Well, Position, to identify the picture
-	Focus center coordinates
-	Focus area (pixels)
-	Focus circularity, to measure the shape (0-1)
-	Total focus intensity, sum of brightness
-	Focus mean intensity, average brightness
-	Detection probability, percentage of how many time a focus was detected over all parameter combinations

Nuclei data:
-	Cell number, to identify the nucleus
-	Well, Position, to identify the picture
-	DAPI area
-	DAPI perimeter
-	DAPI circularity
-	Nucleus center coordinates
-	Texture coefficient of variation (FITC and TRITC)
-	If texture dependent filtering was applied (disabled at the moment)
-	Mean foci intensity across all foci in the nucleus (FITC and TRITC)
-	Minimum + maximum foci intensity (FITC and TRITC)
-	Standard deviation across foci intensities (FITC and TRITC)
-	Minimum + maximum of foci found by any parameter combination (FITC and TRITC)
-	Mean number of detected foci (FITC and TRITC)
-	Standard deviation of foci counts across all parameter combinations (FITC and TRITC)
-	Total and mean intensity across the whole nucleus (DAPI, FITC, TRITC, Cy5)

#### 3.8 Visualization (optional)
In the config.yaml the option for saving all or specific pictures can be activated. This leads to the following pictures to be saved as .png (FITC and TRITC):

Foci locations:
-	DAPI picture as background
-	All the detected foci locations marked as red dots (no foci area information)

Example output:

Watershed segmentation:
-	DAPI picture as background
-	Watershed result (foci areas) with individual colors

Example output:






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

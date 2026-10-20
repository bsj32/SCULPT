SCULPT
Skeletonized Canaliculi Utilizing Lacunar Structure-aware Preservation Tracking and quantification 
Overview
SCULPT is a work in progress Python-based image analysis pipeline for segmenting and characterizing canaliculi within the osteocyte lacuno-canalicular network (LCN) of subchondral bone. The pipeline processes three-dimensional image stacks, enhances and segments canalicular structures, applies post-processing steps to improve network continuity, and skeletonizes the resulting network for quantitative analysis. SCULPT also supports lacuna-level measurements and three-dimensional visualization of segmentation results.
Features
1.	Canaliculi segmentation – intensity normalization, adaptive histogram equalization, Frangi filtering, percentile thresholding, and small-object removal.
2.	Segmentation tuning – parameter selection using manually annotated canaliculi masks and gap bridging.
3.	Skeletonization – Conversion of the segmented canalicular network into a skeleton for branch analysis. 
4.	Quantitative analysis – Canaliculi length, canalicular length density and number of canaliculi per lacunae.
5.	Data export – TIFF image stacks, CSV measurements, histograms and summary tables. 
6.	3D visualization – Interactive viewing of segmentation results using napari.
Input data
SCULPT currently uses raw three-dimensional image data stored as TIFF slices, manually annotated canaliculi masks for parameter tuning, and NPZ segmentation files for lacunae, cartilage, and vessels. Input data dimensions, voxel spacing, file naming, and segmentation formats must be consistent. 
Requirements
The current implementation uses Python and the following packages:
NumPy, pandas, tifffle, Matplotlib, scikit-image, SciPy, scikit-image analysis tools provided by skan, and napari. 

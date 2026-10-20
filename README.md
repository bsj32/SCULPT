# SCULPT

**Skeletonized Canaliculi Utilizing Lacunar Structure-aware Preservation Tracking and Quantification**

## Overview

SCULPT is a work-in-progress Python-based image analysis pipeline for segmenting and characterizing canaliculi within the osteocyte lacuno-canalicular network (LCN) of subchondral bone. The pipeline processes three-dimensional image stacks, enhances and segments canalicular structures, applies post-processing steps to improve network continuity, and skeletonizes the resulting network for quantitative analysis. SCULPT also supports lacuna-level measurements and three-dimensional visualization of segmentation results.

## Features

1. **Canaliculi segmentation** – Intensity normalization, adaptive histogram equalization, Frangi filtering, percentile thresholding, and small-object removal.
2. **Segmentation tuning** – Parameter selection using manually annotated canaliculi masks and gap bridging.
3. **Skeletonization** – Conversion of the segmented canalicular network into a skeleton for branch analysis.
4. **Quantitative analysis** – Canalicular length, canalicular length density, and lacuna-associated measurements.
5. **Data export** – TIFF image stacks, CSV measurements, histograms, and summary tables.
6. **3D visualization** – Interactive viewing of segmentation results using napari.

## Input Data

SCULPT currently uses:

- Raw three-dimensional image data stored as TIFF slices.
- Manually annotated canaliculi masks for parameter tuning.
- NPZ segmentation files for lacunae, cartilage, and vessels.

Input data dimensions, voxel spacing, file naming conventions, and segmentation formats must be consistent with the current implementation.

## Requirements

The current implementation uses Python and the following packages:

- NumPy
- pandas
- tifffile
- Matplotlib
- scikit-image
- SciPy
- skan
- napari

## Project Status

SCULPT is a work in progress. The code and analysis methods may change as development continues.

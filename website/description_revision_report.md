# Description Revision Report

## Scope

This report summarizes the description cleanup pass for `public/data/tasks_db.json`, focused on truncated problem descriptions.

## What was fixed

- Repaired truncated/incomplete task descriptions by reloading from `/data/yjh/task_descriptions/*_description.md`.
- Normalized formatting to avoid hard truncation artifacts (e.g., abrupt endings like `...deco`, `...fitti`, `...optimiz`).
- Preserved existing non-truncated descriptions.

## Verification

- Truncation audit after repair: `0` issues (no empty descriptions, no 500-char truncation-pattern endings).
- Build verification passed: `npm run build`.

## Revised Tasks (71)

- 06 Semi-Blind PSF Deconvolution
- 08 Zernike Wavefront Estimation
- 10 Electrical Impedance Tomography
- 11 Beam Propagation ODT
- 12 Differentiable Optical Metrology
- 13 InSAR Phase Unwrapping
- 14 Photoacoustic Tomography
- 15 Lensless ADMM Reconstruction
- 16 Lensless Deep Learning
- 17 CASSI Hyperspectral Imaging
- 18 Fourier Ptychographic Microscopy
- 19 Fourier Light Field Microscopy
- 20 Gravitational Lensing (Ring)
- 21 Gravitational Lensing (Quad)
- 22 Gravitational Lensing (Double)
- 23 Gravitational Lensing (Shapelets)
- 24 Gravitational Lensing (Host)
- 25 DPI Radio Interferometry
- 26 DPI MRI Reconstruction
- 27 DPI CT Reconstruction
- 28 NODDI Microstructure
- 30 Gravitational Lensing (Caustics)
- 31 CT/MR Perfusion Imaging
- 32 Diffuser Camera Reconstruction
- 33 Diffusion MRI Microstructure
- 34 High Contrast Imaging
- 35 MEG/EEG Source Localization
- 36 Magnetic Particle Imaging
- 37 MR Elastography
- 38 MR Fingerprinting
- 39 MRI ADMM-TV Reconstruction
- 40 OCT Structural Imaging
- 41 FLIM Phasor Analysis
- 42 Ptychographic Reconstruction
- 43 Abel Inversion
- 44 Digital Holographic Microscopy
- 45 Holoscope Reconstruction
- 46 Single Molecule LFM
- 47 Spectral CT Decomposition
- 48 STORM Super-Resolution
- 49 Structured Light 3D
- 50 Model-Based CT Reconstruction
- 51 CT FBP/SIRT Reconstruction
- 52 Ultrasound Beamforming
- 53 Diffuse Optical Tomography
- 54 X-ray/Neutron Reflectometry
- 55 Exoplanet Transit Fitting
- 56 NMR Spectrum Reconstruction
- 57 3D Gravity Inversion
- 58 Magnetotelluric Inversion
- 59 EIS/DRT Analysis
- 60 Spectroscopic Ellipsometry
- 61 SAXS Electron Density
- 62 EXAFS Fitting
- 63 Residual Stress (Contour)
- 64 Inverse Heat Conduction
- 65 Virtual Fields Method
- 66 PIV Pressure Reconstruction
- 67 Hyperspectral Unmixing
- 68 Surface Wave Tomography
- 69 Ambient Noise Tomography
- 70 SAR Imaging
- 71 Ground Penetrating Radar
- 72 Compressive Ghost Imaging
- 73 Non-Line-of-Sight Imaging
- 74 Event Camera to Video
- 75 Sparse-View CT
- 76 Coherent Diffraction Imaging
- 77 Ultrasound CT (FWI)
- 79 PIV Cross-Correlation
- 80 Mie Scattering Inversion

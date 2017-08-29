`xveganx` Project roadmap
---


Tasks:

### Spectroscopy metadata [#1](https://github.com/BrownDwarf/xveganx/issues/1)
- [x] Mine the IGRINS fits header tables into a table [01.01](./01.01-IGRINS_FITS_header_table.ipynb)
- [ ] Construct epochs.csv containing acquisition dates of all spectra [01.02]()

### Photometry analysis [#2](https://github.com/BrownDwarf/xveganx/issues/2)
- [x] Compile all photometry into flat tables
- [x] Compile Grankin08 periods into easy-to-use table
- [x] Compute Seasonally aggregated statistics
- [ ] Make a figure of postage stamps for all objects and seasons possessing IGRINS spectra.

### Photometry [#4](https://github.com/BrownDwarf/xveganx/issues/4)
- [x] Retrieve ASASSN photometry [04.01](04.01-Retrieve_ASASSN_data.ipynb)
- [x] Retrieve Grankin08 photometry [04.02](04.02-Retrieve_Grankin_data.ipynb)
- [ ] Exploratory analysis of ASASSN data
- [ ] Retrieve ASAS3 photometry
- [ ] Retrieve AAVSO photometry
- [ ] Retrieve Integral-OMC photometry
- [ ] Research/add any ancillary data sources


### Stellar parameter estimates [#5](https://github.com/BrownDwarf/xveganx/issues/5)
- [ ] Compile a table of previous estimates of stellar parameters

### Spectroscopy [#6](https://github.com/BrownDwarf/xveganx/issues/6)
- [ ] Make sure spectra are telluric corrected
- [ ] Construct and spotcheck the variance arrays
- [ ] Put spectra into the HDF5 format needed

### Generate config.yaml files [#7](https://github.com/BrownDwarf/xveganx/issues/7)
- [ ] Define environment variables
- [ ] Decide whether to do this automated, or semi-automated?

### K2? [#8](https://github.com/BrownDwarf/xveganx/issues/8)
- [x] Check to see if any sources have K2 campaign 13 photometry available
- [ ] K2 photometry analysis, if available

### Run Starfish grid and PCA jobs, and pre-processing [#9](https://github.com/BrownDwarf/xveganx/issues/9)
- [ ] Pilot programs to assess run-time and debug
- [ ] Establish way of tracking metadata of jobs
- [ ] Run the jobs
- [ ] Copy-over files where necessary

### Run customized Starfish star_mix.py jobs [#10](https://github.com/BrownDwarf/xveganx/issues/10)
- [ ] Pilot programs to assess run-time and debug
- [ ] Establish way of tracking metadata of jobs
- [ ] Spot-check outcomes
- [ ] Run the jobs
- [ ] Copy-over files where necessary

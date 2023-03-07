# MRF
This is the repository for MRF related projects of the Setsompop Lab at Stanford University. The code here is used to read and reconstruct MRF data and accompanying calibration scans.

## Usage
Each directory in `src` can be built into a Docker container that performs one task in the pipeline. By navigating into each directory and running `make` the docker containers will be built with all dependencies. The code can also be run natively, but then you need to manage all dependencies as listed in the Dockerfile for each task. For each task certain input arguments have to be set. The `main.py` file for each task lists input arguments and what format they should be in. Example calls are provided below.

1. `00_io` manages reading ScanArchive files form GE and can write Dicoms with metadata from the ScanArchives. It requires GE's base image `cpp-sdk` (available on the GE user forum) for accessing the ScanArchive data. Add `/mnt/` to the file paths so that the files can be found within the mounted filesystem in Docker.
<b>Example call to save raw k-space data from ScanArchive to Numpy Array</b>
```
docker run -v /:/mnt/:z MRF/scan_archive_io \
        --scn /mnt/<b>FULL PATH TO SCAN ARCHIVE</b>\
        --ksp /mnt/<b>FULL PATH TO SAVE LOCATION OF NUMPY ARRAY</b>
```
2. `01_calib` is used to calibrate the MRF reconstruction with data from a pre-scan. It can pre-calculate the coil compression matrix using RoVir from a large FOV acquisition and shift the FOV so that the brain is within the smaller FOV for MRF.
<b>Example call to calculate shifts and coil compressionmatrix from large FOV GRE data</b>
```
docker run --gpus all -v /:/mnt/:z MRF/calib 
        --ksp /mnt/<b>FULL PATH TO GRE NUMPY ARRAY</b>
        --ccm /mnt/<b>FULL PATH TO SAVE LOCATION OF COILS COMPRESSION MATRIX NUMPY ARRAY</b>
        --shf /mnt/<b>FULL PATH TO SAVE LOCATION OF SHIFTS NUMPY ARRAY</b>
        --nrc <b>NUMBER OF COILS TO KEEP AFTER ROVIR COMPRESSION</b>
        --nsv <b>NUMBER OF COILS TO KEEP AFTER SVD COMPRESSION</b>
```
3. `02_recon` is used to reconstruct the MRF data . It has many optional input arguments for pre-calculated data (e.g. density compensation and coil sensitivity maps) for improved pipeline optimization avoiding recalculating the same thing multiple times. Note that GPU access is important for fast reconstruction.
<b>Example call to reconstruct MRF data using a subspace reconstruction</b>
```
docker run --gpus all -v /:/mnt/:z MRF/recon 
        -p
        --trj /mnt/<b>FULL PATH TO TRAJECTORY FILE</b>
        --ksp /mnt/<b>FULL PATH TO KSPACE DATA FILE</b>
        --res /mnt/<b>FULL PATH TO SAVE LOCATION OF RECONSTRUCTED IMAGES NUMPY ARRAY</b>
        --phi /mnt/<b>FULL PATH TO SUBSPACE BASIS FILE</b>
        --ccm /mnt/<b>FULL PATH TO COIL COMPRESSION MATRIX</b>
        --shf /mnt/<b>FULL PATH TO SHIFTS</b> 
        --mal jsense
        --mtx 256 --ptt 10 --dev 0
        --pdg 0 --blk 8 --lam 5e-5 --mit 40 
```

## How to build docker containers

### How to read GE ScanArchive data
1. Build Docker image (you only need to do this once unless you make changes to the io code):
``` 
cd recon/io
make         
```
This names the docker container `ssi/scan_archive_io` by default. You can change the name in the Makefile if you want to.
2. Set up file paths to read and write: 
```
IN=<full path to ScanArchive>
KSP=<full path to where you want to store the k-space data .npy file>
NSE=<full path to where you want to store the noise measurement data .npy file> #OPTIONAL
NSC=<full path to Noise Measurment .h5 file (if different from Scan Archive)>   #OPTIONAL
PARAMS=<full path to where you want to store parameter .pkl file>               #OPTIONAL
TXT=<full path to where you want to store parameter .txt file>                  #OPTIONAL

```
3. Run Docker container:
```
docker run -v /:/mnt/:z ssi/scan_archive_io \
--scn $IN \
--ksp $KSP \
--nse $NSE \
--nsc $NSC \
--pkl $PARAMS \
--txt $TXT
```
This will read the ScanArchive and NoiseStatistic files and save the raw data in numpy arrays. It will also save some key parameters into the .pkl and/or .txt file.

### How to reconstruct and get parameters from a GRE calibration scan
1. Build Docker image (you only need to do this once unless you make changes to the io code) for the calibration scan. 
``` 
cd recon/calib
make         
```

By default this image is named `ssi/calib`. You can change the name in the Makefile if you want to.

2. Set up file paths to read and write: 
```
KSP=<full path to k-space data saved as .npy>
NSE=<full path to where noise measurement data saved as .npy file> 
CCM=<full path to where you want to save the coil compression matrix as .npy>
SHF=<full path to where you want to save the autoFOV shift parameters as .npy>                    
```

You can run `docker run ssi/calib` to see what additional parameter flags you can tune.

3. Run Docker container, for example:
```
  docker run --gpus all -v /:/mnt/:z ssi/calib \
          --ksp $KSP \
          --nse $NSE \
          --ccm $CCM \
          --shf $SHF \
          --thr 0.01 \          # THRESHOLD FOR AUTOFOV 
          --ofs 0.5 1.5 0.5 \   # OFFSET BETWEEN CALIBRATION IMAGE AND MRF FOR AUTOFOV
          --nrc 40 \            # NUMBER OF ROVIR COILS TO KEEP
          --nsv 2               # NUMBER OF SVD COILS TO COMPRESS TO
```

# Example data
TODO!

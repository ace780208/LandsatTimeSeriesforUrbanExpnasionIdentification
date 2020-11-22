# Algorithm of identifying urbanization based on dense time series of Landsat imagery

## Five .py files for deriving newly urbanzied pixels from dense time series of Landsat imagery
1. SMA.py
2. timeSeries.py
3. ImgTimeSeries.py
4. accuAssess.py
5. main.py

## SMA.py
The script is mainly for deriving Vegetation-Impervious-Soil (V-I-S) fractional map for each date of Landsat imagery.

## timeSeries.py
The script is for plotting V-I-S time series from the time series V-I-S fractional maps.

## ImgTimeSeries.py
The script is for exptracting V-I-S time series and identifying the impervious change (urban expansion and urban renewal) from the extracted V-I-S time series.

## accuAssess.py
The script is for conducting accuracy assessment for individual V-I-S map and the identified newly urbanized pixels.

## main.py
The script is for high performance computing for deriving 300 dates of V-I-S maps with multi-processing cores.

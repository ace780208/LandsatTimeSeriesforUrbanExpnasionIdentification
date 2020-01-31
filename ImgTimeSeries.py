import numpy as np
import pandas as pd
import os
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
import glob
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)


def extractOrgPxTimeSeries(Unmixfolder, xoffset, ImgHgt):
    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    ISeries = []
    SSeries = []

    for index, img in enumerate(VIS_list):
        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(xoffset, 0, 1, ImgHgt)).astype(float)
        data2 = (band2.ReadAsArray(xoffset, 0, 1, ImgHgt)).astype(float)
        data3 = (band3.ReadAsArray(xoffset, 0, 1, ImgHgt)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data2)
        data3 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data3)

        data2 = np.where((data2 == -1), np.nan, data2)
        data3 = np.where((data3 == -1), np.nan, data3)

        ISeries.append((data2.reshape(ImgHgt, )).tolist())
        SSeries.append((data3.reshape(ImgHgt, )).tolist())
        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return np.array(ISeries).T, np.array(SSeries).T


def extractOrgPxTimeSeries2(Unmixfolder, xoffset, yoffset):
    import glob
    import datetime

    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    ISeries = []
    SSeries = []

    for index, img in enumerate(VIS_list):
        tmpdate = datetime.datetime(int(VIS_list[index][:4]), int(VIS_list[index][4:6]), int(VIS_list[index][6:8]))

        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(xoffset, yoffset, 1, 1)).astype(float)
        data2 = (band2.ReadAsArray(xoffset, yoffset, 1, 1)).astype(float)
        data3 = (band3.ReadAsArray(xoffset, yoffset, 1, 1)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data2)
        data3 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data3)

        data2 = np.where((data2 == -1), np.nan, data2)
        data3 = np.where((data3 == -1), np.nan, data3)
        ISeries.append(data2[0, 0])
        SSeries.append(data3[0, 0])

        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return ISeries, SSeries


def extractOrgPxTimeSeries3(Unmixfolder, yoffset, ImgWid):
    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    ISeries = []
    SSeries = []

    for index, img in enumerate(VIS_list):
        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)
        data2 = (band2.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)
        data3 = (band3.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data2)
        data3 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data3)

        data2 = np.where((data2 == -1), np.nan, data2)
        data3 = np.where((data3 == -1), np.nan, data3)

        ISeries.append((data2[0]).tolist())
        SSeries.append((data3[0]).tolist())
        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return np.array(ISeries).T, np.array(SSeries).T


def extractOrgPxTimeSeries4(Unmixfolder, yoffset, ImgWid):
    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    VSeries = []
    ISeries = []
    SSeries = []

    for index, img in enumerate(VIS_list):
        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)
        data2 = (band2.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)
        data3 = (band3.ReadAsArray(0, yoffset, ImgWid, 1)).astype(float)

        data1 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data1)
        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data2)
        data3 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), np.nan, data3)

        data1 = np.where(((data1 == -1) | (data2 == -1) | (data3 == -1)), np.nan, data1)
        data2 = np.where(((data1 == -1) | (data2 == -1) | (data3 == -1)), np.nan, data2)
        data3 = np.where(((data1 == -1) | (data2 == -1) | (data3 == -1)), np.nan, data3)

        VSeries.append((data1[0]).tolist())
        ISeries.append((data2[0]).tolist())
        SSeries.append((data3[0]).tolist())
        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return np.array(VSeries).T, np.array(ISeries).T, np.array(SSeries).T


def timeSeriesInterpolation(pxLCSeries):
    # this is only for single land cover list
    pxLCSeries = np.array(pxLCSeries, dtype=float)
    pxLCSeries[pxLCSeries < 0] = np.nan
    pxLCSeries = pd.Series(pxLCSeries)
    interSeries = pxLCSeries.interpolate(method="linear")
    return interSeries


def SGFilter(interpSeries, halfwindowlen, polynomial):
    from scipy.signal import savgol_filter
    if np.isnan(interpSeries).any():
        ind = np.where(~np.isnan(interpSeries))[0]
        first, last = ind[0], ind[-1]
        interpSeries[:first] = interpSeries[first]
        interpSeries[last + 1:] = interpSeries[last]
    return savgol_filter(interpSeries, window_length=int(halfwindowlen*2+1), polyorder=polynomial)


def LogisticUrbanSprawl(DeYr, Iseries, D8series):
    I = Iseries[~np.isnan(Iseries)]
    I = np.where(I >= 300, 1, 0)
    uI = np.unique(I)

    if len(uI) != 2:
        Utype = 0
        outRange = -1
        outDate = -1

    else:
        modDeYr = DeYr[~np.isnan(Iseries)]
        modDeYr = modDeYr.reshape((len(modDeYr), 1))
        Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(modDeYr, I)
        Ipred1 = Iclf.predict(D8series)
        Ipred = Iclf.predict_proba(D8series)
        modeledImp = Ipred[:, 1]*100
        if (np.min(modeledImp) > 50) or (np.max(modeledImp) < 50):
            Utype = 0
            outRange = -1
            outDate = -1

        else:
            outDf = np.argwhere(modeledImp>50)
            if len(outDf)==0:
                Utype = 0
                outRange = -1
                outDate = -1
            else:
                Utype = 1
                outDate = D8series[outDf[0, 0]]
                beforeIser = Iseries[DeYr<outDate]
                afterIser = Iseries[DeYr>outDate]
                beforeAvgImp = np.nanmean(beforeIser)/10
                afterAvgImp = np.nanmean(afterIser)/10
                outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def LogisticUrbanSprawl2(D81D, D8series, FilterIseries):
    I = np.where(FilterIseries >= 500, 1, 0)
    uI = np.unique(I)

    if len(uI) != 2:
        Utype = 0
        outRange = -1
        outDate = -1

    else:

        scaleD8 = preprocessing.scale(D8series)
        Iclf = LogisticRegression().fit(D8series, I)
        Ipred1 = Iclf.predict(scaleD8)
        Ipred = Iclf.predict_proba(scaleD8)
        modeledImp = Ipred[:, 1]*100
        if (np.min(modeledImp) > 50) or (np.max(modeledImp) < 50):
            Utype = 0
            outRange = -1
            outDate = -1

        else:
            outDf = np.argwhere(modeledImp>50)
            if len(outDf)==0:
                Utype = 0
                outRange = -1
                outDate = -1
            else:
                Utype = 1
                outDate = D8series[outDf[0, 0]]
                beforeIser = FilterIseries[D81D<outDate]
                afterIser = FilterIseries[D81D>outDate]
                beforeAvgImp = np.nanmean(beforeIser)/10
                afterAvgImp = np.nanmean(afterIser)/10
                outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def LogisticUrbanSprawl3(ODYr, d8deyr, Vseries, Iseries, dday_1D, dday_2D, d8dday_1D, d8dday_2D):
    Idata = Iseries[(~np.isnan(Iseries)) & (~np.isnan(Vseries))]
    filterODYr = ODYr[(~np.isnan(Iseries)) & (~np.isnan(Vseries))]
    Vdata = Vseries[(~np.isnan(Iseries)) & (~np.isnan(Vseries))]
    VIndex = np.argwhere(Vdata > Idata)

    if VIndex.shape[0] > 0:
        Idata[:VIndex[-1][0]] = 0

    I = np.where(Idata >= 500, 1, 0)
    uI = np.unique(I)

    if len(uI) != 2:
        Utype = 0
        outRange = -1
        outDate = -1

    else:
        modDeYr = dday_2D[(~np.isnan(Iseries)) & (~np.isnan(Vseries))]
        modDeYr = modDeYr.reshape((len(modDeYr), 1))
        Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(modDeYr, I)
        Ipred = Iclf.predict_proba(d8dday_2D)
        modeledImp = Ipred[:, 1]*100
        if (np.min(modeledImp) > 50) or (np.max(modeledImp) < 50):
            Utype = 0
            outRange = -1
            outDate = -1

        else:
            outDf = np.argwhere(modeledImp>50)
            if len(outDf)==0:
                Utype = 0
                outRange = -1
                outDate = -1
            else:
                tmpDate = d8deyr[outDf[0, 0]]

                if (tmpDate < 1990) or (tmpDate > 2016):
                    Utype = 0
                    outDate = -1
                    outRange = -1

                else:
                    Utype = 1
                    outdday = d8dday_1D[modeledImp > 50]
                    beforeIser = Iseries[dday_1D < outdday[0]]
                    afterIser = Iseries[dday_1D > outdday[0]]
                    outDate = tmpDate
                    beforeAvgImp = np.nanmean(beforeIser)/10
                    afterAvgImp = np.nanmean(afterIser)/10
                    outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def UrbanRenewal(D8series, I8series, S8series):
    styPd = D8series[(D8series>1990) & (D8series<2016)]
    # get the max and the min for S and I
    minI = np.min(I8series[(D8series>1990) & (D8series<2016)])
    maxS = np.max(S8series[(D8series>1990) & (D8series<2016)])
    # get the dates corresponds to the max(S) and min(I)
    minID = (D8series[(D8series>1990) & (D8series<2016)])[I8series[(D8series>1990) & (D8series<2016)]==minI]
    maxSD = (D8series[(D8series>1990) & (D8series<2016)])[S8series[(D8series>1990) & (D8series<2016)]==maxS]

    if len(minID) > 0:
        minID = minID[0]
    if len(maxSD) > 0:
        maxSD = maxSD[0]

    meanI = np.mean(I8series[(D8series>1990) & (D8series<2016)])
    meanS = np.mean(S8series[(D8series>1990) & (D8series<2016)])

    if (abs(minID - maxSD)<0.5):
        if (minI < maxS) and (meanI > meanS):
            Utype = 2
            outRange = (np.mean(I8series[D8series < minID]) - np.mean(I8series[D8series > minID]))/10
            outDate = minID

        else:
            Utype = 0
            outRange = -1
            outDate = -1

    else:
        Utype = 0
        outRange = -1
        outDate = -1

    return Utype, outRange, outDate


def LogisticUrbanAbandon(DeYr, Iseries, D8series):
    I = Iseries[~np.isnan(Iseries)]
    I = np.where(I >= 300, 1, 0)
    uI = np.unique(I)

    if len(uI) != 2:
        Utype = 0
        outRange = -1
        outDate = -1

    else:
        modDeYr = DeYr[~np.isnan(Iseries)]
        modDeYr = modDeYr.reshape((len(modDeYr), 1))
        Iclf = LogisticRegression().fit(modDeYr, I)
        Ipred = Iclf.predict_proba(D8series)
        modeledImp = Ipred[:, 1]*100

        if (np.max(modeledImp) < 50) or (np.min(modeledImp) > 50):
            Utype = 0
            outRange = -1
            outDate = -1

        else:
            outDf = np.argwhere(modeledImp < 50)
            if len(outDf)==0:
                Utype = 0
                outRange = -1
                outDate = -1
            else:
                Utype = 3
                outDate = D8series[outDf[0, 0]]
                beforeIser = Iseries[DeYr < outDate]
                afterIser = Iseries[DeYr > outDate]
                beforeAvgImp = np.nanmean(beforeIser) / 10
                afterAvgImp = np.nanmean(afterIser) / 10
                outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def LogisticUrbanAbandon2(D81D, D8series, FilterIseries):
    I = np.where(FilterIseries >= 500, 1, 0)
    uI = np.unique(I)

    if len(uI) != 2:
        Utype = 0
        outRange = -1
        outDate = -1

    else:

        scaleD8 = preprocessing.scale(D8series)
        Iclf = LogisticRegression().fit(scaleD8, I)
        Ipred = Iclf.predict_proba(scaleD8)
        modeledImp = Ipred[:, 1]*100

        if (np.max(modeledImp) < 50) or (np.min(modeledImp) > 50):
            Utype = 0
            outRange = -1
            outDate = -1

        else:
            outDf = np.argwhere(modeledImp < 50)
            if len(outDf)==0:
                Utype = 0
                outRange = -1
                outDate = -1
            else:
                Utype = 3
                outDate = D8series[outDf[0, 0]]
                beforeIser = FilterIseries[D81D < outDate]
                afterIser = FilterIseries[D81D > outDate]
                beforeAvgImp = np.nanmean(beforeIser) / 10
                afterAvgImp = np.nanmean(afterIser) / 10
                outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def postFiveDateCompare(Unmixfolder):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['CDate'] = pd.to_datetime(df['Date'])
    df['DeciYr'] = pd.DatetimeIndex(df['CDate']).year + (pd.DatetimeIndex(df['CDate']).dayofyear / 366)
    deciYr = df['DeciYr'].values
    df['accuDate'] = np.where((df['VRMSE'] < 27) &
                              (df['IRMSE'] < 27) &
                              (df['SRMSE'] < 27) &
                              (df['Cnt'] > 30), True, False)

    rng8D = pd.date_range('1987/4/9', periods=1403, freq='8D')
    f10yr = rng8D < pd.to_datetime('1993-1-1')
    l10yr = (rng8D >= pd.to_datetime('2013-1-1'))
    D8array = np.array((rng8D.year + rng8D.dayofyear / 366).tolist())
    primaryVISArr = df[df['accuDate'] & (df['CDate'] < '1990-1-1')]['Date'].values
    VISArr = df[df['accuDate'] & (df['CDate'] > '1990-1-1')]['Date'].values

    maskimg = r'C:\dissertation\twimage\LandsatScene\ProcExtent_admin_1b.tif'
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskdataset = gdal.Open(maskimg, gdalconst.GA_ReadOnly)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotransform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()
    if not maskgeotransform is None:
        origin_x = maskgeotransform[0]
        origin_y = maskgeotransform[3]
        pixel_width = maskgeotransform[1]
        pixel_height = maskgeotransform[5]

    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    primaryDatesImp = np.zeros((maskrows, maskcols))

    for img in primaryVISArr:
        tmpImgPath = img + '.tif'
        tmpdataset = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset.GetRasterBand(1)
        band2 = tmpdataset.GetRasterBand(2)
        band3 = tmpdataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        data2 = np.where((data2>=500), 1, 0)

        primaryDatesImp += data2

        band1 = None
        band2 = None
        band3 = None
        tmpdataset = None
    primaryDatesImp = np.where(primaryDatesImp > 1, 1, 0)

    for index, tmpI in enumerate(VISArr):
        if (len(VISArr) - index) < 5:
            break
        tmpImgPath = tmpI + '.tif'
        tmpdataset = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset.GetRasterBand(1)
        band2 = tmpdataset.GetRasterBand(2)
        band3 = tmpdataset.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr = np.where((data2 >= 500), 1, 0)

        tmpdataset1 = gdal.Open(VISArr[index+1]+'.tif', gdalconst.GA_ReadOnly)
        band1 = tmpdataset1.GetRasterBand(1)
        band2 = tmpdataset1.GetRasterBand(2)
        band3 = tmpdataset1.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr1 = np.where((data2 >= 500), 1, 0)

        tmpdataset2 = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset2.GetRasterBand(1)
        band2 = tmpdataset2.GetRasterBand(2)
        band3 = tmpdataset2.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr2 = np.where((data2 >= 500), 1, 0)

        tmpdataset3 = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset3.GetRasterBand(1)
        band2 = tmpdataset3.GetRasterBand(2)
        band3 = tmpdataset3.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr3 = np.where((data2 >= 500), 1, 0)

        tmpdataset4 = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset4.GetRasterBand(1)
        band2 = tmpdataset4.GetRasterBand(2)
        band3 = tmpdataset4.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr4 = np.where((data2 >= 500), 1, 0)

        tmpdataset5 = gdal.Open(tmpImgPath, gdalconst.GA_ReadOnly)
        band1 = tmpdataset5.GetRasterBand(1)
        band2 = tmpdataset5.GetRasterBand(2)
        band3 = tmpdataset5.GetRasterBand(3)

        data1 = (band1.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data2 = (band2.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)
        data3 = (band3.ReadAsArray(0, 0, maskcols, maskrows)).astype(float)

        data2 = np.where(((data1 == 0) & (data2 == 0) & (data3 == 0)), -1, data2)
        tmpArr5 = np.where((data2 >= 500), 1, 0)

        outArr1 = np.where((tmpArr == 0) & (tmpArr1 == 1), 1, 0)
        outArr2 = np.where((tmpArr == 0) & (tmpArr2 == 1), 1, 0)
        outArr3 = np.where((tmpArr == 0) & (tmpArr3 == 1), 1, 0)
        outArr4 = np.where((tmpArr == 0) & (tmpArr4 == 1), 1, 0)
        outArr5 = np.where((tmpArr == 0) & (tmpArr5 == 1), 1, 0)

    pass


def UrbanChgTypeAndTime(RMSE, window, polyDegree, startcol, endcol):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['CDate'] = pd.to_datetime(df['Date'])
    df['ODate'] = pd.to_datetime('1987-04-09')
    df['ddays'] = (df['CDate'] - df['ODate']).dt.days
    df['accuDate'] = np.where((df['VRMSE'] < RMSE) &
                              (df['IRMSE'] < RMSE) &
                              (df['SRMSE'] < RMSE) &
                              (df['Cnt'] > 30), True, False)

    rng8D = pd.date_range('1987/4/9', periods=1403, freq='8D')
    first10yr = rng8D < pd.to_datetime('1998-1-1')
    last10yr = (rng8D >= pd.to_datetime('2008-1-1'))
    LogDateDf = pd.DataFrame({'date': rng8D})
    LogDateDf['ODate'] = pd.to_datetime('1987-04-09')
    LogDateDf['ddays'] = (LogDateDf['date'] - LogDateDf['ODate']).dt.days
    styPdD8TimeSeries = LogDateDf.loc[
        (LogDateDf['date'] > pd.to_datetime('1989-12-31')) & (LogDateDf['date'] < pd.to_datetime('2016-1-1'))]
    styPDD8ary = (styPdD8TimeSeries['ddays'].values).reshape((len(styPdD8TimeSeries), 1))

    # open mask for study area
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskImg = r'C:\dissertation\twimage\LandsatScene\ProcExtent_admin_1b.tif'
    maskdataset = gdal.Open(maskImg, gdalconst.GA_ReadOnly)
    maskband = maskdataset.GetRasterBand(1)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotrasform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()
    if not maskgeotrasform is None:
        maskorig_x = maskgeotrasform[0]
        maskorig_y = maskgeotrasform[3]
        pixel_width = maskgeotrasform[1]
        pixel_height = maskgeotrasform[5]
    maskArray = maskband.ReadAsArray(0, 0, maskcols, maskrows)

    # create 3 empty arrays for store whether change, change range, and change time
    chgTypePath = r'C:\dissertation\twimage\ImpChg\impTypeChg' + str(startcol) + '.tif'
    chgRngPath = r'C:\dissertation\twimage\ImpChg\impChgRng' + str(startcol) + '.tif'
    chgDatePath = r'C:\dissertation\twimage\ImpChg\impChgDate' + str(startcol) + '.tif'
    # this is for storing change type
    outputdataset_ChgType = driver.Create(chgTypePath, maskcols, maskrows, 1, gdal.GDT_Byte)
    outChgTypeBand = outputdataset_ChgType.GetRasterBand(1)
    outChgType_array = np.zeros((maskrows, maskcols))
    # this is for storing change range
    outputdataset_ChgRng = driver.Create(chgRngPath, maskcols, maskrows, 1, gdal.GDT_Float32)
    outChgRngBand = outputdataset_ChgRng.GetRasterBand(1)
    outChgRng_array = np.ones((maskrows, maskcols)) * (-1)
    # this is for storing change date
    outputdataset_ChgDate = driver.Create(chgDatePath, maskcols, maskrows, 1, gdal.GDT_Float32)
    outChgDateBand = outputdataset_ChgDate.GetRasterBand(1)
    outChgDate_array = np.ones((maskrows, maskcols)) * (-1)

    accuDate = np.array([list(df['accuDate'])] * maskrows)

    for xoff in range(startcol, endcol, 1):
        print('processing {0} %'.format(xoff / maskcols * 100))
        Iry, Sry = extractOrgPxTimeSeries(r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent', xoff, maskrows)
        #pxTimeSeries = pd.read_table(r'C:\dissertation\twimage\urbanChg\UrbanChgAssess\ID1.txt', sep='\t', header=0)
        Iry = np.where(accuDate, Iry, np.nan)
        Sry = np.where(accuDate, Sry, np.nan)
        tmpCol = maskArray[:, xoff]
        usedPx = np.argwhere(tmpCol == 1)

        if len(usedPx)==0:
            continue

        for px in usedPx:
            ISer = pd.Series(Iry[px[0]], index=list(df['CDate']))
            SSer = pd.Series(Sry[px[0]], index=list(df['CDate']))

            IR = ISer.resample('8D')
            SR = SSer.resample('8D')

            I8DSer = IR.interpolate(method='linear')
            S8DSer = SR.interpolate(method='linear')

            I8DSer = SGFilter(np.array(I8DSer.tolist()), window, polyDegree)
            S8DSer = SGFilter(np.array(S8DSer.tolist()), window, polyDegree)

            I_range = np.ptp(I8DSer)

            if (I_range < 230):
                continue
            avgFirst10yrImp = np.nanmean(np.where(first10yr, I8DSer, np.nan))
            avgLast10yrImp = np.nanmean(np.where(last10yr, I8DSer, np.nan))

            if (avgLast10yrImp < 500) and (avgFirst10yrImp < 500):
                continue

            if (avgFirst10yrImp < 500) and (avgLast10yrImp >= 500):
                tmpdf = pd.DataFrame({'Date': df['CDate'],
                                      'Ipro': Iry[px[0]].tolist(),
                                      'ddays': list(df['ddays'])})
                LUtype, I_chg_range, ChgTime = LogisticUrbanSprawl(tmpdf, styPdD8TimeSeries, styPDD8ary)

            elif (avgFirst10yrImp >= 500) and (avgLast10yrImp >= 500):
                tmpdf = pd.DataFrame({'Date': rng8D,
                                      'Ipro': I8DSer.tolist(),
                                      'Spro': S8DSer.tolist()})
                LUtype, I_chg_range, ChgTime = UrbanRenewal(tmpdf)

            else:
                tmpdf = pd.DataFrame({'Date': df['CDate'],
                                      'Ipro': Iry[px[0]].tolist(),
                                      'ddays': list(df['ddays'])})
                LUtype, I_chg_range, ChgTime = LogisticUrbanAbandon(tmpdf, styPdD8TimeSeries, styPDD8ary)

            outChgType_array[px[0], xoff] = LUtype
            outChgRng_array[px[0], xoff] = I_chg_range
            outChgDate_array[px[0], xoff] = ChgTime

    outChgTypeBand.WriteArray(outChgType_array, 0, 0)
    outChgTypeBand.SetNoDataValue(0)
    outChgRngBand.WriteArray(outChgRng_array, 0, 0)
    outChgRngBand.SetNoDataValue(-1)
    outChgDateBand.WriteArray(outChgDate_array, 0, 0)
    outChgDateBand.SetNoDataValue(-1)
    outChgTypeBand = None
    outChgDateBand = None
    outChgRngBand = None

    outputdataset_ChgType.SetProjection(maskprojection)
    outputdataset_ChgType.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgDate.SetProjection(maskprojection)
    outputdataset_ChgDate.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgRng.SetProjection(maskprojection)
    outputdataset_ChgRng.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgType = None
    outputdataset_ChgRng = None
    outputdataset_ChgDate = None
    maskdataset = None


def UrbanChgTypeAndTimeinShp(RMSE, window, polyDegree, chgShp):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['CDate'] = pd.to_datetime(df['Date'])
    df['DeciYr'] = pd.DatetimeIndex(df['CDate']).year + (pd.DatetimeIndex(df['CDate']).dayofyear/366)
    df['accuDate'] = np.where((df['VRMSE'] < RMSE) &
                              (df['IRMSE'] < RMSE) &
                              (df['SRMSE'] < RMSE) &
                              (df['Cnt'] > 30), True, False)
    OdeciYr = df['DeciYr'].values

    rng8D = pd.date_range('1987/4/9', periods=1403, freq='8D')
    first10yr = rng8D < pd.to_datetime('1998-1-1')
    last10yr = (rng8D >= pd.to_datetime('2008-1-1'))
    LogDateDf = pd.DataFrame({'date': rng8D})
    LogDateDf['ODate'] = pd.to_datetime('1987-04-09')
    D8arr = np.array((rng8D.year + rng8D.dayofyear/366).tolist())
    D8arr_2D = D8arr.reshape((len(D8arr), 1))

    # open mask for study area
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskImg = r'C:\dissertation\twimage\LandsatScene\ProcExtent_admin_1b.tif'
    maskdataset = gdal.Open(maskImg, gdalconst.GA_ReadOnly)
    maskband = maskdataset.GetRasterBand(1)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotrasform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()
    if not maskgeotrasform is None:
        maskorig_x = maskgeotrasform[0]
        maskorig_y = maskgeotrasform[3]
        pixel_width = maskgeotrasform[1]
        pixel_height = maskgeotrasform[5]
    maskArray = maskband.ReadAsArray(0, 0, maskcols, maskrows)

    # create 3 empty arrays for store whether change, change range, and change time
    chgTypePath = r'C:\dissertation\twimage\ImpChg\NochgSam_impTypeChg.tif'
    chgRngPath = r'C:\dissertation\twimage\ImpChg\NochgSam_impChgRng.tif'
    chgDatePath = r'C:\dissertation\twimage\ImpChg\NochgSam_impChgDate.tif'
    # this is for storing change type
    outputdataset_ChgType = driver.Create(chgTypePath, maskcols, maskrows, 1, gdal.GDT_Byte)
    outChgTypeBand = outputdataset_ChgType.GetRasterBand(1)
    outChgType_array = np.zeros((maskrows, maskcols))
    # this is for storing change range
    outputdataset_ChgRng = driver.Create(chgRngPath, maskcols, maskrows, 1, gdal.GDT_Float32)
    outChgRngBand = outputdataset_ChgRng.GetRasterBand(1)
    outChgRng_array = np.ones((maskrows, maskcols)) * (-1)
    # this is for storing change date
    outputdataset_ChgDate = driver.Create(chgDatePath, maskcols, maskrows, 1, gdal.GDT_Float32)
    outChgDateBand = outputdataset_ChgDate.GetRasterBand(1)
    outChgDate_array = np.ones((maskrows, maskcols)) * (-1)
    accuDate = df['accuDate'].values
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(chgShp, 0)
    layer = ds.GetLayer()

    feature = layer.GetNextFeature()

    while feature:
        id = feature.GetField('Id')
        geom = feature.GetGeometryRef()
        geo_X = geom.GetX()
        geo_Y = geom.GetY()

        x_offset = int(round((geo_X - maskorig_x) / pixel_width))
        y_offset = int(round((geo_Y - maskorig_y) / pixel_height))

        xoff_list = [x_offset-1, x_offset, x_offset+1]
        yoff_list = [y_offset-1, y_offset, y_offset+1]

        for xoff in xoff_list:
            for yoff in yoff_list:
                Iry, Sry = extractOrgPxTimeSeries2(r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent', xoff, yoff)
                Iry = np.where(accuDate, Iry, np.nan)
                Sry = np.where(accuDate, Sry, np.nan)

                ISer = pd.Series(Iry, index=list(df['CDate']))
                SSer = pd.Series(Sry, index=list(df['CDate']))

                IR = ISer.resample('8D')
                SR = SSer.resample('8D')

                I8DSer = IR.interpolate(method='linear')
                S8DSer = SR.interpolate(method='linear')

                if len(Iry[~np.isnan(Iry)])< 100:
                    continue
                if len(Sry[~np.isnan(Sry)])< 100:
                    continue

                I8DSer = SGFilter(np.array(I8DSer.tolist()), window, polyDegree)
                S8DSer = SGFilter(np.array(S8DSer.tolist()), window, polyDegree)

                I_range = np.ptp(I8DSer)

                if (I_range < 230):
                    continue
                avgFirst10yrImp = np.nanmean(np.where(first10yr, I8DSer, np.nan))
                avgLast10yrImp = np.nanmean(np.where(last10yr, I8DSer, np.nan))

                if (avgLast10yrImp < 500) and (avgFirst10yrImp < 500):
                    continue

                if (avgFirst10yrImp < 500) and (avgLast10yrImp >= 500):
                    LUtype, I_chg_range, ChgTime = LogisticUrbanSprawl(OdeciYr, Iry, D8arr_2D)

                elif (avgFirst10yrImp >= 500) and (avgLast10yrImp >= 500):
                    LUtype, I_chg_range, ChgTime = UrbanRenewal(D8arr, I8DSer, S8DSer)

                elif(avgFirst10yrImp >= 500) and (avgLast10yrImp < 500):
                    LUtype, I_chg_range, ChgTime = LogisticUrbanAbandon(OdeciYr, Iry, D8arr_2D)

                else:
                    continue

                outChgType_array[yoff, xoff] = LUtype
                outChgRng_array[yoff, xoff] = I_chg_range
                outChgDate_array[yoff, xoff] = ChgTime
        feature.Destroy()
        feature = layer.GetNextFeature()

    ds.Destroy()
    maskds = None

    outChgTypeBand.WriteArray(outChgType_array, 0, 0)
    outChgTypeBand.SetNoDataValue(0)
    outChgRngBand.WriteArray(outChgRng_array, 0, 0)
    outChgRngBand.SetNoDataValue(-1)
    outChgDateBand.WriteArray(outChgDate_array, 0, 0)
    outChgDateBand.SetNoDataValue(-1)
    outChgTypeBand = None
    outChgDateBand = None
    outChgRngBand = None

    outputdataset_ChgType.SetProjection(maskprojection)
    outputdataset_ChgType.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgDate.SetProjection(maskprojection)
    outputdataset_ChgDate.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgRng.SetProjection(maskprojection)
    outputdataset_ChgRng.SetGeoTransform(maskgeotrasform)

    outputdataset_ChgType = None
    outputdataset_ChgRng = None
    outputdataset_ChgDate = None
    maskdataset = None


def UrbanChgTypeAndTimeRow(df, first5yr, last5yr, OdeciYr, D8arr, D8arr_2D, window, polyDegree, row):

    # open mask for study area
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskImg = r'C:\dissertation\twimage\LandsatScene\ProcExtent.tif'
    maskdataset = gdal.Open(maskImg, gdalconst.GA_ReadOnly)
    maskband = maskdataset.GetRasterBand(1)
    maskcols = maskdataset.RasterXSize

    maskArray = maskband.ReadAsArray(0, row, maskcols, 1)

    accuDate = np.array([list(df['accuDate'])] * maskcols)

    chgTypePath = r'C:\dissertation\twimage\ImpChgRow\Type_row' + str(row) + '.p'
    chgRngPath = r'C:\dissertation\twimage\ImpChgRow\Rng_row' + str(row) + '.p'
    chgDatePath = r'C:\dissertation\twimage\ImpChgRow\Date_row' + str(row) + '.p'

    TypeArr = [0] * maskcols
    RngArr = [-1] * maskcols
    DateArr = [-1] * maskcols

    Iry, Sry = extractOrgPxTimeSeries3(r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent2', row, maskcols)
    Iry = np.where(accuDate, Iry, np.nan)
    Sry = np.where(accuDate, Sry, np.nan)
    for index, xoff in enumerate(maskArray[0]):
        if xoff == 0:
            continue

        ISer = pd.Series(Iry[index], index=list(df['CDate']))
        SSer = pd.Series(Sry[index], index=list(df['CDate']))

        IR = ISer.resample('8D')
        SR = SSer.resample('8D')

        I8DSer = IR.interpolate(method='linear')
        S8DSer = SR.interpolate(method='linear')

        if len(Iry[~np.isnan(Iry)]) < 50:
            continue
        if len(Sry[~np.isnan(Sry)]) < 50:
            continue

        I8DSer = SGFilter(np.array(I8DSer.tolist()), window, polyDegree)
        S8DSer = SGFilter(np.array(S8DSer.tolist()), window, polyDegree)

        I_range = np.ptp(I8DSer)

        if (I_range < 230):
            continue
        avgFirst5yrImp = np.nanmean(np.where(first5yr, I8DSer, np.nan))
        avgLast5yrImp = np.nanmean(np.where(last5yr, I8DSer, np.nan))

        if (avgLast5yrImp < 500) and (avgFirst5yrImp < 500):
            continue

        if (avgFirst5yrImp < 500) and (avgLast5yrImp >= 500):
            TypeArr[index], RngArr[index], DateArr[index] = LogisticUrbanSprawl2(D8arr, D8arr_2D, I8DSer)

        elif (avgFirst5yrImp >= 500) and (avgLast5yrImp >= 500):
            TypeArr[index], RngArr[index], DateArr[index] = UrbanRenewal(D8arr, I8DSer, S8DSer)

        elif (avgFirst5yrImp >= 500) and (avgLast5yrImp < 500):
            TypeArr[index], RngArr[index], DateArr[index] = LogisticUrbanAbandon2(D8arr, D8arr_2D, I8DSer)

        else:
            continue
    TypefileObject = open(chgTypePath, 'wb')
    RngfileObject = open(chgRngPath, 'wb')
    DatefileObject = open(chgDatePath, 'wb')
    pickle.dump(TypeArr, TypefileObject)
    pickle.dump(RngArr, RngfileObject)
    pickle.dump(DateArr, DatefileObject)
    TypefileObject.close()
    RngfileObject.close()
    DatefileObject.close()
    print('Done for row {0}'.format(row))


def UrbanChgTypeAndTimeRow2(df, first5yr, last5yr, DeYr, D8DeYr, Ddays_1D, Ddays_2D, D8arr_1D, D8arr_2D, window, polyDegree, row):
    # open mask for study area
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    #maskImg = r'C:\dissertation\twimage\ImpChg\TypeChg.tif'
    maskImg = r'C:\dissertation\twimage\LandsatScene\ProcExtent.tif'
    maskdataset = gdal.Open(maskImg, gdalconst.GA_ReadOnly)
    maskband = maskdataset.GetRasterBand(1)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskArray = maskband.ReadAsArray(0, row, maskcols, 1)

    accuDate = np.array([list(df['accuDate'])] * maskcols)

    chgTypePath = r'C:\dissertation\twimage\ImpChgRow\Type_row' + str(row) + '.p'
    chgRngPath = r'C:\dissertation\twimage\ImpChgRow\Rng_row' + str(row) + '.p'
    chgDatePath = r'C:\dissertation\twimage\ImpChgRow\Date_row' + str(row) + '.p'

    TypeArr = [0] * maskcols
    RngArr = [-1] * maskcols
    DateArr = [-1] * maskcols
    Vry, Iry, Sry = extractOrgPxTimeSeries4(r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent2',
                                            yoffset=row, ImgWid=maskcols)
    Vry = np.where(accuDate, Vry, np.nan)
    Iry = np.where(accuDate, Iry, np.nan)
    Sry = np.where(accuDate, Sry, np.nan)

    for index, xoff in enumerate(maskArray[0]):
        if (xoff != 1):
            continue

        VSer = pd.Series(Vry[index], index=list(df['CDate']))
        ISer = pd.Series(Iry[index], index=list(df['CDate']))
        SSer = pd.Series(Sry[index], index=list(df['CDate']))

        VR = VSer.resample('8D')
        IR = ISer.resample('8D')
        SR = SSer.resample('8D')

        V8DSer = VR.interpolate(method='linear')
        I8DSer = IR.interpolate(method='linear')
        S8DSer = SR.interpolate(method='linear')

        if len(Vry[~np.isnan(Vry)]) < 50:
            continue
        if len(Iry[~np.isnan(Iry)]) < 50:
            continue
        if len(Sry[~np.isnan(Sry)]) < 50:
            continue

        #V8DSer = SGFilter(np.array(V8DSer.tolist()), window, polyDegree)
        I8DSer = SGFilter(np.array(I8DSer.tolist()), window, polyDegree)
        #S8DSer = SGFilter(np.array(S8DSer.tolist()), window, polyDegree)

        I_range = np.ptp(I8DSer)

        if (I_range < 230):
            continue
        avgFirst5yrImp = np.nanmean(np.where(first5yr, I8DSer, np.nan))
        avgLast5yrImp = np.nanmean(np.where(last5yr, I8DSer, np.nan))

        if (avgLast5yrImp < 500) and (avgFirst5yrImp < 500):
            continue

        if (avgFirst5yrImp < 500) and (avgLast5yrImp >= 500):
            TypeArr[index], RngArr[index], DateArr[index] = LogisticUrbanSprawl3(DeYr, D8DeYr,
                                                                                 Vry[index], Iry[index], Ddays_1D,
                                                                                 Ddays_2D, D8arr_1D, D8arr_2D)

        else:
            continue
    TypefileObject = open(chgTypePath, 'wb')
    RngfileObject = open(chgRngPath, 'wb')
    DatefileObject = open(chgDatePath, 'wb')
    pickle.dump(TypeArr, TypefileObject)
    pickle.dump(RngArr, RngfileObject)
    pickle.dump(DateArr, DatefileObject)
    TypefileObject.close()
    RngfileObject.close()
    DatefileObject.close()
    print('Done for row {0}'.format(row))


def combineRow(parsetxt, outimg):
    import os
    os.chdir(r'C:\dissertation\twimage\ImpChgRow')
    rows = []
    for i in range(3704):
        path = parsetxt + str(i) + '.p'
        file = open(path, 'rb')
        data = pickle.load(file)
        rows.append(data)
        file.close()

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskdataset = gdal.Open(r'C:\dissertation\twimage\LandsatScene\ProcExtent.tif', gdalconst.GA_ReadOnly)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotransform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()
    maskband = maskdataset.GetRasterBand(1)
    if not maskgeotransform is None:
        maskorig_x = maskgeotransform[0]
        maskorig_y = maskgeotransform[3]
        pix_wid = maskgeotransform[1]
        pix_hgt = maskgeotransform[5]

    if parsetxt == 'Type_row':
        dtyp = gdal.GDT_Byte
    else:
        dtyp = gdal.GDT_Float32

    outRas = driver.Create(outimg, maskcols, maskrows, 1, dtyp)
    outBand = outRas.GetRasterBand(1)
    outArr = np.array(np.array(rows))
    outBand.WriteArray(outArr, 0, 0)
    outBand.SetNoDataValue(0)
    outBand = None

    outRas.SetProjection(maskprojection)
    outRas.SetGeoTransform(maskgeotransform)
    outRas = None


def primaryImp(Unmixfolder):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['CDate'] = pd.to_datetime(df['Date'])
    df['accuDate'] = np.where((df['VRMSE'] < 27) &
                              (df['IRMSE'] < 27) &
                              (df['SRMSE'] < 27) &
                              (df['Cnt'] > 30), True, False)
    primaryVISArr = df[df['accuDate'] & (df['CDate'] < '1990-1-1')]['Date'].values

    maskimg = r'C:\dissertation\twimage\LandsatScene\ProcExtent_admin_1b.tif'
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    maskdataset = gdal.Open(maskimg, gdalconst.GA_ReadOnly)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotransform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()

    os.chdir(Unmixfolder)

    VegAry = np.zeros((maskrows, maskcols))
    ImpAry = np.zeros((maskrows, maskcols))
    for VISimg in primaryVISArr:
        path = VISimg + '.tif'
        tmpdataset = gdal.Open(path, gdalconst.GA_ReadOnly)
        tmpB1 = tmpdataset.GetRasterBand(1)
        tmpB2 = tmpdataset.GetRasterBand(2)

        VegArr = tmpB1.ReadAsArray(0, 0, maskcols, maskrows)
        ImpArr = tmpB2.ReadAsArray(0, 0, maskcols, maskrows)

        V = np.where(VegArr > 300, 1, 0)
        VegAry += V
        I = np.where(ImpArr > 500, 1, 0)
        ImpAry += I

    outArray = np.where(VegAry > 0, 0, ImpAry)
    outArray = np.where(outArray > 0, 1, 0)

    outImg = driver.Create(r'C:\dissertation\twimage\ImpChg2\Imp1990.tif', maskcols, maskrows, 1, gdal.GDT_Byte)
    outBand = outImg.GetRasterBand(1)
    outBand.WriteArray(outArray, 0, 0)
    outBand.SetNoDataValue(-1)
    outBand = None

    outImg.SetProjection(maskprojection)
    outImg.SetGeoTransform(maskgeotransform)


#unmixfo = r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent'
#primaryImp(unmixfo)

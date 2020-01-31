import numpy as np
import pandas as pd
import os
from osgeo import gdal
from osgeo import gdalconst
import datetime
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LogisticRegression


def imgDateMatrix():
    imglistfile = r'C:\dissertation\twimage\LandsatScene\espa_request.txt'
    yrMthMatrx = np.zeros((31, 12))
    f = open(imglistfile, 'r')
    str = f.readline()
    while str:
        date = str[17:25]
        tmpyr = int(date[:4])
        tmpMth = int(date[4:6])
        yrMthMatrx[tmpyr-1990, tmpMth-1] += 1
        str = f.readline()
    f.close()
    print(yrMthMatrx)
    return yrMthMatrx


def plotimgDate(imgdateMtrx):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    Z = imgdateMtrx

    fig = plt.figure()
    fig.set_figheight(30)
    fig.set_figwidth(5)
    ax = fig.add_subplot(111)
    cax = ax.matshow(Z, cmap='summer', interpolation=None, aspect='auto')
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])

    ax.set_yticklabels([''] + list(int(y) for y in range(1987, 2018, 1)))
    ax.set_xticklabels([''] + list(int(x) for x in range(1, 13, 1)))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    fig.savefig(r'C:\dissertation\figure\1st_SMA\imgMatx.png', dpi=400, format='png')


def dateTxtList2DateList(dateTxtList):
    outList = []
    for index, i in enumerate(dateTxtList):
        tmpdate = datetime.date(int(str(i)[:4]), int(str(i)[4:6]), int(str(i)[6:]))
        outList.append(tmpdate)
    return outList


def plotPxSeries(inSeries):
    import matplotlib.pyplot as plt
    ts = pd.Series(inSeries, index=pd.date_range('4/9/1987', periods=1403, freq='8D'))
    ts.plot()
    plt.show()


def reNameVIS(VISfolder):
    import glob
    os.chdir(VISfolder)
    VISSeries = glob.glob('*')
    for VIS in VISSeries:
        date = VIS[10:18]
        datetif = str(date)+'.tif'
        os.rename(VIS, datetif)
        print('Done rename to %s'% datetif)


def extractRegPxTimeSeries(Unmixfolder, VIStype, xoffset, yoffset):
    import glob

    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    firstdate = datetime.datetime(int(VIS_list[0][:4]), int(VIS_list[0][4:6]), int(VIS_list[0][6:8]))
    lastdate = datetime.datetime(int(VIS_list[-1][:4]), int(VIS_list[-1][4:6]), int(VIS_list[-1][6:8]))
    timedel = datetime.timedelta(days=8)
    seriesLen = int((int((lastdate - firstdate).days)/8)+1)
    LCSeries = [-1]*seriesLen

    for index, img in enumerate(LCSeries):
        tmpdate = datetime.datetime(int(img[:4]), int(img[4:6]), int(img[6:8]))
        if tmpdate != (firstdate+index*timedel):
            continue

        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band = dataset.GetRasterBand(VIStype)
        data = band.ReadAsArray(xoffset, yoffset, 1, 1)

        if data[0, 0] == 0:
            tmpb1 = dataset.GetRasterBand(1)
            tmpb2 = dataset.GetRasterBand(2)
            tmpb3 = dataset.GetRasterBand(3)
            b1data = tmpb1.ReadAsArray(xoffset, yoffset, 1, 1)
            b2data = tmpb2.ReadAsArray(xoffset, yoffset, 1, 1)
            b3data = tmpb3.ReadAsArray(xoffset, yoffset, 1, 1)
            tmpb1 = None
            tmpb2 = None
            tmpb3 = None
            if b1data[0, 0] == 0 and b2data[0, 0] == 0 and b3data[0, 0] == 0:
                continue

        LCSeries[index] = data[0, 0]
        dataset = None
        band = None

    return LCSeries


def extractOrgPxTimeSeries(Unmixfolder, xoffset, yoffset):
    import glob
    import datetime

    os.chdir(Unmixfolder)

    # load all raster files that their file name contains "VIS"
    VIS_list = glob.glob('*')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    datelist = []
    VSeries = []
    ISeries = []
    SSeries = []

    for index, img in enumerate(VIS_list):
        tmpdate = datetime.datetime(int(VIS_list[index][:4]), int(VIS_list[index][4:6]), int(VIS_list[index][6:8]))
        datelist.append(tmpdate)

        dataset = gdal.Open(img, gdalconst.GA_ReadOnly)
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)

        data1 = band1.ReadAsArray(xoffset, yoffset, 1, 1)
        data2 = band2.ReadAsArray(xoffset, yoffset, 1, 1)
        data3 = band3.ReadAsArray(xoffset, yoffset, 1, 1)

        if data1[0, 0] == 0 and data2[0, 0] == 0 and data3[0, 0] == 0:
            VSeries.append(-1)
            ISeries.append(-1)
            SSeries.append(-1)

        else:
            VSeries.append(data1[0, 0])
            ISeries.append(data2[0, 0])
            SSeries.append(data3[0, 0])

        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return datelist, VSeries, ISeries, SSeries


def regluar8dayInterval(datelist, VSeries, ISeries, SSeries):

    firstdate = datelist[0]
    lastdate = datelist[-1]
    timedel = datetime.timedelta(days=8)
    seriesLen = int((int((lastdate - firstdate).days)/8)+1)
    outVSeries = [-1] * seriesLen
    outISeries = [-1] * seriesLen
    outSSeries = [-1] * seriesLen
    outDateList = []

    for index, img in enumerate(outVSeries):
        tmpdate = firstdate+index*timedel
        outDateList.append(tmpdate)
        if tmpdate in datelist:
            outVSeries[index] = VSeries[datelist.index(tmpdate)]
            outISeries[index] = ISeries[datelist.index(tmpdate)]
            outSSeries[index] = SSeries[datelist.index(tmpdate)]
        else:
            continue

    return outDateList, outVSeries, outISeries, outSSeries


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


def movingTtest(inSeries):
    from scipy import stats
    import math

    pval = np.ones(len(inSeries))

    if len(inSeries) == 295:
        for index, prop in enumerate(inSeries):
            # 14 is deterimined by the second earliest date between 1990 and 2015
            # 1131 is determined by last date between 1990 and 2015
            if index < 14 or index > 270:
                continue

            prelist = inSeries[:index+1]
            prelist = [value for value in prelist if not math.isnan(value)]
            prolist = inSeries[index+1:]
            prolist = [value for value in prolist if not math.isnan(value)]
            TRes = stats.ttest_ind(prelist, prolist, equal_var=False)
            pval[index] = TRes[1]
    else:
        for index, prop in enumerate(inSeries):
            # 183 is deterimined by 365/8*4 (year days/ 8 day interval * 4 years) to identify change between 1990 and 2015
            # 1131 is determined by 1403 - int(365/8*2) minus 1 for list index
            if index < 183 or index > 1311:
                continue
            prelist = inSeries[:index + 1]
            prelist = [value for value in prelist if not math.isnan(value)]
            prolist = inSeries[index + 1:]
            prolist = [value for value in prolist if not math.isnan(value)]
            TRes = stats.ttest_ind(prelist, prolist, equal_var=False, nan_policy='omit')
            pval[index] = TRes[1]

    return pval


def movingWindowTtest(Datelist, inSeries, yearInterval):
    from scipy import stats
    import math

    pval = np.ones(len(inSeries))

    if len(inSeries) == 295:
        for index, prop in enumerate(inSeries):
            # 14 is deterimined by the second earliest date between 1990 and 2015
            # 1131 is determined by last date between 1990 and 2015
            if Datelist[index] < np.datetime64('1990-01-01') or Datelist[index] > np.datetime64('2015-12-31'):
                continue

            tmpSeries = inSeries

            timedel = np.timedelta64(int(365*yearInterval), 'D')
            earlythreshold = Datelist[index] - timedel
            latethreshold = Datelist[index] + timedel

            early = Datelist >= earlythreshold
            late = Datelist <= latethreshold
            monitorPeriod = early * late
            monitorPeriod = monitorPeriod.tolist()

            for index, t in enumerate(monitorPeriod):
                if not t:
                    tmpSeries[index] = np.nan

            prelist = tmpSeries[:index+1]
            prelist = [value for value in prelist if not math.isnan(value)]
            prolist = tmpSeries[index+1:]
            prolist = [value for value in prolist if not math.isnan(value)]
            TRes = stats.ttest_ind(prelist, prolist, equal_var=False)
            pval[index] = TRes[1]
    else:
        for index, prop in enumerate(inSeries):
            # 183 is deterimined by 365/8*4 (year days/ 8 day interval * 4 years) to identify change between 1987 and 2017
            # 1131 is determined by 1403 - int(365/8*2) minus 1 for list index
            if Datelist[index] < np.datetime64('1990-01-01') or Datelist[index] > np.datetime64('2015-12-31'):
                continue

            tmpSeries = np.array(inSeries)

            timedel = np.timedelta64(int(365*yearInterval), 'D')
            earlythreshold = Datelist[index] - timedel
            latethreshold = Datelist[index] + timedel

            early = np.where(Datelist >= earlythreshold, 1, np.nan)
            late = np.where(Datelist <= latethreshold, 1, np.nan)
            monitorPeriod = (early * late) * tmpSeries

            prelist = monitorPeriod[:index]
            prelist = [value for value in prelist if not math.isnan(value)]
            prolist = monitorPeriod[index:]
            prolist = [value for value in prolist if not math.isnan(value)]
            TRes = stats.ttest_ind(prelist, prolist, equal_var=False, nan_policy='omit')
            pval[index] = TRes[1]

    return pval


def plotMultiSeries(filepath, Filter=False, filterWindow=23, polydegree=1):
    samples = pd.read_table(filepath, sep='\t', header=0)
    datelist = dateTxtList2DateList(samples['Date'].tolist())
    samples['Date'] = datelist

    date8list, V8list, I8list, S8list = regluar8dayInterval(datelist,
                                                            samples['Vpro'].tolist(),
                                                            samples['Ipro'].tolist(), samples['Spro'].tolist())

    V8list = timeSeriesInterpolation(V8list)
    I8list = timeSeriesInterpolation(I8list)
    S8list = timeSeriesInterpolation(S8list)

    if Filter:
        V8list = SGFilter(V8list, filterWindow, polydegree)
        I8list = SGFilter(I8list, filterWindow, polydegree)
        S8list = SGFilter(S8list, filterWindow, polydegree)

    inter8df = pd.DataFrame({'date': date8list, 'Vpro': V8list.tolist(), 'Ipro': I8list.tolist(), 'Spro': S8list.tolist()}, index=date8list)
    inter8df['Vpro'] = inter8df['Vpro'] / 10
    inter8df['Ipro'] = inter8df['Ipro'] / 10
    inter8df['Spro'] = inter8df['Spro'] / 10

    fig = plt.figure(figsize=(12, 5))
    axV = inter8df.Vpro.plot(color='green', label='veg')
    axI = inter8df.Ipro.plot(color='blue', label='imp')
    axS = inter8df.Spro.plot(color='red', label='soil')

    plt.ylim((0, 100))
    plt.xlim(('1987-4-1', '2018-1-1'))

    axV.legend()
    axI.legend()
    axS.legend()

    plt.show()
    outimg = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\plot35_SG55.png'
    fig.savefig(outimg, dpi=400, format='png')


def timeSeriesRMSEforNoChg():
    import re
    workspace = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess'
    os.chdir(workspace)
    VISRefPath = r'VISnoChgProp_selectedDates.txt'
    VISRef = pd.read_table(VISRefPath, sep=',', header=0)
    VISRef = (VISRef[['Id', 'Vpro', 'Ipro', 'Spro']].copy()).sort_values(by=['Id'])
    samplelist = glob.glob('*ID*')

    alldf = np.loadtxt(samplelist[0], skiprows=1)
    samID = np.ones((295, 1), dtype=int) * int(re.search(r'\d+', samplelist[0]).group())
    alldf = np.hstack((samID, alldf))
    for index, i in enumerate(samplelist):
        if index == 0:
            continue
        tmpdf = np.loadtxt(samplelist[index], skiprows=1)
        samID = np.ones((295, 1), dtype=int) * int(re.search(r'\d+', i).group())
        tmpdf = np.hstack((samID, tmpdf))
        alldf = np.concatenate((alldf, tmpdf))

    alldf = pd.DataFrame(data=alldf, columns=['Id', 'Date', 'VproM', 'IproM', 'SproM'])
    alldf['Id'] = alldf['Id'].astype(int)
    alldf['Date'] = alldf['Date'].astype(int)

    f = open(r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt', 'w')
    f.write('Date\tMAE\tRMSE\tSE\tVMAE\tVRMSE\tVSE\tIMAE\tIRMSE\tISE\tSMAE\tSRMSE\tSSE\tCnt\n')

    for date in sorted(set(alldf['Date'].tolist())):
        tmpdf = alldf.loc[alldf['Date'] == date]
        refTmpdf = VISRef.set_index('Id').join(tmpdf.set_index('Id'))
        refTmpdf = refTmpdf.loc[refTmpdf['VproM'] != -1]

        if len(refTmpdf) == 0:
            f.write(str(date)+'\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\n')
            continue

        refTmpdf['VRes'] = (refTmpdf['Vpro'] - refTmpdf['VproM'])/10
        refTmpdf['IRes'] = (refTmpdf['Ipro'] - refTmpdf['IproM'])/10
        refTmpdf['SRes'] = (refTmpdf['Spro'] - refTmpdf['SproM'])/10

        MAE = (refTmpdf['VRes'].abs() + refTmpdf['IRes'].abs() + refTmpdf['SRes'].abs()).sum() / (len(refTmpdf)*3)
        RMSE = np.sqrt((refTmpdf['VRes']**2 + refTmpdf['IRes']**2 + refTmpdf['SRes']**2).sum() / (len(refTmpdf)*3))
        SE = (refTmpdf['VRes'] + refTmpdf['IRes'] + refTmpdf['SRes']).sum() / (len(refTmpdf)*3)

        VMAE = (refTmpdf['VRes'].abs()).mean()
        VRMSE = np.sqrt((refTmpdf['VRes']**2).mean())
        VSE = (refTmpdf['VRes']).mean()
        IMAE = (refTmpdf['IRes'].abs()).mean()
        IRMSE = np.sqrt((refTmpdf['IRes']**2).mean())
        ISE = (refTmpdf['IRes']).mean()
        SMAE = (refTmpdf['SRes'].abs()).mean()
        SRMSE = np.sqrt((refTmpdf['SRes']**2).mean())
        SSE = (refTmpdf['SRes']).mean()

        f.write(str(date)+'\t'+str(MAE)+'\t'+str(RMSE)+'\t'+str(SE)+'\t'
                +str(VMAE)+'\t'+str(VRMSE)+'\t'+str(VSE)+'\t'
                +str(IMAE)+'\t'+str(IRMSE)+'\t'+str(ISE)+'\t'
                +str(SMAE)+'\t'+str(SRMSE)+'\t'+str(SSE)+'\t'
                +str(len(refTmpdf))+'\n')

    f.flush()
    f.close()


def RMSEFilteredTimeSeries(workdir, samID, RMSEThreshold=26.71):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['year'] = (df['Date'].str[:4]).astype(int)
    df['month'] = (df['Date'].str[4:6]).astype(int)
    df['day'] = (df['Date'].str[6:]).astype(int)
    df['accuDate'] = np.where((df['VRMSE'] < RMSEThreshold) &
                              (df['IRMSE'] < RMSEThreshold) &
                              (df['SRMSE'] < RMSEThreshold) &
                              (df['Cnt'] > 30), True, False)

    os.chdir(workdir)
    sample = 'ID' + str(samID) + '.txt'
    samples = pd.read_table(sample, sep='\t', header=0)
    samples['Date'] = samples['Date'].astype(str)
    samplesJoin = samples.set_index('Date').join(df.set_index('Date'))
    samplesJoin['VproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Vpro'], -1)
    samplesJoin['IproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Ipro'], -1)
    samplesJoin['SproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Spro'], -1)

    outdf = pd.DataFrame({'Date': samples['Date'].tolist(),
                          'Vpro': samplesJoin['VproF'].tolist(),
                          'Ipro': samplesJoin['IproF'].tolist(),
                          'Spro': samplesJoin['SproF'].tolist()})

    return outdf


def PlotRMSEFilteredSmoothedTimeSeries(workdir, samID, RMSEThreshold=26.71, smooth=False, window=23, polyDegree=1):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['year'] = (df['Date'].str[:4]).astype(int)
    df['month'] = (df['Date'].str[4:6]).astype(int)
    df['day'] = (df['Date'].str[6:]).astype(int)
    df['accuDate'] = np.where((df['VRMSE'] < RMSEThreshold) &
                              (df['IRMSE'] < RMSEThreshold) &
                              (df['SRMSE'] < RMSEThreshold) &
                              (df['Cnt'] > 30), True, False)

    os.chdir(workdir)
    sample = 'ID' + str(samID) + '.txt'
    samples = pd.read_table(sample, sep='\t', header=0)
    samples['Date'] = samples['Date'].astype(str)
    samplesJoin = samples.set_index('Date').join(df.set_index('Date'))
    samplesJoin['VproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Vpro'], -1)
    samplesJoin['IproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Ipro'], -1)
    samplesJoin['SproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Spro'], -1)

    datelist = dateTxtList2DateList(samples['Date'].tolist())
    date8list, V8list, I8list, S8list = regluar8dayInterval(datelist,
                                                            samplesJoin['VproF'].tolist(),
                                                            samplesJoin['IproF'].tolist(),
                                                            samplesJoin['SproF'].tolist())
    V8list = timeSeriesInterpolation(V8list)
    I8list = timeSeriesInterpolation(I8list)
    S8list = timeSeriesInterpolation(S8list)

    if smooth:
        V8list = SGFilter(V8list, window, polyDegree)
        I8list = SGFilter(I8list, window, polyDegree)
        S8list = SGFilter(S8list, window, polyDegree)

    inter8df = pd.DataFrame({'date': date8list, 'Vpro': V8list.tolist(), 'Ipro': I8list.tolist(), 'Spro': S8list.tolist()}, index=date8list)

    inter8df['Vpro'] = inter8df['Vpro'] / 10
    inter8df['Ipro'] = inter8df['Ipro'] / 10
    inter8df['Spro'] = inter8df['Spro'] / 10

    fig = plt.figure(figsize=(12, 5))
    axV = inter8df.Vpro.plot(color='green', label='veg')
    axI = inter8df.Ipro.plot(color='blue', label='imp')
    axS = inter8df.Spro.plot(color='red', label='soil')

    plt.ylim((0, 100))
    plt.xlim(('1987-4-1', '2018-1-1'))

    axV.legend()
    axI.legend(fontsize=20)
    axS.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    outimg = 'plot'+str(samID)+'_nochange.png'
    fig.savefig(outimg, dpi=400, format='png')


def PlotlogisticUrbanSprawl(workdir, samID, RMSEThreshold=26.71, smooth=False, window=23, polyDegree=1):
    from sklearn.linear_model import LogisticRegression
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['year'] = (df['Date'].str[:4])
    df['month'] = (df['Date'].str[4:6])
    df['day'] = (df['Date'].str[6:])
    df['CDate'] = pd.to_datetime(df['year']+'-'+df['month']+'-'+df['day'])
    df['ODate'] = pd.to_datetime('1987-04-09')
    df['ddays'] = (df['CDate'] - df['ODate']).dt.days
    df['accuDate'] = np.where((df['VRMSE'] < RMSEThreshold) &
                              (df['IRMSE'] < RMSEThreshold) &
                              (df['SRMSE'] < RMSEThreshold) &
                              (df['Cnt'] > 30), True, False)

    os.chdir(workdir)
    sample = 'ID' + str(samID) + '.txt'
    samples = pd.read_table(sample, sep='\t', header=0)
    samples['Date'] = samples['Date'].astype(str)
    samplesJoin = samples.set_index('Date').join(df.set_index('Date'))
    samplesJoin['VproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Vpro'], -1)
    samplesJoin['IproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Ipro'], -1)
    samplesJoin['SproF'] = np.where(samplesJoin['accuDate'], samplesJoin['Spro'], -1)

    testdf = samplesJoin.loc[samplesJoin['IproF'] > -1]
    I = np.where(testdf['IproF'] >= 500, 1, 0)
    testdf = samplesJoin.loc[samplesJoin['VproF'] > -1]
    V = np.where(testdf['VproF'] >= 500, 1, 0)

    testDateNum = (testdf['ddays'].values).reshape((len(testdf), 1))
    Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(testDateNum, I)
    dateNum = (samplesJoin['ddays'].values).reshape((len(samplesJoin), 1))
    Ipred = Iclf.predict_proba(dateNum)

    samplesJoin['PredictImp'] = (Ipred[:, 1]*1000).tolist()
    datelist = dateTxtList2DateList(samples['Date'].tolist())
    date8list, V8list, I8list, S8list = regluar8dayInterval(datelist,
                                                            samplesJoin['VproF'].tolist(),
                                                            samplesJoin['IproF'].tolist(),
                                                            samplesJoin['PredictImp'].tolist())
    V8list = timeSeriesInterpolation(V8list)
    I8list = timeSeriesInterpolation(I8list)
    S8list = timeSeriesInterpolation(S8list)

    if smooth:
        V8list = SGFilter(V8list, window, polyDegree)
        I8list = SGFilter(I8list, window, polyDegree)
        S8list = SGFilter(S8list, window, polyDegree)

    date8StrList = []
    for date in date8list:
        dateStr = date.strftime('%Y-%m-%d')
        date8StrList.append(dateStr)

    inter8df = pd.DataFrame({'date': date8StrList, 'Vpro': V8list.tolist(), 'Ipro': I8list.tolist(), 'Spro': S8list.tolist()}, index=date8list)
    """
    inter8df['CDate'] = pd.to_datetime(inter8df['date'])
    inter8df['ODate'] = pd.to_datetime('1987-04-09')
    inter8df['ddays'] = (inter8df['CDate'] - inter8df['ODate']).dt.days

    I = np.where(inter8df['Ipro'] >= 500, 1, 0)
    testDateNum = (inter8df['ddays'].values).reshape((len(inter8df), 1))
    Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(testDateNum, I)
    dateNum = (inter8df['ddays'].values).reshape((len(inter8df), 1))
    Ipred = Iclf.predict_proba(dateNum)
    print((Ipred[:, 1] * 1000).tolist())
    inter8df['PredictImp'] = (Ipred[:, 1] * 1000).tolist()
    """

    inter8df['Vpro'] = inter8df['Vpro'] / 10
    inter8df['Ipro'] = inter8df['Ipro'] / 10
    inter8df['Spro'] = inter8df['Spro'] / 10

    date = (((inter8df.loc[inter8df['Spro'] >= 50])['date']).tolist())[0]

    pdate = pd.Series(pd.to_datetime(date))
    print(pdate)

    fig = plt.figure(figsize=(12, 5))
    #axV = inter8df.Vpro.plot(color='green', label='veg')
    axI = inter8df.Ipro.plot(color='blue', label='imp')
    axS = inter8df.Spro.plot(color='orange', label='logistics modeled imp')

    plt.ylim((0, 100))
    plt.xlim(('1987-4-1', '2018-1-1'))

    #axV.legend()
    axI.legend(fontsize=20)
    axS.legend(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()
    #outimg = 'plot'+str(samID)+'_Renewal.png'
    #fig.savefig(outimg, dpi=400, format='png')


def LogisticUrbanSprawl(pxTimeSeries, D8TimeSeries):
    testdf = pxTimeSeries.loc[pxTimeSeries['Ipro'] > -1]
    I = np.where(testdf['Ipro'] >= 500, 1, 0)

    testDateNum = (testdf['ddays'].values).reshape((len(testdf), 1))
    Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(testDateNum, I)
    tmpD8TimeSeries = D8TimeSeries.loc[
        (D8TimeSeries['date'] > pd.to_datetime('1989-12-31')) & (D8TimeSeries['date'] < pd.to_datetime('2016-1-1'))]
    dateNum = (tmpD8TimeSeries['ddays'].values).reshape((len(tmpD8TimeSeries), 1))
    Ipred = Iclf.predict_proba(dateNum)

    tmpD8TimeSeries['PredictImp'] = (Ipred[:, 1]*100).tolist()

    outDf = (tmpD8TimeSeries.loc[tmpD8TimeSeries['PredictImp'] >= 50])

    if len(outDf)==0:
        Utype = 0
        outRange = 0.
        outDate = -1

    elif (tmpD8TimeSeries['PredictImp'].max() - tmpD8TimeSeries['PredictImp'].min()) < 50:
        Utype = 0
        outRange = 0.
        outDate = -1

    else:
        Utype = 1
        date = outDf['date'].iloc[0]
        outDate = (float(date.year) + (float(date.dayofyear) / 366))
        beforeAvgImp = ((pxTimeSeries.loc[pxTimeSeries['Date'] < date])['Ipro']).mean()
        afterAvgImp = ((pxTimeSeries.loc[pxTimeSeries['Date'] > date])['Ipro']).mean()
        outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def UrbanRenewal(pxTimeSeries):
    #pxTimeSeries['SoiDom'] = np.where((pxTimeSeries['Spro']>pxTimeSeries['Ipro']), (pxTimeSeries['Spro']-pxTimeSeries['Ipro']), -1)
    #outDf = pxTimeSeries.loc[(pxTimeSeries['Date'] > pd.to_datetime('1989-12-31')) & (pxTimeSeries['Date'] < pd.to_datetime('2016-1-1'))
    tmppx = pxTimeSeries.loc[(pxTimeSeries['Date'] > pd.to_datetime('1989-12-31')) & (pxTimeSeries['Date'] < pd.to_datetime('2016-1-1'))]
    minI = tmppx['Ipro'].min()
    maxS = tmppx['Spro'].max()
    minID = list((tmppx.loc[tmppx['Ipro']==minI])['Date'])
    #minID = minID.reset_index()
    maxSD = list((tmppx.loc[tmppx['Spro']==maxS])['Date'])
    #maxSD = maxSD.reset_index()
    if len(minID) > 0:
        minID = pd.to_datetime(minID[0])
    if len(maxSD) > 0:
        maxSD = pd.to_datetime(maxSD[0])

    meanI = ((tmppx.loc[tmppx['Date']!= minID])['Ipro']).mean()
    meanS = ((tmppx.loc[tmppx['Date']!= maxSD])['Spro']).mean()

    if (minID == maxSD):
        if (minI < maxS) and (meanI > meanS) and ((maxS - meanS) > 230) and ((meanI - minI) > 230):
            Utype = 2
            outRange = (pxTimeSeries.loc[pxTimeSeries['Date'] > minID])['Ipro'].mean() - (pxTimeSeries.loc[pxTimeSeries['Date'] < minID])['Ipro'].mean()
            date = minID
            outDate = (float(date.year) + (float(date.dayofyear) / 366))

        else:
            Utype = 0
            outRange = 0.
            outDate = -1

    else:
        Utype = 0
        outRange = 0.
        outDate = -1

    return Utype, outRange, outDate


def LogisticUrbanAbandon(pxTimeSeries, D8TimeSeries):
    testdf = pxTimeSeries.loc[pxTimeSeries['Ipro'] > -1]
    I = np.where(testdf['Ipro'] >= 500, 1, 0)

    testDateNum = (testdf['ddays'].values).reshape((len(testdf), 1))
    Iclf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(testDateNum, I)
    tmpD8TimeSeries = D8TimeSeries.loc[
        (D8TimeSeries['date'] > pd.to_datetime('1989-12-31')) & (D8TimeSeries['date'] < pd.to_datetime('2016-1-1'))]
    dateNum = (tmpD8TimeSeries['ddays'].values).reshape((len(tmpD8TimeSeries), 1))
    Ipred = Iclf.predict_proba(dateNum)

    tmpD8TimeSeries['PredictImp'] = (Ipred[:, 1] * 100).tolist()


    outDf = (tmpD8TimeSeries.loc[tmpD8TimeSeries['PredictImp'] < 50])

    if (len(outDf) == 0) or (tmpD8TimeSeries['PredictImp'].max()<50):
        Utype = 0
        outRange = 0.
        outDate = -1
    else:

        Utype = 3
        date = outDf['date'].iloc[0]
        outDate = (float(date.year) + (float(date.dayofyear) / 366))
        beforeAvgImp = ((pxTimeSeries.loc[pxTimeSeries['Date'] < date])['Ipro']).mean()
        afterAvgImp = ((pxTimeSeries.loc[pxTimeSeries['Date'] > date])['Ipro']).mean()
        outRange = afterAvgImp - beforeAvgImp

    return Utype, outRange, outDate


def UrbanChgTypeAndTime(RMSE, window, polyDegree, startcol, endcol):
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['year'] = (df['Date'].str[:4])
    df['month'] = (df['Date'].str[4:6])
    df['day'] = (df['Date'].str[6:])
    df['CDate'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'])
    df['ODate'] = pd.to_datetime('1987-04-09')
    df['ddays'] = (df['CDate'] - df['ODate']).dt.days
    df['accuDate'] = np.where((df['VRMSE'] < RMSE) &
                              (df['IRMSE'] < RMSE) &
                              (df['SRMSE'] < RMSE) &
                              (df['Cnt'] > 30), True, False)

    rng8D = pd.date_range('1987/4/9', periods=1403, freq='8D')
    first10yr = rng8D < pd.to_datetime('1998-1-1')
    mid10yr = (rng8D >= pd.to_datetime('1998-1-1')) * (rng8D < pd.to_datetime('2008-1-1'))
    last10yr = (rng8D >= pd.to_datetime('2008-1-1'))
    LogDateDf = pd.DataFrame({'date': rng8D})
    LogDateDf['ODate'] = pd.to_datetime('1987-04-09')
    LogDateDf['ddays'] = (LogDateDf['date'] - LogDateDf['ODate']).dt.days

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
    outChgRng_array = np.ones((maskrows, maskcols)) *(-1)
    # this is for storing change date
    outputdataset_ChgDate = driver.Create(chgDatePath, maskcols, maskrows, 1, gdal.GDT_Float32)
    outChgDateBand = outputdataset_ChgDate.GetRasterBand(1)
    outChgDate_array = np.ones((maskrows, maskcols)) * (-1)
    totalproc = maskrows*maskcols
    cur_proc = 0

    for xoff in range(startcol, endcol, 1):

        for yoff in range(maskrows):
            cur_proc += 1
            if (cur_proc % 300 == 0):
                print('processing {0} %'.format(cur_proc/totalproc*100))

            if maskArray[yoff, xoff]==0:
                continue

            datelist, Vlist, Ilist, Slist = extractOrgPxTimeSeries(r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent', xoff, yoff)
            #pxTimeSeries = pd.read_table(r'C:\dissertation\twimage\urbanChg\UrbanChgAssess\ID1.txt', sep='\t', header=0)
            pxTimeSeries = pd.DataFrame({'Date': list(df['Date']), 'Vpro': Vlist, 'Ipro': Ilist, 'Spro': Slist})
            pxTimeSeries['accuDate'] = df['accuDate']
            pxTimeSeries['VproF'] = np.where(pxTimeSeries['accuDate'], pxTimeSeries['Vpro'], -1)
            pxTimeSeries['IproF'] = np.where(pxTimeSeries['accuDate'], pxTimeSeries['Ipro'], -1)
            pxTimeSeries['SproF'] = np.where(pxTimeSeries['accuDate'], pxTimeSeries['Spro'], -1)
            date8list, V8list, I8list, S8list = regluar8dayInterval(datelist,
                                                            pxTimeSeries['VproF'].tolist(),
                                                            pxTimeSeries['IproF'].tolist(),
                                                            pxTimeSeries['SproF'].tolist())

            V8list = timeSeriesInterpolation(V8list)
            I8list = timeSeriesInterpolation(I8list)
            S8list = timeSeriesInterpolation(S8list)

            V8list = SGFilter(V8list, window, polyDegree)
            I8list = SGFilter(I8list, window, polyDegree)
            S8list = SGFilter(S8list, window, polyDegree)

            V_range = np.ptp(V8list)
            I_range = np.ptp(I8list)
            S_range = np.ptp(S8list)

            if (I_range < 230):
                continue

            avgFirst10yrImp = np.nanmean(np.where(first10yr, I8list, np.nan))
            avgLast10yrImp = np.nanmean(np.where(last10yr, I8list, np.nan))

            if (avgFirst10yrImp < 500) and (avgLast10yrImp >= 500):
                tmpdf = pd.DataFrame({'Date': df['CDate'],
                                      'Vpro': list(pxTimeSeries['VproF']),
                                      'Ipro': list(pxTimeSeries['IproF']),
                                      'Spro': list(pxTimeSeries['SproF']),
                                      'ddays': list(df['ddays'])})
                tmpLogDateDf = LogDateDf
                Utype, I_chg_range, ChgTime = LogisticUrbanSprawl(tmpdf, tmpLogDateDf)

            elif (avgFirst10yrImp >= 500) and (avgLast10yrImp >= 500):
                tmpdf = pd.DataFrame({'Date': rng8D,
                                      'Vpro': V8list.tolist(),
                                      'Ipro': I8list.tolist(),
                                      'Spro': S8list.tolist()})
                Utype, I_chg_range, ChgTime = UrbanRenewal(tmpdf)

            elif (avgFirst10yrImp >= 500) and (avgLast10yrImp < 500):
                tmpdf = pd.DataFrame({'Date': df['CDate'],
                                      'Vpro': list(pxTimeSeries['VproF']),
                                      'Ipro': list(pxTimeSeries['IproF']),
                                      'Spro': list(pxTimeSeries['SproF']),
                                      'ddays': list(df['ddays'])})
                tmpLogDateDf = LogDateDf
                Utype, I_chg_range, ChgTime = LogisticUrbanAbandon(tmpdf, tmpLogDateDf)
            else:
                Utype = 0
                I_chg_range = 0
                ChgTime = -1


            outChgType_array[yoff, xoff] = Utype
            outChgRng_array[yoff, xoff] = I_chg_range
            outChgDate_array[yoff, xoff] = ChgTime

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


#UrbanChgTypeAndTime(27, 45, 1)

#dmatrix = imgDateMatrix()
#plotimgDate(dmatrix)
#PlotlogisticUrbanSprawl(r'C:\dissertation\twimage\urbanChg\UrbanChgAssess', 60, RMSEThreshold=27, smooth=True, window=45)
#PlotRMSEFilteredSmoothedTimeSeries(r'C:\dissertation\twimage\VISnoChg\SingleDateAssess', 102, RMSEThreshold=27, smooth=True, window=45)
#plotMultiSeries(r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\ID35.txt', Filter=True, filterWindow=55)
#PlotRMSEFilteredSmoothedTimeSeries(r'C:\dissertation\twimage\urbanChg\UrbanChgAssess', 32, RMSEThreshold=27, smooth=True, window=45)
#RMSEFilteredTimeSeries(r'C:\dissertation\twimage\VISnoChg\SingleDateAssess', 0, RMSEThreshold=25)
#plotMultiSeries(r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\ID0.txt')
#timeSeriesRMSEforNoChg()

#testSample = r'C:\dissertation\twimage\urbanChg\UrbanChgAssess\ID51.txt'

#samples = pd.read_table(testSample, sep='\t', header=0)

#PlotRMSEFilteredSmoothedTimeSeries(r'C:\dissertation\twimage\ImpRegress', 2, RMSEThreshold=27, smooth=True, window=45)

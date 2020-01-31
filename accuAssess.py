import numpy as np
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
from osgeo import osr
import os
import pandas as pd
import glob
import datetime


def VISPropforNoChg(targetShp, VISRefFolder):
    os.chdir(VISRefFolder)
    foldername = os.path.basename(os.getcwd())

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(targetShp, 1)
    layer = ds.GetLayer()
    '''
    vfd = ogr.FieldDefn('Vpro', ogr.OFTReal)
    ifd = ogr.FieldDefn('Ipro', ogr.OFTReal)
    sfd = ogr.FieldDefn('Spro', ogr.OFTReal)

    layer.CreateField(vfd)
    layer.CreateField(ifd)
    layer.CreateField(sfd)
    '''
    F = open('VISpro.txt', 'w')
    F.write('Id\tVpro\tIpro\tSpro\n')
    feature = layer.GetNextFeature()

    while feature:
        id = feature.GetField('Id')
        targetSample = 'ID' + str(id) + '_VIS.shp'
        sam_ds = driver.Open(targetSample, 1)
        sam_layer = sam_ds.GetLayer()
        Varea = 0.
        Iarea = 0.
        Sarea = 0.
        sam_feature = sam_layer.GetNextFeature()

        while sam_feature:
            LCtype = sam_feature.GetField('Name')
            LCarea = sam_feature.GetField('POLY_AREA')

            if LCtype == 'V':
                Varea += LCarea
            elif LCtype == 'I':
                Iarea += LCarea
            else:
                Sarea += LCarea

            sam_feature.Destroy()
            sam_feature = sam_layer.GetNextFeature()

        Vpro = float(Varea) * 1000 / float(Varea + Iarea + Sarea)
        Ipro = float(Iarea) * 1000 / float(Varea + Iarea + Sarea)
        Spro = float(Sarea) * 1000 / float(Varea + Iarea + Sarea)
        '''
        feature.SetField('Vpro', Vpro)
        feature.SetField('Ipro', Ipro)
        feature.SetField('Spro', Spro)
        '''
        F.write(str(id)+'\t'+str(Vpro)+'\t'+str(Ipro)+'\t'+str(Spro)+'\n')
        sam_ds.Destroy()
        feature.Destroy()
        feature = layer.GetNextFeature()
    F.flush()
    F.close()
    ds.Destroy()


def refVISPropForChg(targetShp, VISRefFolder):
    os.chdir(VISRefFolder)
    allRefDate = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    for i in range(1, 61):
        targetSample = 'ID' + str(i) + '_VIS.shp'
        ds = driver.Open(targetSample, 0)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        while feature:
            LCDate = feature.GetField('Name')
            tmpDate = LCDate[2:]
            if tmpDate not in allRefDate:
                allRefDate.append(tmpDate)

            feature.Destroy()
            feature = layer.GetNextFeature()

        ds.Destroy()

    allRefDate.sort()

    f = open('VISforChange.txt', 'w')
    head = 'Id\t'
    for date in allRefDate:
        if date != allRefDate[-1]:
            tmptxt = 'v'+date+'\t'+'i'+date+'\t'+'s'+date+'\t'
            head = head + tmptxt
        else:
            tmptxt = 'v' + date + '\t' + 'i' + date + '\t' + 's' + date + '\n'
            head = head + tmptxt
    f.write(head)

    ds = driver.Open(targetShp, 0)
    layer = ds.GetLayer()
    feature = layer.GetNextFeature()

    while feature:
        id = feature.GetField('Id')
        targetSample = 'ID' + str(id) + '_VIS.shp'
        sam_ds = driver.Open(targetSample, 0)
        sam_layer = sam_ds.GetLayer()
        sam_feature = sam_layer.GetNextFeature()

        Datelist = []
        LClist = []
        Arealist = []

        while sam_feature:
            LCDate = sam_feature.GetField('Name')
            tmpDate = LCDate[2:]
            tmpLC = LCDate[0]
            LCarea = sam_feature.GetField('POLY_AREA')
            Datelist.append(tmpDate)
            LClist.append(tmpLC)
            Arealist.append(LCarea)
            sam_feature.Destroy()
            sam_feature = sam_layer.GetNextFeature()

        sam_ds.Destroy()

        df = pd.DataFrame({
            'monitorDate': Datelist,
            'LCtype': LClist,
            'area': Arealist
        })

        groups = df.groupby(['monitorDate', 'LCtype']).sum()
        groups_df = groups.add_suffix('sum').reset_index()

        lineval = str(id)+'\t'
        for date in allRefDate:
            Varea = None
            Iarea = None
            Sarea = None

            if date != allRefDate[-1]:

                if date not in Datelist:
                    lineval = lineval + '\t\t\t'
                    continue

                tmpV = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'V')]['areasum']
                tmpI = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'I')]['areasum']
                tmpS = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'S')]['areasum']

                if not tmpV.empty:
                    Varea = float(tmpV)
                else:
                    Varea = 0.
                if not tmpI.empty:
                    Iarea = float(tmpI)
                else:
                    Iarea = 0.
                if not tmpS.empty:
                    Sarea = float(tmpS)
                else:
                    Sarea = 0.

                Vpro = float(Varea) * 1000 / float(Varea + Iarea + Sarea)
                Ipro = float(Iarea) * 1000 / float(Varea + Iarea + Sarea)
                Spro = float(Sarea) * 1000 / float(Varea + Iarea + Sarea)

                lineval = lineval + str(Vpro) + '\t' + str(Ipro) + '\t' + str(Spro) + '\t'

            else:
                if date not in Datelist:
                    lineval = lineval + '\t\t\n'
                    continue

                tmpV = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'V')]['areasum']
                tmpI = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'I')]['areasum']
                tmpS = groups_df[(groups_df['monitorDate'] == date) & (groups_df['LCtype'] == 'S')]['areasum']

                if not tmpV.empty:
                    Varea = float(tmpV)
                else:
                    Varea = 0.
                if not tmpI.empty:
                    Iarea = float(tmpI)
                else:
                    Iarea = 0.
                if not tmpS.empty:
                    Sarea = float(tmpS)
                else:
                    Sarea = 0.

                Vpro = float(Varea) * 1000 / float(Varea + Iarea + Sarea)
                Ipro = float(Iarea) * 1000 / float(Varea + Iarea + Sarea)
                Spro = float(Sarea) * 1000 / float(Varea + Iarea + Sarea)

                lineval = lineval + str(Vpro) + '\t' + str(Ipro) + '\t' + str(Spro) + '\n'

        f.write(lineval)
        feature.Destroy()
        feature = layer.GetNextFeature()
    f.flush()
    f.close()
    ds.Destroy()


def extractOrg3X3PxTimeSeries(Unmixfolder, xoffset, yoffset):
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

        data1 = band1.ReadAsArray(xoffset-1, yoffset-1, 3, 3)
        data2 = band2.ReadAsArray(xoffset-1, yoffset-1, 3, 3)
        data3 = band3.ReadAsArray(xoffset-1, yoffset-1, 3, 3)

        v_sum = np.sum(data1)
        i_sum = np.sum(data2)
        s_sum = np.sum(data3)

        if (-1 in data1) or (-1 in data2) or (-1 in data3):
            VSeries.append(-1)
            ISeries.append(-1)
            SSeries.append(-1)

        elif v_sum == 0 and i_sum == 0 and s_sum == 0:
            VSeries.append(-1)
            ISeries.append(-1)
            SSeries.append(-1)

        else:
            vprop = float(v_sum) / float(v_sum + i_sum + s_sum) * 1000
            iprop = float(i_sum) / float(v_sum + i_sum + s_sum) * 1000
            sprop = float(s_sum) / float(v_sum + i_sum + s_sum) * 1000
            VSeries.append(vprop)
            ISeries.append(iprop)
            SSeries.append(sprop)

        dataset = None
        band1 = None
        band2 = None
        band3 = None

    return datelist, VSeries, ISeries, SSeries


def extractVISForUrbanChg(Unmixfolder, UrbanChgShp):
    os.chdir(Unmixfolder)
    outputpath = os.path.dirname(UrbanChgShp)

    VISlist = glob.glob('*')
    datelist = []
    for i in VISlist:
        datelist.append(i[:8])

    mask = r'C:\dissertation\twimage\LandsatScene\ProcExtent.tif'
    Imgdriver = gdal.GetDriverByName('GTiff')
    Imgdriver.Register()
    maskds = gdal.Open(mask, gdalconst.GA_ReadOnly)
    geotransform = maskds.GetGeoTransform()
    if not geotransform is None:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(UrbanChgShp, 0)
    layer = ds.GetLayer()

    feature = layer.GetNextFeature()

    while feature:
        id = feature.GetField('Id')
        geom = feature.GetGeometryRef()
        geo_X = geom.GetX()
        geo_Y = geom.GetY()

        x_offset = int(round((geo_X - origin_x) / pixel_width))
        y_offset = int(round((geo_Y - origin_y) / pixel_height))

        dateindate, Vlist, Ilist, Slist = extractOrg3X3PxTimeSeries(Unmixfolder, x_offset, y_offset)

        filename = outputpath + '\\ID' + str(id) + '.txt'
        F = open(filename, 'w')
        F.write('Date\tVpro\tIpro\tSpro\n')

        for index, date in enumerate(datelist):
            F.write(str(date) + '\t' + str(Vlist[index]) + '\t' + str(Ilist[index]) + '\t' + str(Slist[index]) + '\n')

        F.flush()
        F.close()

        feature.Destroy()
        feature = layer.GetNextFeature()

    ds.Destroy()
    maskds = None


def VISCompareforChg(targetfolder, RMSEFilter=False, FilterThreshold=26.71, Filter=False , filterWindow=22, polydegree=1):
    import timeSeries
    os.chdir(targetfolder)
    sampleList = glob.glob('*ID*')

    ref = r'C:\dissertation\twimage\urbanChg\UrbanChgAssess\RefVISforChange.txt'
    refTable = pd.read_table(ref, sep='\t', header=0, index_col='Id')

    v_res = 0.
    i_res = 0.
    s_res = 0.

    v_res_abs = 0.
    i_res_abs = 0.
    s_res_abs = 0.

    v_res_sq = 0.
    i_res_sq = 0.
    s_res_sq = 0.

    totalSamDates = 0
    for sam in sampleList:
        samples = pd.read_table(sam, sep='\t', header=0)
        ID = sam[2:sam.index('.txt')]

        if RMSEFilter:
            samples = timeSeries.RMSEFilteredTimeSeries(targetfolder, int(sam[2:sam.index('.txt')]), RMSEThreshold=FilterThreshold)

        datelist = timeSeries.dateTxtList2DateList(samples['Date'].tolist())
        samples['Date'] = datelist

        date8list, V8list, I8list, S8list = timeSeries.regluar8dayInterval(datelist,
                                                                samples['Vpro'].tolist(),
                                                                samples['Ipro'].tolist(), samples['Spro'].tolist())

        for index, date in enumerate(date8list):
            date8list[index] = date.strftime('%Y-%m-%d')

        V8list = timeSeries.timeSeriesInterpolation(V8list)
        I8list = timeSeries.timeSeriesInterpolation(I8list)
        S8list = timeSeries.timeSeriesInterpolation(S8list)

        if Filter:
            V8list = timeSeries.SGFilter(V8list, filterWindow, polydegree)
            I8list = timeSeries.SGFilter(I8list, filterWindow, polydegree)
            S8list = timeSeries.SGFilter(S8list, filterWindow, polydegree)

        inter8df = pd.DataFrame({'date': date8list, 'Vpro': V8list.tolist(), 'Ipro': I8list.tolist(), 'Spro': S8list.tolist()})
        inter8df['Vpro'] = inter8df['Vpro']
        inter8df['Ipro'] = inter8df['Ipro']
        inter8df['Spro'] = inter8df['Spro']
        inter8df['date'] = pd.to_datetime(inter8df['date'])

        tmpDF = refTable.loc[int(ID)]
        tmpDF = tmpDF.dropna()

        tmpDF = pd.DataFrame({'ind': list(tmpDF.index),
                              'area': list(tmpDF)})
        tmpDF['LC'] = tmpDF.ind.str[0]
        tmpDF['year'] = tmpDF.ind.str[1:5]
        tmpDF['month'] = tmpDF.ind.str[5:7]
        tmpDF['day'] = tmpDF.ind.str[7:]

        tmpDF['date'] = pd.to_datetime(tmpDF['year']+'-'+tmpDF['month']+'-'+tmpDF['day'])
        uDate = tmpDF.date.unique()

        for d in uDate:
            vref = np.float(tmpDF['area'][(tmpDF['date'] == d) & (tmpDF['LC'] == 'v')])
            iref = np.float(tmpDF['area'][(tmpDF['date'] == d) & (tmpDF['LC'] == 'i')])
            sref = np.float(tmpDF['area'][(tmpDF['date'] == d) & (tmpDF['LC'] == 's')])

            inter8df['timeDif'] = np.abs((inter8df['date'] - d).dt.days)

            minday = np.min(inter8df['timeDif'])

            vpro = inter8df.loc[inter8df['timeDif']==minday, 'Vpro'].iloc[0]
            ipro = inter8df.loc[inter8df['timeDif']==minday, 'Ipro'].iloc[0]
            spro = inter8df.loc[inter8df['timeDif']==minday, 'Spro'].iloc[0]

            totalSamDates += 1

            v_res += (vpro - vref)
            i_res += (ipro - iref)
            s_res += (spro - sref)

            v_res_abs += np.abs(vpro - vref)
            i_res_abs += np.abs(ipro - iref)
            s_res_abs += np.abs(spro - sref)

            v_res_sq += np.square(vpro - vref)
            i_res_sq += np.square(ipro - iref)
            s_res_sq += np.square(spro - sref)

    v_res = (v_res / totalSamDates) / 10
    i_res = (i_res / totalSamDates) / 10
    s_res = (s_res / totalSamDates) / 10

    v_res_abs = (v_res_abs / totalSamDates) / 10
    i_res_abs = (i_res_abs / totalSamDates) / 10
    s_res_abs = (s_res_abs / totalSamDates) / 10

    v_res_sq = np.sqrt((v_res_sq / totalSamDates)) / 10
    i_res_sq = np.sqrt((i_res_sq / totalSamDates)) / 10
    s_res_sq = np.sqrt((s_res_sq / totalSamDates)) / 10

    print('SE V:{0}, I:{1}, S:{2}'.format(v_res, i_res, s_res))
    print('MAE V:{0}, I:{1}, S:{2}'.format(v_res_abs, i_res_abs, s_res_abs))
    print('RMSE V:{0}, I:{1}, S:{2}'.format(v_res_sq, i_res_sq, s_res_sq))


def SingleDateRMSEAnalysis():
    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)
    df['Date'] = df['Date'].astype(str)
    df['year'] = (df['Date'].str[:4]).astype(int)
    df['month'] = (df['Date'].str[4:6]).astype(int)
    df['day'] = (df['Date'].str[6:]).astype(int)

    print(df.groupby(['month'])['RMSE'].mean())
    df['season'] = np.where(((df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5)), 'spring',
                            np.where(((df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8)), 'summer',
                                     np.where(((df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11)), 'fall',
                                              'winter')))
    print(df.groupby(['season'])['VRMSE', 'IRMSE', 'SRMSE'].mean())
    print(df.groupby(['year'])['RMSE'].mean())

    sensorfile = r'C:\dissertation\twimage\LandsatScene\LsatImgDate.txt'
    sensordf = pd.read_table(sensorfile, sep='\t', header=0)
    sensordf['date'] = sensordf['date'].astype(str)
    sensorDatedf = df.set_index('Date').join(sensordf.set_index('date'))
    print(sensorDatedf.groupby(['Sensor'])['VRMSE', 'IRMSE', 'SRMSE'].mean())

    accudf = df.loc[(df['VRMSE'] < 18.63) & (df['RMSE'] < 22.58) & (df['RMSE'] < 24.63) & (df['Cnt'] > 30)]
    #print(len(accudf))


def timeSeriesNoChgRange(Filtered=False, FilterThreshold=26.71, smooth=False, window=23, polyDegree=1):
    import timeSeries
    workspace = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess'
    os.chdir(workspace)

    samplelist = glob.glob('*ID*')

    RMSEfile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    RMSEdf = pd.read_table(RMSEfile, sep='\t', header=0)
    RMSEdf['Date'] = RMSEdf['Date'].astype(str)
    RMSEdf['accuDate'] = np.where((RMSEdf['VRMSE'] < FilterThreshold) &
                              (RMSEdf['IRMSE'] < FilterThreshold) &
                              (RMSEdf['SRMSE'] < FilterThreshold) &
                              (RMSEdf['Cnt'] > 30), True, False)
    accuDate = (RMSEdf.loc[RMSEdf['accuDate']]['Date']).tolist()

    Vrangelist = []
    Irangelist = []
    Srangelist = []

    for index, i in enumerate(samplelist):
        samTable = pd.read_table(i, sep='\t', header=0)
        samTable['Date'] = samTable['Date'].astype(str)
        datelist = timeSeries.dateTxtList2DateList(samTable['Date'].tolist())
        if Filtered:
            samTable['Vpro'] = np.where(samTable['Date'].isin(accuDate), samTable['Vpro'], -1)
            samTable['Ipro'] = np.where(samTable['Date'].isin(accuDate), samTable['Ipro'], -1)
            samTable['Spro'] = np.where(samTable['Date'].isin(accuDate), samTable['Spro'], -1)

        if smooth:
            date8list, V8list, I8list, S8list = timeSeries.regluar8dayInterval(datelist,
                                                                               samTable['Vpro'].tolist(),
                                                                               samTable['Ipro'].tolist(),
                                                                               samTable['Spro'].tolist())
            V8list = timeSeries.SGFilter(V8list, window, polyDegree)
            I8list = timeSeries.SGFilter(I8list, window, polyDegree)
            S8list = timeSeries.SGFilter(S8list, window, polyDegree)

            samTable = pd.DataFrame({'Date': date8list, 'Vpro': V8list, 'Ipro': I8list, 'Spro': S8list})

        tmpdf = samTable.loc[samTable['Vpro'] != -1]
        Vrangelist.append((tmpdf['Vpro'].max() - tmpdf['Vpro'].min())/10)
        Irangelist.append((tmpdf['Ipro'].max() - tmpdf['Ipro'].min())/10)
        Srangelist.append((tmpdf['Spro'].max() - tmpdf['Spro'].min())/10)

    return len(accuDate), max(Vrangelist), max(Irangelist), max(Srangelist)


def timeSeriesChgRange(Filtered=False, FilterThreshold=26.71, smooth=False, window=23, polyDegree=1, urbanSprawlOnly=False):
    import timeSeries
    workspace = r'C:\dissertation\twimage\urbanChg\UrbanChgAssess'
    os.chdir(workspace)

    samplelist = glob.glob('*ID*')

    RMSEfile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    RMSEdf = pd.read_table(RMSEfile, sep='\t', header=0)
    RMSEdf['Date'] = RMSEdf['Date'].astype(str)
    RMSEdf['accuDate'] = np.where((RMSEdf['VRMSE'] < FilterThreshold) &
                              (RMSEdf['IRMSE'] < FilterThreshold) &
                              (RMSEdf['SRMSE'] < FilterThreshold) &
                              (RMSEdf['Cnt'] > 30), True, False)
    accuDate = (RMSEdf.loc[RMSEdf['accuDate']]['Date']).tolist()

    Vrangelist = []
    Irangelist = []
    Srangelist = []

    for index, i in enumerate(samplelist):
        samTable = pd.read_table(i, sep='\t', header=0)
        samTable['Date'] = samTable['Date'].astype(str)
        datelist = timeSeries.dateTxtList2DateList(samTable['Date'].tolist())
        if Filtered:
            samTable['Vpro'] = np.where(samTable['Date'].isin(accuDate), samTable['Vpro'], -1)
            samTable['Ipro'] = np.where(samTable['Date'].isin(accuDate), samTable['Ipro'], -1)
            samTable['Spro'] = np.where(samTable['Date'].isin(accuDate), samTable['Spro'], -1)

        if smooth:
            date8list, V8list, I8list, S8list = timeSeries.regluar8dayInterval(datelist,
                                                                               samTable['Vpro'].tolist(),
                                                                               samTable['Ipro'].tolist(),
                                                                               samTable['Spro'].tolist())
            V8list = timeSeries.SGFilter(V8list, window, polyDegree)
            I8list = timeSeries.SGFilter(I8list, window, polyDegree)
            S8list = timeSeries.SGFilter(S8list, window, polyDegree)

            samTable = pd.DataFrame({'Date': date8list, 'Vpro': V8list, 'Ipro': I8list, 'Spro': S8list})

        tmpdf = samTable.loc[samTable['Vpro'] != -1]
        Vrangelist.append((tmpdf['Vpro'].max() - tmpdf['Vpro'].min())/10)
        Irangelist.append((tmpdf['Ipro'].max() - tmpdf['Ipro'].min())/10)
        Srangelist.append((tmpdf['Spro'].max() - tmpdf['Spro'].min())/10)

    if urbanSprawlOnly:
        Vrangelist = Vrangelist[30:]
        Irangelist = Irangelist[30:]
        Srangelist = Srangelist[30:]

    return len(accuDate), min(Vrangelist), min(Irangelist), min(Srangelist)


def ChgTimeFromImgSeries(targetfolder):
    pass


#folder = r'C:\dissertation\twimage\urbanChg\UrbanChgAssess'
#VISCompareforChg(folder, RMSEFilter=True, FilterThreshold=27, Filter=True, filterWindow=45, polydegree=1)

#SingleDateRMSEAnalysis()

#VISfolder = r'C:\dissertation\twimage\LandsatScene\DoneImg\VISPercent'
#shpfile = r'C:\dissertation\twimage\ImpRegress\UrbanRegressPNT.shp'
#extractVISForUrbanChg(VISfolder, shpfile)

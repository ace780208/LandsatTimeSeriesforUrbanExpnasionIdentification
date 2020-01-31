import numpy as np
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
import os
import glob
from scipy.optimize import minimize


def rasterProperties(targetImg):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    dataset = gdal.Open(targetImg, gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    geotransform = dataset.GetGeoTransform()
    if not geotransform is None:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

    col = dataset.RasterXSize
    row = dataset.RasterYSize

    band = dataset.GetRasterBand(1)
    nodataVal = band.GetNoDataValue()
    data = band.ReadAsArray(0, 0, col, row)
    print(data)
    avg = np.nanmean(data)
    std = np.nanstd(data)
    print("avg=%s" % avg)
    print("std=%s" % std)

    nandata = np.where(data==nodataVal, np.nan, data)
    print(nandata)
    avg = np.nanmean(nandata)
    std = np.nanstd(nandata)
    print("avg=%s"%avg)
    print("std=%s"%std)
    data=None
    band=None
    dataset=None
    return avg, std



def imgPreporcess(infolder):
    # map clear and water pixels for each date of image
    # the input is a Landsat folder for image preprocessing

    os.chdir(infolder)
    foldername = os.path.basename(os.getcwd())

    # the main raster for determining the noise-free pixels is pixel_qa.tif
    px_qa = glob.glob('*pixel_qa.tif')
    # gdal raster driver is used to open the tif file
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    dataset = gdal.Open(px_qa[0], gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    geotransform = dataset.GetGeoTransform()
    if not geotransform is None:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

    # the raster file for the extent (study area) for image processing is loaded
    maskdataset = gdal.Open(r'C:\dissertation\twimage\LandsatScene\ProcExtent.tif', gdalconst.GA_ReadOnly)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotrasform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()

    mask_band = maskdataset.GetRasterBand(1)

    # the UL coordinate of the extent file is loaded
    if not maskgeotrasform is None:
        maskorig_x = maskgeotrasform[0]
        maskorig_y = maskgeotrasform[3]

    # this function produces a new raster for indicating clear, water, and cloud pixel within the extent
    out_array1 = mask_band.ReadAsArray(0, 0, maskcols, maskrows)
    out_array2 = mask_band.ReadAsArray(0, 0, maskcols, maskrows)
    outputdataset_pre = driver.Create('mask_pre.tif', maskcols, maskrows, 1, gdal.GDT_Byte)
    out_band1 = outputdataset_pre.GetRasterBand(1)
    outputdataset_pos = driver.Create('mask_pos.tif', maskcols, maskrows, 1, gdal.GDT_Byte)
    out_band2 = outputdataset_pos.GetRasterBand(1)

    # the starting point for indicating noise-free pixels
    x_offset = int((maskorig_x-origin_x)/pixel_width)
    y_offset = int((maskorig_y-origin_y)/pixel_height)

    # below process for the extent within the image
    px_qa_band = dataset.GetRasterBand(1)
    data = np.array(px_qa_band.ReadAsArray(x_offset, y_offset, maskcols, maskrows))
    if foldername[3] == '8':
        # clear = [322, 386, 834, 898, 1346]
        # water = [324, 388, 836, 900, 1348]
        out_array1 = out_array1 * np.where(((data == 322) | (data == 386) | (data == 834) | (data == 898) | (data == 1346)), 1,
                    np.where(((data == 324) | (data == 388) | (data == 836) | (data == 900) | (data == 1348)), 1, 0))
        out_array2 = np.where(((data == 322) | (data == 386) | (data == 834) | (data == 898) | (data == 1346)), 1,
                    np.where(((data == 324) | (data == 388) | (data == 836) | (data == 900) | (data == 1348)), 2, 0))

    else:
        # clear = [66, 130]
        # water = [68, 132]
        out_array1 = out_array1 * np.where(((data == 66) | (data == 130)), 1, np.where(((data == 68) | (data == 132)), 1, 0))
        out_array2 = np.where(((data == 66) | (data == 130)), 1, np.where(((data == 68) | (data == 132)), 2, 0))

    out_band1.WriteArray(out_array1, 0, 0)
    out_band1.SetNoDataValue(0)
    outputdataset_pre.SetGeoTransform(maskgeotrasform)
    outputdataset_pre.SetProjection(maskprojection)

    out_band2.WriteArray(out_array2, 0, 0)
    out_band2.SetNoDataValue(0)
    outputdataset_pos.SetGeoTransform(maskgeotrasform)
    outputdataset_pos.SetProjection(maskprojection)
    dataset = None
    maskdataset = None
    outputdataset = None
    print('done for scene ' + foldername)


def getEndmemberFromShp(Imgfolder):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    foldername = os.path.basename(Imgfolder)
    df = Imgfolder + '\\' + 'Endmember.shp'
    datasource = driver.Open(df, 0)
    layer = datasource.GetLayer()

    feature = layer.GetNextFeature()
    V_all = []
    I_all = []
    S_all = []
    while feature:
        if foldername[:4] == 'LC08':
            LCType = feature.GetField('Type')
            db = feature.GetField('y17_n443')
            b = feature.GetField('y17_n482')
            g = feature.GetField('y17_n562')
            r = feature.GetField('y17_n655')
            NIR = feature.GetField('y17_n865')
            SWIR1 = feature.GetField('y17_n1610')
            SWIR2 = feature.GetField('y17_n2200')

            if LCType == 2:
                I_all.append([db, b, g, r, NIR, SWIR1, SWIR2])
            elif LCType == 1:
                V_all.append([db, b, g, r, NIR, SWIR1, SWIR2])
            else:
                S_all.append([db, b, g, r, NIR, SWIR1, SWIR2])

        else:
            LCType = feature.GetField('Type')
            b = feature.GetField('y90_n485')
            g = feature.GetField('y90_n560')
            r = feature.GetField('y90_n660')
            NIR = feature.GetField('y90_n830')
            SWIR1 = feature.GetField('y90_n1650')
            SWIR2 = feature.GetField('y90_n2215')

            if LCType == 2:
                I_all.append([b, g, r, NIR, SWIR1, SWIR2])
            elif LCType == 1:
                V_all.append([b, g, r, NIR, SWIR1, SWIR2])
            else:
                S_all.append([b, g, r, NIR, SWIR1, SWIR2])

        feature.Destroy()
        feature = layer.GetNextFeature()

    datasource.Destroy()
    V_all = np.array(V_all)
    I_all = np.array(I_all)
    S_all = np.array(S_all)

    from sklearn.cluster import KMeans
    vegKmean = KMeans(n_clusters=1, random_state=0).fit(V_all)
    ImpKmean = KMeans(n_clusters=3, random_state=0).fit(I_all)
    SoiKmean = KMeans(n_clusters=2, random_state=0).fit(S_all)
    endmember = np.array([vegKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[1],
                          ImpKmean.cluster_centers_[2],
                          SoiKmean.cluster_centers_[0],
                          SoiKmean.cluster_centers_[1]])
    return endmember


def FullContraintUnmix(end, y):
    def loss(x):
        return np.sum(np.square((np.dot(x, end) - y)))
    cons = ({'type': 'eq',
             'fun': lambda x: np.sum(x) - 1.0})

    x0 = np.zeros(end.shape[0])
    res = minimize(loss, x0, method='SLSQP', constraints=cons,
                   bounds=[(0, np.inf) for i in range(end.shape[0])])
    return res.x


def FCLS(tar, end):
    from pysptools import abundance_maps
    return abundance_maps.amaps.FCLS(tar, end)


def stableEndmenber(targetImgfolder, baseImgfolder, used_band, stdthreshold, outputfile=False):
    # the input is a landsat folder for image preprocessing
    os.chdir(baseImgfolder)
    basefoldername = os.path.basename(os.getcwd())

    # the band specification of Landsat 8 is different from Landsat 5 and 7
    used_band_L8 = used_band + 1
    base_img = None
    if basefoldername[:4] == 'LC08':
        if used_band != 7:
            base_img = glob.glob('*sr_band' + str(used_band_L8) + '.tif')
        else:
            base_img = glob.glob('*sr_band' + str(used_band) + '.tif')

    else:
        base_img = glob.glob('*sr_band' + str(used_band) + '.tif')

    # the base img band and mask will be loaded below
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    base_dataset = gdal.Open(base_img[0], gdalconst.GA_ReadOnly)
    base_mask_Dataset = gdal.Open('mask_pre.tif', gdalconst.GA_ReadOnly)

    base_img_geotransform = base_dataset.GetGeoTransform()

    # the starting coordinate for the base img will be loaded
    if not base_img_geotransform is None:
        base_origin_x = base_img_geotransform[0]
        base_origin_y = base_img_geotransform[3]

    baseband = base_dataset.GetRasterBand(1)
    base_mask_band = base_mask_Dataset.GetRasterBand(1)

    # the base VIS map will be loaded below
    baseVIS = glob.glob('*_VIS.tif')
    base_VIS_dataset = gdal.Open(baseVIS[0], gdalconst.GA_ReadOnly)
    base_V_band = base_VIS_dataset.GetRasterBand(1)
    base_I_band = base_VIS_dataset.GetRasterBand(2)
    base_S_band = base_VIS_dataset.GetRasterBand(3)

    # the workspace switches to the target image folder below
    os.chdir(targetImgfolder)
    targetfoldername = os.path.basename(os.getcwd())

    # check whether the target image is from landsat 8, if yes, change the band specification
    img_diff_band = None
    if targetfoldername[:4] == 'LC08':
        if used_band != 7:
            img_diff_band = glob.glob('*sr_band' + str(used_band+1) + '.tif')
        else:
            img_diff_band = glob.glob('*sr_band' + str(used_band) + '.tif')
    else:
    # load the target band of the target img
        img_diff_band = glob.glob('*sr_band' + str(used_band) + '.tif')

    target_dataset = gdal.Open(img_diff_band[0], gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    target_img_geotransform = target_dataset.GetGeoTransform()
    if not target_img_geotransform is None:
        target_origin_x = target_img_geotransform[0]
        target_origin_y = target_img_geotransform[3]
        pixel_width = target_img_geotransform[1]
        pixel_height = target_img_geotransform[5]

    # the raster file for the extent (study area) for image processing is loaded
    target_maskdataset = gdal.Open('mask_pre.tif', gdalconst.GA_ReadOnly)
    maskcols = target_maskdataset.RasterXSize
    maskrows = target_maskdataset.RasterYSize
    maskgeotrasform = target_maskdataset.GetGeoTransform()
    maskprojection = target_maskdataset.GetProjection()

    target_band = target_dataset.GetRasterBand(1)
    target_mask_band = target_maskdataset.GetRasterBand(1)

    # the UL coordinate of the extent file is loaded
    if not maskgeotrasform is None:
        target_maskorig_x = maskgeotrasform[0]
        target_maskorig_y = maskgeotrasform[3]

    # the starting point pixels in the mask of study area
    base_x_offset = int((target_maskorig_x - base_origin_x) / pixel_width)
    base_y_offset = int((target_maskorig_y - base_origin_y) / pixel_height)
    target_x_offset = int((target_maskorig_x - target_origin_x) / pixel_width)
    target_y_offset = int((target_maskorig_y - target_origin_y) / pixel_height)

    # read base image, mask, target image, target mask, and base VIS into array
    base_img_array = baseband.ReadAsArray(base_x_offset, base_y_offset, maskcols, maskrows)
    base_mask_array = base_mask_band.ReadAsArray(0, 0, maskcols, maskrows)
    base_V_array = base_V_band.ReadAsArray(0, 0, maskcols, maskrows)
    base_I_array = base_I_band.ReadAsArray(0, 0, maskcols, maskrows)
    base_S_array = base_S_band.ReadAsArray(0, 0, maskcols, maskrows)

    target_band_array = target_band.ReadAsArray(target_x_offset, target_y_offset, maskcols, maskrows)
    target_mask_array = target_mask_band.ReadAsArray(0, 0, maskcols, maskrows)

    base_img_array = np.where(base_mask_array == 1, base_img_array, np.nan)
    target_band_array = np.where(target_mask_array == 1, target_band_array, np.nan)
    base_mask_array = None
    target_mask_array = None

    out_stable_array = base_img_array - target_band_array

    # this part use gray and song method for identifying stable training pixels
    avg = np.nanmean(out_stable_array)
    std = np.nanstd(out_stable_array)
    upperlimit = avg + std*stdthreshold
    lowerlimit = avg - std*stdthreshold
    
    out_stable_array = np.where(out_stable_array > lowerlimit,
                                np.where(out_stable_array < upperlimit, 1, 0),
                                0)

    base_img_array = None
    target_band_array = None

    base_V_array = np.where(base_V_array == 1000, 1, 0)
    base_I_array = np.where(base_I_array == 1000, 2, 0)
    base_S_array = np.where(base_S_array == 1000, 3, 0)
    final_training_array = (base_V_array + base_I_array + base_S_array) * out_stable_array
    final_training_array = final_training_array.astype(np.int8)

    base_V_array = None
    base_I_array = None
    base_S_array = None

    base_dataset = None
    target_dataset = None
    base_VIS_dataset = None
    base_mask_Dataset = None
    target_maskdataset = None

    std = None
    if stdthreshold == 0.5:
        std = '_sta05std'
    elif stdthreshold == 0.25:
        std = '_sta025std'
    else:
        std = '_sta' + str(stdthreshold) + 'std'

    out_training_path = targetfoldername + '_band' + str(used_band) + std + '_Train.tif'
    output_training_dataset = driver.Create(out_training_path, maskcols, maskrows, 1, gdal.GDT_Byte)
    outputTrainingband = output_training_dataset.GetRasterBand(1)
    outputTrainingband.WriteArray(final_training_array, 0, 0)
    outputTrainingband.SetNoDataValue(0)

    output_training_dataset.SetProjection(maskprojection)
    output_training_dataset.SetGeoTransform(maskgeotrasform)

    outputTrainingband = None
    output_training_dataset = None

    baseband = None
    base_mask_band = None
    base_V_band = None
    base_I_band = None
    base_S_band = None
    target_band = None
    target_mask_band = None

    if outputfile:
        outputdataset = driver.Create(targetfoldername + '_band' + str(used_band) + std + '.tif', maskcols, maskrows, 1, gdal.GDT_Byte)
        out_band = outputdataset.GetRasterBand(1)
        out_stable_array = out_stable_array.astype(np.int8)
        out_band.WriteArray(out_stable_array, 0, 0)
        out_band.SetNoDataValue(0)

        outputdataset.SetProjection(maskprojection)
        outputdataset.SetGeoTransform(maskgeotrasform)

        out_stable_array = None
        out_band = None
        outputdataset = None

    print('Done getting stable training for ' + targetfoldername)


def modifiedStableEndmember(targetImgfolder, baseImgfolder):
    # the input is a Landsat folder for extracting reflectance of all bands
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shpdf = baseImgfolder + '\\' + 'Endmember.shp'
    print(shpdf)
    datasource = driver.Open(shpdf, 0)
    layer = datasource.GetLayer()

    # the workspace switches to the target image folder below
    os.chdir(targetImgfolder)
    targetfoldername = os.path.basename(os.getcwd())

    # get all reflectance bands in the target folder
    RF_bands = glob.glob('*sr_band*')

    b1_dataset = gdal.Open(RF_bands[0], gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    target_img_geotransform = b1_dataset.GetGeoTransform()
    if not target_img_geotransform is None:
        target_origin_x = target_img_geotransform[0]
        target_origin_y = target_img_geotransform[3]
        pixel_width = target_img_geotransform[1]
        pixel_height = target_img_geotransform[5]

    # the raster file for the extent (study area) for image processing is loaded
    target_maskdataset = gdal.Open('mask_pre.tif', gdalconst.GA_ReadOnly)
    maskcols = target_maskdataset.RasterXSize
    maskrows = target_maskdataset.RasterYSize
    maskgeotrasform = target_maskdataset.GetGeoTransform()

    if not maskgeotrasform is None:
        mask_orig_x = maskgeotrasform[0]
        mask_orig_y = maskgeotrasform[3]

    if targetfoldername[:4] == 'LC08':
        b2_dataset = gdal.Open(RF_bands[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(RF_bands[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(RF_bands[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(RF_bands[4], gdalconst.GA_ReadOnly)
        b6_dataset = gdal.Open(RF_bands[5], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(RF_bands[6], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b6 = b6_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

    else:
        b2_dataset = gdal.Open(RF_bands[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(RF_bands[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(RF_bands[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(RF_bands[4], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(RF_bands[5], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

    target_mask_band = target_maskdataset.GetRasterBand(1)

    V_all = []
    I_all = []
    S_all = []
    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geo_X = geom.GetX()
        geo_Y = geom.GetY()
        img_x_offset = int(round((geo_X - target_origin_x) / pixel_width))
        img_y_offset = int(round((geo_Y - target_origin_y) / pixel_height))
        mask_x_offset = int(round((geo_X - mask_orig_x) / pixel_width))
        mask_y_offset = int(round((geo_Y - mask_orig_y) / pixel_height))

        maskarr = target_mask_band.ReadAsArray(mask_x_offset, mask_y_offset, 1, 1)
        if maskarr[0, 0] > 0:
            if targetfoldername[:4] == 'LC08':
                LCType = feature.GetField('Type')
                db = b1.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                b = b2.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                g = b3.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                r = b4.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                NIR = b5.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                SWIR1 = b6.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                SWIR2 = b7.ReadAsArray(img_x_offset, img_y_offset, 1, 1)

                ndata = (db[0, 0] + b[0, 0] + g[0, 0] + r[0, 0] + NIR[0, 0] + SWIR1[0, 0] + SWIR2[0, 0]) / 7
                b1data = db[0, 0] / ndata * 100
                b2data = b[0, 0] / ndata * 100
                b3data = g[0, 0] / ndata * 100
                b4data = r[0, 0] / ndata * 100
                b5data = NIR[0, 0] / ndata * 100
                b6data = SWIR1[0, 0] / ndata * 100
                b7data = SWIR2[0, 0] / ndata * 100

                if LCType == 2:
                    I_all.append([b1data, b2data, b3data, b4data, b5data, b6data, b7data])
                elif LCType == 1:
                    V_all.append([b1data, b2data, b3data, b4data, b5data, b6data, b7data])
                else:
                    S_all.append([b1data, b2data, b3data, b4data, b5data, b6data, b7data])

            else:
                LCType = feature.GetField('Type')
                b = b1.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                g = b2.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                r = b3.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                NIR = b4.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                SWIR1 = b5.ReadAsArray(img_x_offset, img_y_offset, 1, 1)
                SWIR2 = b7.ReadAsArray(img_x_offset, img_y_offset, 1, 1)

                ndata = (b[0, 0] + g[0, 0] + r[0, 0] + NIR[0, 0] + SWIR1[0, 0] + SWIR2[0, 0]) / 6
                b1data = b[0, 0] / ndata * 100
                b2data = g[0, 0] / ndata * 100
                b3data = r[0, 0] / ndata * 100
                b4data = NIR[0, 0] / ndata * 100
                b5data = SWIR1[0, 0] / ndata * 100
                b7data = SWIR2[0, 0] / ndata * 100

                if LCType == 2:
                    I_all.append([b1data, b2data, b3data, b4data, b5data, b7data])
                elif LCType == 1:
                    V_all.append([b1data, b2data, b3data, b4data, b5data, b7data])
                else:
                    S_all.append([b1data, b2data, b3data, b4data, b5data, b7data])

        feature.Destroy()
        feature = layer.GetNextFeature()

    datasource.Destroy()
    V_all = np.array(V_all)
    I_all = np.array(I_all)
    S_all = np.array(S_all)

    if (I_all.shape[0] < 60):
        print("limit Imp endmembers %s"%I_all.shape[0])
    if (S_all.shape[0] < 50):
        print("limit Imp endmembers %s"%S_all.shape[0])

    from sklearn.cluster import KMeans
    vegKmean = KMeans(n_clusters=1, random_state=0).fit(V_all)
    ImpKmean = KMeans(n_clusters=3, random_state=0).fit(I_all)
    SoiKmean = KMeans(n_clusters=2, random_state=0).fit(S_all)
    endmember = np.array([vegKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[1],
                          ImpKmean.cluster_centers_[2],
                          SoiKmean.cluster_centers_[0],
                          SoiKmean.cluster_centers_[1]])
    return endmember


def getEndmemberFromRaster(targetFolder):

    os.chdir(targetFolder)
    foldername = os.path.basename(os.getcwd())

    Lsatband = glob.glob('*sr_band*')

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    b1_dataset = gdal.Open(Lsatband[0], gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    geotransform = b1_dataset.GetGeoTransform()
    if not geotransform is None:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

    # the raster file for the extent (study area) for image processing is loaded
    Trainband = glob.glob('*_Train*')

    # !!! start editing from here
    Traindataset = gdal.Open(Trainband[0], gdalconst.GA_ReadOnly)
    Traincols = Traindataset.RasterXSize
    Trainrows = Traindataset.RasterYSize
    Traingeotrasform = Traindataset.GetGeoTransform()
    Trainprojection = Traindataset.GetProjection()

    # the UL coordinate of the extent file is loaded
    if not Traingeotrasform is None:
        TrainOrig_x = Traingeotrasform[0]
        TrainOrig_y = Traingeotrasform[3]

    # the starting point for indicating noise-free pixels
    x_offset = int((TrainOrig_x - origin_x) / pixel_width)
    y_offset = int((TrainOrig_y - origin_y) / pixel_height)
    Trainband = Traindataset.GetRasterBand(1)
    TrainData = Trainband.ReadAsArray(0, 0, Traincols, Trainrows)

    target = None

    V_all = None
    I_all = None
    S_all = None
    if foldername[:4] == 'LC08':
        b2_dataset = gdal.Open(Lsatband[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(Lsatband[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(Lsatband[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(Lsatband[4], gdalconst.GA_ReadOnly)
        b6_dataset = gdal.Open(Lsatband[5], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(Lsatband[6], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b6 = b6_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

        b1data = b1.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b2data = b2.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b3data = b3.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b4data = b4.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b5data = b5.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b6data = b6.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b7data = b7.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        print('Done loading all bands')

        ndata = (b1data + b2data + b3data + b4data + b5data + b6data + b7data) / 7

        b1data = ((b1data) / ndata * 100).flatten()
        b2data = ((b2data) / ndata * 100).flatten()
        b3data = ((b3data) / ndata * 100).flatten()
        b4data = ((b4data) / ndata * 100).flatten()
        b5data = ((b5data) / ndata * 100).flatten()
        b6data = ((b6data) / ndata * 100).flatten()
        b7data = ((b7data) / ndata * 100).flatten()
        print('Done normalizing all bands')

        for i in range(1, 4):
            index = np.argwhere((TrainData.flatten())!= i)
            b1data_temp = np.delete(b1data, index)
            b2data_temp = np.delete(b2data, index)
            b3data_temp = np.delete(b3data, index)
            b4data_temp = np.delete(b4data, index)
            b5data_temp = np.delete(b5data, index)
            b6data_temp = np.delete(b6data, index)
            b7data_temp = np.delete(b7data, index)
            target = (np.vstack((b1data_temp, b2data_temp, b3data_temp,
                                 b4data_temp, b5data_temp, b6data_temp, b7data_temp))).T
            if i == 1:
                V_all = target
            elif i == 2:
                I_all = target
            else:
                S_all = target
            target = None

    else:
        b2_dataset = gdal.Open(Lsatband[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(Lsatband[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(Lsatband[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(Lsatband[4], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(Lsatband[5], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

        b1data = b1.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b2data = b2.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b3data = b3.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b4data = b4.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b5data = b5.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        b7data = b7.ReadAsArray(x_offset, y_offset, Traincols, Trainrows).astype(float)
        print('Done loading all bands')

        ndata = (b1data + b2data + b3data + b4data + b5data + b7data)/6
        b1data = (b1data / ndata*100).flatten()
        b2data = (b2data / ndata*100).flatten()
        b3data = (b3data / ndata*100).flatten()
        b4data = (b4data / ndata*100).flatten()
        b5data = (b5data / ndata*100).flatten()
        b7data = (b7data / ndata*100).flatten()

        print('Done normalizing all bands')

        for i in range(1, 4):
            index = np.argwhere((TrainData.flatten()) != i)
            b1data_temp = np.delete(b1data, index)
            b2data_temp = np.delete(b2data, index)
            b3data_temp = np.delete(b3data, index)
            b4data_temp = np.delete(b4data, index)
            b5data_temp = np.delete(b5data, index)
            b7data_temp = np.delete(b7data, index)
            target = (np.vstack((b1data_temp, b2data_temp, b3data_temp,
                                 b4data_temp, b5data_temp, b7data_temp))).T
            if i == 1:
                V_all = target
            elif i == 2:
                I_all = target
            else:
                S_all = target
            target = None

    from sklearn.cluster import KMeans
    vegKmean = KMeans(n_clusters=1, random_state=0).fit(V_all)
    ImpKmean = KMeans(n_clusters=3, random_state=0).fit(I_all)
    SoiKmean = KMeans(n_clusters=2, random_state=0).fit(S_all)
    endmember = np.array([vegKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[0],
                          ImpKmean.cluster_centers_[1],
                          ImpKmean.cluster_centers_[2],
                          SoiKmean.cluster_centers_[0],
                          SoiKmean.cluster_centers_[1]])
    return endmember


def UnmixImg(Imgfolder, basefolder):
    # this function will unmix image pixels into V-I-S land cover
    # check if there is shapefile for endmembers in the image folder
    import glob
    os.chdir(Imgfolder)
    infolder = Imgfolder + "\\"
    foldername = os.path.basename(os.getcwd())

    endmember = None
    if 'Endmember.shp' in os.listdir(infolder):
        endmember = getEndmemberFromShp(Imgfolder)
    else:
        endmember = modifiedStableEndmember(Imgfolder, basefolder)

    print(endmember)
    print('Done loading endmembers!')

    Lsatband = glob.glob('*sr_band*')

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    b1_dataset = gdal.Open(Lsatband[0], gdalconst.GA_ReadOnly)
    # the UL coordinate is loaded along with the size of pixels
    geotransform = b1_dataset.GetGeoTransform()
    if not geotransform is None:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

    # the raster file for the extent (study area) for image processing is loaded
    maskdataset = gdal.Open('mask_pre.tif', gdalconst.GA_ReadOnly)
    maskcols = maskdataset.RasterXSize
    maskrows = maskdataset.RasterYSize
    maskgeotrasform = maskdataset.GetGeoTransform()
    maskprojection = maskdataset.GetProjection()

    # the UL coordinate of the extent file is loaded
    if not maskgeotrasform is None:
        maskorig_x = maskgeotrasform[0]
        maskorig_y = maskgeotrasform[3]

    # this function produces a new raster for store V-I-S proportion
    maskband = maskdataset.GetRasterBand(1)
    mask_array = maskband.ReadAsArray(0, 0, maskcols, maskrows)
    out_array_V = np.zeros((maskrows, maskcols)).astype(np.float32)
    out_array_I = np.zeros((maskrows, maskcols)).astype(np.float32)
    out_array_S = np.zeros((maskrows, maskcols)).astype(np.float32)


    outputdataset = driver.Create(foldername +'_VIS.tif', maskcols, maskrows, 3, gdal.GDT_Int16)
    out_band1 = outputdataset.GetRasterBand(1)
    out_band2 = outputdataset.GetRasterBand(2)
    out_band3 = outputdataset.GetRasterBand(3)

    # the starting point for indicating noise-free pixels
    x_offset = int((maskorig_x - origin_x) / pixel_width)
    y_offset = int((maskorig_y - origin_y) / pixel_height)

    target = None
    if foldername[:4] == 'LC08':
        b2_dataset = gdal.Open(Lsatband[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(Lsatband[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(Lsatband[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(Lsatband[4], gdalconst.GA_ReadOnly)
        b6_dataset = gdal.Open(Lsatband[5], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(Lsatband[6], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b6 = b6_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

        b1data = b1.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b2data = b2.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b3data = b3.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b4data = b4.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b5data = b5.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b6data = b6.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b7data = b7.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        print('Done loading all bands')

        ndata = (b1data + b2data + b3data + b4data + b5data + b6data + b7data) / 7
        b1data = (b1data / ndata * 100).flatten()
        b2data = (b2data / ndata * 100).flatten()
        b3data = (b3data / ndata * 100).flatten()
        b4data = (b4data / ndata * 100).flatten()
        b5data = (b5data / ndata * 100).flatten()
        b6data = (b6data / ndata * 100).flatten()
        b7data = (b7data / ndata * 100).flatten()
        print('Done normalizing all bands')

        target = (np.vstack((b1data, b2data, b3data, b4data, b5data, b6data, b7data))).T

    else:
        b2_dataset = gdal.Open(Lsatband[1], gdalconst.GA_ReadOnly)
        b3_dataset = gdal.Open(Lsatband[2], gdalconst.GA_ReadOnly)
        b4_dataset = gdal.Open(Lsatband[3], gdalconst.GA_ReadOnly)
        b5_dataset = gdal.Open(Lsatband[4], gdalconst.GA_ReadOnly)
        b7_dataset = gdal.Open(Lsatband[5], gdalconst.GA_ReadOnly)

        b1 = b1_dataset.GetRasterBand(1)
        b2 = b2_dataset.GetRasterBand(1)
        b3 = b3_dataset.GetRasterBand(1)
        b4 = b4_dataset.GetRasterBand(1)
        b5 = b5_dataset.GetRasterBand(1)
        b7 = b7_dataset.GetRasterBand(1)

        b1data = b1.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b2data = b2.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b3data = b3.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b4data = b4.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b5data = b5.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        b7data = b7.ReadAsArray(x_offset, y_offset, maskcols, maskrows).astype(float)
        print('Done loading all bands')

        ndata = (b1data + b2data + b3data + b4data + b5data + b7data)/6
        b1data = ((b1data) / ndata*100).flatten()
        b2data = ((b2data) / ndata*100).flatten()
        b3data = ((b3data) / ndata*100).flatten()
        b4data = ((b4data) / ndata*100).flatten()
        b5data = ((b5data) / ndata*100).flatten()
        b7data = ((b7data) / ndata*100).flatten()

        print('Done normalizing all bands')
        target = (np.vstack((b1data, b2data, b3data, b4data, b5data, b7data))).T

    print("start SMA modeling...")

    rowpix = np.zeros((b1data.shape[0], endmember.shape[0]))
    for index, i in enumerate(mask_array.flatten()):
        if i < 0:
            continue
        else:
            rowpix[index] = FullContraintUnmix(endmember, target[index])

        if (index % 1000) == 0:
            print('pix %s' % index)
            #print(rowpix[index])

    print('Starting wirting bands...')
    out_array_V = np.where(mask_array > 0, (rowpix[:, 0]).reshape((maskrows, maskcols)), -1)
    out_array_I = np.where(mask_array > 0, (rowpix[:, 1] + rowpix[:, 2] + rowpix[:, 3]).reshape((maskrows, maskcols)), -1)
    out_array_S = np.where(mask_array > 0, (rowpix[:, 4] + rowpix[:, 5]).reshape((maskrows, maskcols)), -1)

    out_array_V = (out_array_V * 1000).astype(int)
    out_array_I = (out_array_I * 1000).astype(int)
    out_array_S = (out_array_S * 1000).astype(int)

    mask = np.where(out_array_V > 1000, -1, np.where(out_array_V < 0, -1,
            np.where(out_array_I > 1000, -1, np.where(out_array_I < 0, -1,
            np.where(out_array_S > 1000, -1, np.where(out_array_S < 0, -1, 0))))))

    out_array_V = (np.where(mask == -1, -1, out_array_V)).astype(int)
    out_array_I = (np.where(mask == -1, -1, out_array_I)).astype(int)
    out_array_S = (np.where(mask == -1, -1, out_array_S)).astype(int)

    out_band1.WriteArray(out_array_V, 0, 0)
    out_band2.WriteArray(out_array_I, 0, 0)
    out_band3.WriteArray(out_array_S, 0, 0)

    out_band1.SetNoDataValue(-1)
    out_band2.SetNoDataValue(-1)
    out_band3.SetNoDataValue(-1)

    b1 = None
    b2 = None
    b3 = None
    b4 = None
    b5 = None
    b7 = None
    maskband = None

    out_band1 = None
    out_band2 = None
    out_band3 = None

    outputdataset.SetProjection(maskprojection)
    outputdataset.SetGeoTransform(maskgeotrasform)

    outputdataset = None
    b1_dataset = None
    b2_dataset = None
    b3_dataset = None
    b4_dataset = None
    b5_dataset = None
    b7_dataset = None
    maskdataset = None

    if foldername[:4] == 'LC08':
        b6 = None
        b6_dataset = None

    print('Done for ' + foldername)


def kmeanElbow(X):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']

    # k means determine k
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def multitest(targetfolder, basefolder):
    print(targetfolder)


def f(x):
    return x*x


def assignKMeanCenterToShp(shpF):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shpF, 0)
    layer = datasource.GetLayer()

    feature = layer.GetNextFeature()
    V_all = []
    I_all = []
    S_all = []
    while feature:
        Id = feature.GetField('Id')
        LCType = feature.GetField('Type')
        b = feature.GetField('n485')
        g = feature.GetField('n560')
        r = feature.GetField('n660')
        NIR = feature.GetField('n830')
        SWIR1 = feature.GetField('n1650')
        SWIR2 = feature.GetField('n2215')

        if LCType == 2:
            I_all.append([Id, b, g, r, NIR, SWIR1, SWIR2])
        elif LCType == 1:
            V_all.append([Id, b, g, r, NIR, SWIR1, SWIR2])
        else:
            S_all.append([Id, b, g, r, NIR, SWIR1, SWIR2])

        feature.Destroy()
        feature = layer.GetNextFeature()

    datasource.Destroy()
    V_all_arr = np.array(V_all)
    I_all_arr = np.array(I_all)
    S_all_arr = np.array(S_all)

    V_arr_fea = V_all_arr[:, 1:]
    I_arr_fea = I_all_arr[:, 1:]
    S_arr_fea = S_all_arr[:, 1:]

    from sklearn.cluster import KMeans
    vegKmean = KMeans(n_clusters=1, random_state=0).fit(V_arr_fea)
    ImpKmean = KMeans(n_clusters=3, random_state=0).fit(I_arr_fea)
    SoiKmean = KMeans(n_clusters=2, random_state=0).fit(S_arr_fea)

    print(V_all_arr[:,0])
    V_lab = np.hstack((V_all_arr[:,0].reshape((200,1)), vegKmean.labels_.reshape((200, 1))))
    I_lab = np.hstack((I_all_arr[:,0].reshape((200,1)), ImpKmean.labels_.reshape((200, 1))))
    S_lab = np.hstack((S_all_arr[:,0].reshape((200,1)), SoiKmean.labels_.reshape((200, 1))))
    allLabel = np.vstack((V_lab, I_lab, S_lab))
    outTxtF = r'C:\dissertation\twimage\baseImg2\IdCluster.csv'
    np.savetxt(outTxtF, allLabel, delimiter=',', header='OID,Cluster', fmt='%i')

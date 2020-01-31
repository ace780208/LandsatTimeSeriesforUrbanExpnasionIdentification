from multiprocessing import Process, managers
import time
import SMA
import os
import ImgTimeSeries
import numpy as np
import pandas as pd


if __name__ == '__main__':
    num_process = 1  # modify here after identify the number of processes can be used
    #baseF = r'C:\dissertation\twimage\LandsatScene\DoneImg\LC081170432017122301T1-SC20180629194242'
    #subfolderDir = r'C:\dissertation\twimage\LandsatScene\testImg'
    #folderlist = list(x[0] for x in os.walk(subfolderDir))
    #folderlist = folderlist[1:]

    sourcefile = r'C:\dissertation\twimage\VISnoChg\SingleDateAssess\RMSESingleDate.txt'
    df = pd.read_table(sourcefile, sep='\t', header=0)

    df['Date'] = df['Date'].astype(str)
    df['CDate'] = pd.to_datetime(df['Date'])
    df['ODeYr'] = pd.DatetimeIndex(df['CDate']).year + (pd.DatetimeIndex(df['CDate']).dayofyear / 366)
    df['ddays'] = df['CDate'] - pd.to_datetime('1987/4/9')
    ddays_1D = (df['ddays'].dt.days).values
    ddays_2D = (ddays_1D).reshape((len(df), 1))
    df['accuDate'] = np.where((df['VRMSE'] < 27) &
                              (df['IRMSE'] < 27) &
                              (df['SRMSE'] < 27) &
                              (df['Cnt'] > 30), True, False)

    rng8D = pd.date_range('1987/4/9', periods=1403, freq='8D')
    D8Arr1D = ((rng8D - pd.to_datetime('1987/4/9')).days).values
    f10yr = rng8D < pd.to_datetime('1993-1-1')
    l10yr = (rng8D >= pd.to_datetime('2013-1-1'))
    ODeYr = df['ODeYr'].values
    D8DeciYr = np.array((rng8D.year + rng8D.dayofyear / 366).tolist())
    D8Arr2D = D8Arr1D.reshape((len(D8DeciYr), 1))

    manager = managers.SyncManager()
    manager.start()
    win = 45
    polyDeg = 1
    rowlist = (np.arange(3870, 3872).reshape((2, 1))).tolist()
    start = time.time()
    processes = []
    for durow in rowlist:
        for R in durow:
            p = Process(target=ImgTimeSeries.UrbanChgTypeAndTimeRow2, args=(df, f10yr, l10yr,
                                                                            ODeYr, D8DeciYr, ddays_1D, ddays_2D,
                                                                            D8Arr1D, D8Arr2D, win, polyDeg, R, ))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    end = time.time()
    elapsed_time = end - start

    print("completed in %s (sec)" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))



########################################################################################################
'''
    RMSEV = 27
    win = 45
    polyDeg = 1
    startx = list(range(0, 4698, 1566))
    start = time.time()
    processes = []
    for xstr in startx:
        endx = xstr+1566
        p = Process(target=ImgTimeSeries.UrbanChgTypeAndTime, args=(RMSEV, win, polyDeg, xstr, endx, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end = time.time()
    elapsed_time = end - start

    print("completed in %s (sec)" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

'''


'''
    for i in folderlist:
        SMA.imgPreporcess(i)

    start = time.time()

    chunks = [folderlist[x:x+num_process] for x in range(0, len(folderlist), num_process)]

    for ch in chunks:
        processes = []
        for i in ch:
            p = Process(target=SMA.UnmixImg, args=(i, baseF,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    end = time.time()
    elapsed_time = end - start

    print("completed in %s (sec)" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
'''
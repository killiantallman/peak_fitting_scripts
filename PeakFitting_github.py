"""
#author: Killian Tallman
#created: October 2019
#usage: to process x-ray diffraction data and fit a single selected peak to yield the integrated peak area and other peak metrics (location & fwhm)
#input requirements: needs to be Excel files (.xlsx or .xls) that contain two column data of 2theta angle and intensity
"""
#   improvements:  append fitting contraints (e.g. fitting range) to the end of the out file to be saved
    #add if statement noted below to see whether the background determination needs two steps or just one(or more). need to look at the number of points in the background vs in the peak after the first time and then do it again based off that- seems like if there are a lot of points in the peak then it will be to be preformed again
    #add ability to use Pearson 7 function to fit
    
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import wofz
from scipy.optimize import curve_fit
import numba # for speeding up calculations in def functions 
start = time.time()#to see how long code takes to run

working_directory = "/Users/.../XRD_Files"
out_directory = "/Users/.../XRD_Files"
outfilename = 'XRDpeakfit.xlsx'

lower_lim = 6.8 #8.78 #22.8 #42.1 # 22.8 # 35 # 37.6 , 32
upper_lim = 7.35 #9.5 #24.5 #43.6 # 24.5 # 36.5 #39.6,  37

filelist = [f for f in os.listdir(working_directory) if f.endswith('.xlsx') or f.endswith('.xls')]
#sortedfilelist = sorted(filelist, key=lambda x: int(x.split('-')[2].split('_')[0]))
sortedfilelist = filelist
headers = ['2theta','int']

def read_data(df): #READ IN DATA
    df = pd.read_excel(os.path.join(working_directory, df),sep='\s+', names=headers)
    return df

def select_peak(df):   
    df_select_peak = df.loc[(df_data['2theta'] > lower_lim) & (df_data['2theta'] < upper_lim)]
    return df_select_peak

def initial_params(x,y): #GUESS PARAMETERS
    cen_guess = x[y==max(y)][0] #np.mean(x)
    amp_guess = (max(y)-min(y))/2
    sigma_guess = 0.05
    gamma_guess = 0.05 # 0.07    
    initial_params = [cen_guess, sigma_guess, gamma_guess, amp_guess]
    return initial_params

def eval_func(func,x,params):
    out = np.fromiter((func(x,*params) for x in x), np.float)
    return out

def background(x, a, b, c,): #second order polynomial a, b ,c 
#    c = 5500
    return a * x**2 + x * b + c
    
def voigt( x, sig, gam):
#    gam = 0
    z = ( x + 1j * gam ) / (sig * np.sqrt( 2. ))
    v = wofz( z ) #Feddeeva
    out = np.real( v ) / sig / np.sqrt(2 * np.pi)
    return out

def voigtfitfunc( x, cen, sig, gam, amp): #rename? func for integration
    return voigt(x - cen, sig, gam) * amp  #x-cen?? why in voigt and need in BG?

def peak_width(sigma, gamma): #solving for Voigt fwhm from G and L parts
    #https://en.wikipedia.org/wiki/Voigt_profile
    f_G = 2 * sigma * np.sqrt(2 * np.log(2)) # gaussian fwhm
    f_L = 2 * gamma  # gaussian fwhm
    peak_width = ( 0.5346 * f_L ) + np.sqrt(( 0.2166 * f_L**2 ) + f_G**2 )
    return float(peak_width)

out_data = []

for file in sortedfilelist:
    df_data = read_data(file) #reading in file
    df_peak = select_peak(df_data) #selected range
    
    x = df_peak['2theta'].values
    y = df_peak['int'].values #- min(df_peak['int'])
    
    guess_params = initial_params(x,y)
    
    #BACJGOUND SELECTION (twice) sometimes twice not needed
    y_mean = np.sum(y)/(y.shape[0])
    x_BG = x[y<y_mean]
    y_BG = y[y<y_mean]
    
    #improvement here if statemnemnt
    
    y_mean = np.sum(y_BG)/(y_BG.shape[0])
    x_BG = x_BG[y_BG<y_mean]
    y_BG = y_BG[y_BG<y_mean]
     
    #BACKGROUND FITTING
    BGfit, error = curve_fit(background, x_BG, y_BG)
       
    ybg = eval_func(background,x,BGfit) #calculate background
    
    y_noBG =  y - ybg  #subtracting background
    
    #FITTING PEAK with background subtracted data
    solv, error = curve_fit(voigtfitfunc, x, y_noBG, p0=guess_params)
    
    #PLOTTING making data from optimized fit parameters to plot and use       
    y_guess = eval_func(voigtfitfunc,x,guess_params)
    yv = eval_func(voigtfitfunc,x,solv)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].plot(x, y, 'ko', label = 'data')
    axes[0].plot(x, ybg,'g-',label = 'background')
    axes[0].legend(loc='best')
    axes[0].set_ylim([-min(y),max(y)*1.1])
    
    axes[1].plot(x, y_noBG, 'ko', label = 'data')
    axes[1].plot(x, y_guess,'b-',label = 'initial guess')
    axes[1].legend(loc='best')
    axes[1].set_ylim([-min(y),max(y)*1.1])
    
    axes[2].plot(x, y_noBG, 'ko', label = 'data')
    axes[2].plot(x, yv,'r-',label = 'bestfit')
    axes[2].legend(loc='best')
    axes[2].set_ylim([-min(y),max(y)*1.1])
    plt.show()

    #PRINTING OUT FIT RESULTS
    print('\nFile:',file.split('.')[0])
    residuals = y_noBG - voigtfitfunc(x, *solv)
    sumsq_resid = np.sum(residuals**2) #residual sum of squares
    sumsq_total = np.sum((y_noBG - np.mean(y_noBG))**2) #total sum of sqrs
    r_squared = 1 - (sumsq_resid / sumsq_total)
    p =  len(solv) - 1
    adj_r_squared = r_squared - (1 - r_squared) * p / (len(x)-p-1)
    print('\nAdj R-squared:', str(adj_r_squared)[0:8]) #R-squared (COD)
    print('\nBackground fit parameters:',BGfit)
    print('\nPeak fit parameters:',solv)
 
    #AREA OF THE CURVE      
    area, area_error = quad(voigtfitfunc, x[0], x[-1], args=tuple(solv))
    print('\nPeak area:', str(area)[:-8], ' +/- ', str(area_error)[:-14])
    
    #SAVING PARAMETERS
    peak_loc = solv[0]
    fwhm = peak_width(solv[1],solv[2])
    column_headers = ['file','adj r^2','peak area','area error','peak location','fwhm']
    file_params = [file.split('.')[0], str(adj_r_squared)[0:8], str(area)[:-8],str(area_error)[:-14],str(peak_loc)[:-10], str(fwhm)[:-10]] 
        
    out_data.append(file_params)

#EXPORTING
df_final = pd.DataFrame(np.array(out_data),columns = column_headers) ##Making it a dataframe
writer = pd.ExcelWriter(outfilename, engine='xlsxwriter',options={'strings_to_numbers': True}) #Create a Pandas Excel writer using XlsxWriter as the engine
df_final.to_excel(writer, sheet_name='Sheet1',index=False) #Convert the dataframe to an XlsxWriter Excel object
workbook  = writer.book #Get the xlsxwriter workbook and worksheet objects
worksheet = writer.sheets['Sheet1']
worksheet.set_column('B:CZ',15) #sets column width in excel
for columnnumber, columnname in enumerate(df_final.columns):
    worksheet.write(0,columnnumber,columnname)
writer.save()
os.startfile(outfilename)
print('\n',df_final)
end = time.time()
timediff = str((end - start)/60)
print('\nComplete  Runtime:', timediff[:6], 'minutes')
print('\nCompleted... Opening Excel File...')

#end
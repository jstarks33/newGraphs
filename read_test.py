#!/bin/python

###################################################################################################
### Name: Read Test 
### Developer: Tucker McClanahan edits by Jana Starks
### Purpose: Reads zeTorch Data from a run where all of the run files are in the same directory
### Use: python read_test.py 
### Dependents: data_directory/ --> this is the directory with all of the spectrometer data files
###                                 from a single run
### Notes: 
###################################################################################################

import numpy as np
from scipy import signal as sig
from scipy import stats as stats
from scipy.special import wofz
import sys
import os 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from lmfit import Model, Parameters
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import scipy.odr as spODR
import re 

def main():
    """ Reading and Processing the Run Data
   
    This definition is where the data is parsed and filtered by other definitions. It ends with the
    plotting of the data by another set of definitions. 
    """
    
    ### Initializing definition of temperatures
    Temps = {} 
    Names = [] 
   
    path = '/Users/janastarks/learnPython/' 
    directories = np.array(['Smaller_Dataset']) 
    filename = 'JanasTest.pdf'
    name = filename.split('.')[0]
    Names.append(name)
    Temps[name] = {}
    pp = PdfPages(filename)
    
    for i in range(np.size(directories)):
        directory_path = path+directories[i]+'/'
        sys.stdout.write(directory_path+"\n")
        sys.stdout.flush()
        title = directories[i]
        all_data = parse_input(directory_path)
        for dat in all_data:
            dat.calibrate_data('./calibration_curve.dat')
        corrected_data = fitting_bb_data(all_data)
        temp_ar, temp_ar_err = boltz_ar(corrected_data, pp, title)
        temp_bb = (np.sum([dat.temp for dat in corrected_data])/np.size(corrected_data))
        temp_bb_err = (np.sqrt(np.sum([dat.temp_err**2 for dat in corrected_data]))/np.size(corrected_data))
        
        Temps[name][directories[i]] = {}
        Temps[name][directories[i]]['Temp_Ar'] = temp_ar
        Temps[name][directories[i]]['Temp_Ar_Err'] = temp_ar_err
        Temps[name][directories[i]]['Temp_BB'] = temp_bb
        Temps[name][directories[i]]['Temp_BB_Err'] = temp_bb_err
        Temps[name][directories[i]]['Temp_H'] = 0.0 
        Temps[name][directories[i]]['Temp_H_Err'] = 0.0 
    
    pp.close()
    f = open('Table_of_Temperatures_'+name+'.txt', 'w')
    f.write("### Table of Temperatures for "+name+"\n")
    f.write("###\n")
    f.write("###Date Argon_Temp Argon_Temp_Err H_Temp H_Temp_Err BB_Temp BB_Temp_Err \n") 
    for direct in list(Temps[name].keys()):
        ar_temp = Temps[name][direct]['Temp_Ar']
        ar_temp_err = Temps[name][direct]['Temp_Ar_Err']
        H_temp = Temps[name][direct]['Temp_H']
        H_temp_err = Temps[name][direct]['Temp_H_Err']
        bb_temp = Temps[name][direct]['Temp_BB']
        bb_temp_err = Temps[name][direct]['Temp_BB_Err']
        f.write("%s %1.4E %1.4E %1.4E %1.4E %1.4E %1.4E \n" % (direct, ar_temp, ar_temp_err, H_temp, H_temp_err, bb_temp, bb_temp_err))

    f.close()
     

def fitting_bb_data(all_data):
    """ Fits the black body portion of the spectrum for each datafile.

    This definition goes through all of the Run classes for a given dataset and filters out the
    peaks to fit a black body curve to the filtered data. The LMFIT fitting routine is used as a wrapper for the SCIPY optmize tools to fit the Planck's Black Body curve. This function feeds in initial guesses for the parameters and returns a best fitted parameters for the curve.  Keyword Arguments:
    all_data -- List of Run classes

    """

    filtered_data = []
    for dat in all_data:
        counts = np.asarray(dat.calibrated_counts)
        lam = np.asarray(dat.calibrated_lam)
        bb = sig.medfilt(counts,kernel_size=81)

        p = Parameters()
         #          (Name   ,        Value,    Vary,    Min,     Max,    Expr)
        p.add_many(('T'     ,        5000.,    True,    None,    None,    None),
                   ('scale'     ,        1E-15,    True,    None,    None,    None),
                   ('shift'     ,        0.0,    False,    None,    None,    None))

        func = Model(pbb)
        result = func.fit(bb, lam=lam, params=p)
        
         
        dat.bb = bb
        dat.bb_err = 0.0
        dat.temp = result.params['T'].value
        dat.temp_err = result.params['T'].stderr
        dat.aic = result.aic
        filtered_data.append(dat) 

    return filtered_data

def pbb(lam, T, scale, shift):
    h = 6.626070040E-34 #Js
    c = 299792458. #m/s
    k = 1.38064852E-23 #J/K
    lamm = lam*1E-9 #m
    return ((2.*h*c**2/(lamm+shift)**5)*(1./(np.exp(h*c/((lamm+shift)*k*T))-1.)))*scale

def boltz_ar(data, pp, plot_title):

    y_point = []
    x_point = []
    y_point_err = []
    x_point_err = []
    for dat in data:
        lam = np.asarray(dat.calibrated_lam)
        raw_counts = dat.calibrated_counts - np.asarray(dat.bb)
        raw_counts_err = np.sqrt(dat.calibrated_counts_err**2 + np.asarray(dat.bb_err)**2)
        counts = raw_counts/0.473 # units of 1/nm 
        counts_err = raw_counts_err/0.473 
         
        peaks_of_interest = np.array([810.035, 762.182, 749.397, 737.032, 705.584, 695.367, 839.292, 601.445, 910.828]) 
        max_peak_height = []
        max_peak_height_err = []
        max_peak_lam = []
        max_peak_lam_err = []
        for poi in peaks_of_interest:
            peak_indexes = [i for i in range(np.size(lam)) if lam[i]>(poi-5.) and
            lam[i]<(poi+5.)]
            peak_lam = lam[peak_indexes[0]:peak_indexes[-1]] 
            peak_counts = counts[peak_indexes[0]:peak_indexes[-1]] 
            peak_counts_err = counts_err[peak_indexes[0]:peak_indexes[-1]] 
            
            p = Parameters()
            p.add_many(('sigma'     ,        5.0,    True,    0.0,    None,    None),
                       ('gamma'     ,       1.0,    True,    0.0,      None,    None),
                       ('amp'     ,        0.1,    True,    0.0,    None,    None),
                       ('lam0'     ,        poi,    True,    0.0,    None,    None))
            func = Model(voigt)
            result = func.fit(peak_counts, lam=peak_lam, params=p)

            pi = [i for i in range(np.size(peak_lam)) if peak_lam[i]>(poi-2.) and
            peak_lam[i]<(poi+2.)]
            
            T = 0.0
            T_err = 0.0
            for i in range(np.size(peak_counts[pi[0]:pi[-1]])-1):
                i+=1
                a = peak_lam[pi[0]:pi[-1]][i-1]
                b = peak_lam[pi[0]:pi[-1]][i]
                fa = peak_counts[pi[0]:pi[-1]][i-1] 
                fb = peak_counts[pi[0]:pi[-1]][i] 
                dfa = peak_counts_err[pi[0]:pi[-1]][i-1] 
                dfb = peak_counts_err[pi[0]:pi[-1]][i] 

                T += (b-a)*(fb+fa)/2.
                T_err += ((b-a)*0.5*np.sqrt(dfb**2+dfa**2))**2
            
            T_err = np.sqrt(T_err)
            
            max_peak_height.append(T)
            max_peak_height_err.append(T_err)  
            max_peak_lam.append(poi)
            max_peak_lam_err.append(0.0)
             
        max_peak_lam = np.asarray(max_peak_lam)        
        max_peak_lam_err = np.asarray(max_peak_lam_err)        
        max_peak_height = np.asarray(max_peak_height)
        max_peak_height_err = np.asarray(max_peak_height_err)
        const_err = 0.0
        g_k = np.array([7.0,     5.,      1.0,     5.,      5.,       3.,     5.,      9.,      3.]) # no units
        g_k_err = const_err*g_k
        A = np.array([ 0.366,    0.274,   0.472,   0.087,   0.0395,  0.067,  0.244,    0.0246,   0.212]) # 1E8 1/s
        A_err = 0.25*A
        e_k = np.array([105463., 106238., 108723., 107290., 107290., 107496., 107290., 122036., 104102.]) # 1/cm
        e_k_err = const_err*e_k
        e_i = np.array([93144.,  93144.,  95400.,  93751.,  93144.,  93144.,  95400.,  105463., 93144.])
        e_i_err = const_err*e_i
        y = np.log(max_peak_height*max_peak_lam/(g_k*A))
        y_err = np.sqrt((max_peak_height_err/max_peak_height)**2+(max_peak_lam_err/max_peak_lam)**2+(g_k_err/g_k)**2+(A_err/A)**2)
        e = e_k
        #e_err = np.sqrt(e_k_err**2+e_i_err**2)
        e_err=e_k_err
        
        indexs = np.array([0, 1, 2, 3, 4, 5, 7, 8])
        y_point = np.concatenate((y_point,np.asarray([y[i] for i in indexs] ))) 
        x_point = np.concatenate((x_point,np.asarray([e[i] for i in indexs] ))) 
        y_point_err = np.concatenate((y_point_err,np.asarray([y_err[i] for i in indexs] ))) 
        x_point_err = np.concatenate((x_point_err,np.asarray([e_err[i] for i in indexs] ))) 
    e = np.asarray(x_point)
    y = np.asarray(y_point)
    e_err = np.asarray(x_point_err)
    y_err = np.asarray(y_point_err)
    
    lin_model = spODR.Model(lin_func)
    data = spODR.RealData(e, y, sy=y_err)
    odr = spODR.ODR(data, lin_model, beta0=[-0.0003, 1.0])
    out = odr.run()
    
    slope = out.beta[0]
    intercept = out.beta[1]
    slope_err = out.sd_beta[0]  
    
    DELTA = np.size(e)*np.sum(e**2)-np.sum(e)**2
    B = (np.size(e)*np.sum(e*y)-np.sum(e)*np.sum(y))/DELTA
    A = (np.sum(e**2)*np.sum(y)-np.sum(e)*np.sum(e*y))/DELTA

    sigma_y = np.sqrt(1./(np.size(e)-2.)*np.sum((y-A-B*e)**2))
    sigma_B = sigma_y*np.sqrt(np.size(e)/DELTA)
    
    boltzmann = 0.69503476
    temp = -1./(slope*boltzmann)
    temp_err = (-1./((slope+slope_err)*boltzmann))-temp
    y_line = slope*np.asarray(e)+intercept
    plt.figure()
    plt.title(plot_title+' Ar Calc Temp = %1.4E +- %1.4E K' % (temp, temp_err))
    plt.xlabel('Delta E (1/cm)')
    plt.ylabel('$\ln((I\lambda)/(g_k*A))$')
    plt.errorbar(e,y, yerr=y_err, xerr=e_err, fmt='*')
    plt.plot(e, y_line, '-')
    plt.savefig(pp, format='pdf')

    return temp, temp_err

def boltz_H(data, pp, plot_title ):

    y_point = []
    x_point = []
    y_point_err = []
    x_point_err = []
    for dat in data:
        lam = np.asarray(dat.calibrated_lam)
        raw_counts = dat.calibrated_counts - np.asarray(dat.bb)
        raw_counts_err = np.sqrt(dat.calibrated_counts_err**2 + np.asarray(dat.bb_err)**2)
        counts = raw_counts/0.473 # units of 1/nm 
        counts_err = raw_counts_err/0.473 
         
        peaks_of_interest = np.array([656., 485., 434.])
        max_peak_height = []
        max_peak_lam = []
        max_peak_height_err = []
        max_peak_lam_err = []
        for poi in peaks_of_interest:
            peak_indexes = [i for i in range(np.size(lam)) if lam[i]>(poi-5.) and
            lam[i]<(poi+5.)]
            peak_lam = lam[peak_indexes[0]:peak_indexes[-1]] 
            peak_counts = counts[peak_indexes[0]:peak_indexes[-1]] 
            peak_counts_err = counts_err[peak_indexes[0]:peak_indexes[-1]] 
            
            p = Parameters()
            p.add_many(('sigma'     ,        5.0,    True,    0.0,    None,    None),
                       ('gamma'     ,       1.0,    True,    0.0,      None,    None),
                       ('amp'     ,        0.1,    True,    0.0,    None,    None),
                       ('lam0'     ,        poi,    True,    0.0,    None,    None))
            func = Model(voigt)
            result = func.fit(peak_counts, lam=peak_lam, params=p)
            pi = [i for i in range(np.size(peak_lam)) if peak_lam[i]>(poi-2.) and
            peak_lam[i]<(poi+2.)]
            T = 0.0
            T_err = 0.0
            for i in range(np.size(peak_counts[pi[0]:pi[-1]])-1):
                i+=1
                a = peak_lam[pi[0]:pi[-1]][i-1]
                b = peak_lam[pi[0]:pi[-1]][i]
                fa = peak_counts[pi[0]:pi[-1]][i-1] 
                fb = peak_counts[pi[0]:pi[-1]][i] 
                dfa = peak_counts_err[pi[0]:pi[-1]][i-1] 
                dfb = peak_counts_err[pi[0]:pi[-1]][i] 

                T += (b-a)*(fb+fa)/2.
                T_err += ((b-a)*0.5*np.sqrt(dfb**2+dfa**2))**2
            
            T_err = np.sqrt(T_err)
            
            max_peak_height.append(T)
            max_peak_height_err.append(T_err)  
            max_peak_lam.append(poi)
            max_peak_lam_err.append(0.0)

        max_peak_lam = np.asarray(max_peak_lam)        
        max_peak_lam_err = np.asarray(max_peak_lam_err)        
        max_peak_height = np.asarray(max_peak_height)
        max_peak_height_err = np.asarray(max_peak_height_err)
        
        const_err = 0.0
        g_k = np.array([18., 32., 50.])
        g_k_err = const_err*g_k
        A = np.array([ 0.441, 0.0841, 0.0253])
        A_err = 0.01*A
        e_k = np.array([97492., 102824., 105291.])
        e_k_err = const_err*e_k
        e_i = np.array([82259., 82259., 82259.])
        e_i_err = const_err*e_i
        e = e_k-e_i
        e_err = np.sqrt(e_k_err**2+e_i_err**2)
        y = np.log(max_peak_height*max_peak_lam/(g_k*A))
        y_err = np.sqrt((max_peak_height_err/max_peak_height)**2+(max_peak_lam_err/max_peak_lam)**2+(g_k_err/g_k)**2+(A_err/A)**2)
        
        y_point = np.concatenate((y_point,y )) 
        y_point_err = np.concatenate((y_point_err,y_err )) 
        x_point = np.concatenate((x_point,e)) 
        x_point_err = np.concatenate((x_point_err,e_err)) 

    e = np.asarray(x_point)
    e_err = np.asarray(x_point_err)
    y = np.asarray(y_point)
    y_err = np.asarray(y_point_err)
    
    lin_model = spODR.Model(lin_func)
    data = spODR.RealData(e, y, sy=y_err)
    odr = spODR.ODR(data, lin_model, beta0=[-0.0003, 1.0])
    out = odr.run()
    
    slope = out.beta[0]
    intercept = out.beta[1]
    slope_err = out.sd_beta[0]  
    
    DELTA = np.size(e)*np.sum(e**2)-np.sum(e)**2
    B = (np.size(e)*np.sum(e*y)-np.sum(e)*np.sum(y))/DELTA
    A = (np.sum(e**2)*np.sum(y)-np.sum(e)*np.sum(e*y))/DELTA

    sigma_y = np.sqrt(1./(np.size(e)-2.)*np.sum((y-A-B*e)**2))
    sigma_B = sigma_y*np.sqrt(np.size(e)/DELTA)
    
    boltzmann = 0.69503476
    temp = -1./(slope*boltzmann)
    temp_err = (-1./((slope+slope_err)*boltzmann))-temp
    y_line = slope*np.asarray(e)+intercept
    
    plt.figure()
    plt.title(plot_title+' H Calc Temp = %1.4E +- %1.4E K' % (temp, temp_err))
    plt.xlabel('Delta E (1/cm)')
    plt.ylabel('$\ln((I\lambda)/(g_k*A))$')
    plt.errorbar(e,y, yerr=y_err, xerr=e_err, fmt='*')
    plt.plot(e, y_line, '-')
    plt.savefig(pp, format='pdf')
    return temp, temp_err

def voigt(lam, sigma, gamma, amp, lam0):
    z = ((lam-lam0)+gamma*1j)/(sigma*np.sqrt(2))
    return np.real(wofz(z))/(sigma*np.sqrt(2.*np.pi))*amp

def lin_func(p, x):
    m, b = p
    return m*x+b

class Run(object):
    """ Class for one run datafile's contents.
    
    This class contains every piece of information for a single data file.

    Attributes:
    self.filename -- String of the filename 
    self.time -- Float of the time data in milliseconds 
    self.name -- String of name within the file
    self.user -- String of the username
    self.spectrometer -- String of the spectrometer name
    self.trigger_mode -- Integer of the triggermode of the spectrometer
    self.integration_time -- Float of the integration time
    self.scans_to_average -- Integer of the scans to average value
    self.electric_dark_correction_enabled -- String
    self.nonlinearity_correction_enabled -- String
    self.boxcar_width -- Integer of the width of the boxcar
    self.xaxis_mode -- String
    self.number_of_pixels -- Integer
    self.wavelengths -- List of floats of the wavelengths in units of nm
    self.counts -- List of floats of the counts 
    
    self.add_data(self,line) -- Definition to add wavelength and count data to their respective
    lists
    self.load_file(self, filename) -- Loads the contents of the file and assigns it to the
    respective attribute 
    """   


    
    def __init__(self):
        self.filename = ''
        self.time = 0.0
        self.name = ''
        self.user = ''
        self.spectrometer = ''
        self.trigger_mode = 0
        self.integration_time = 0.0
        self.scans_to_average = 0
        self.electric_dark_correction_enabled = ''
        self.nonlinearity_correction_enabled = ''
        self.boxcar_width = 0
        self.xaxis_mode = ''
        self.number_of_pixels = 0
        self.wavelengths = []
        self.counts = []
    
    def calibrate_data(self, calibration_file):
        cal_data = np.loadtxt(calibration_file)
        lam = np.asarray(self.wavelengths)
        counts = np.asarray(self.counts)
        
        min_lam = 300.
        max_lam = 915.
         
        indexes = [i for i in range(np.size(cal_data[:,0])) if cal_data[i,0] > min_lam and cal_data[i,0]< max_lam] 
        slashed_cal_data = cal_data[indexes,:]
        indexes = [i for i in range(np.size(lam)) if lam[i] > min_lam and lam[i]< max_lam] 
        slashed_lam = lam[indexes]
        slashed_counts = counts[indexes]

        cal_interp = interp1d(slashed_cal_data[:,0], slashed_cal_data[:,1], bounds_error=False, fill_value=0.0)
        meas_interp = interp1d(slashed_lam, slashed_counts, bounds_error=False, fill_value=0.0)
        
        filtered_data = sig.wiener(meas_interp(slashed_lam))
        T = 0.0
        T_err = 0.0
        for i in range(np.size(filtered_data)-1):
            i+=1
            a = slashed_lam[i-1]
            b = slashed_lam[i]
            fa = filtered_data[i-1]
            fb = filtered_data[i]

            T += (b-a)*(fb+fa)/2.
            T_err += ((b-a)*0.5*np.sqrt(fb+fa))**2
        T_err = np.sqrt(T_err)
        temp_filtered_data_err = np.sqrt(filtered_data)
        frac_C = np.sqrt((temp_filtered_data_err/filtered_data)**2+(T_err/T)**2)
        filtered_data /= T
        filtered_data_err = filtered_data*frac_C
        
        self.calibrated_lam = slashed_lam
        self.calibrated_counts = filtered_data/(cal_interp(slashed_lam)/np.max(cal_interp(slashed_lam)))
        self.calibrated_counts_err = filtered_data_err/(cal_interp(slashed_lam)/np.max(cal_interp(slashed_lam)))
        
        
    def load_file(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            data = f.readlines()
            found_data = 0
            for line in data:
                line = line.rstrip()
                if found_data > 0:
                    self.add_data(line)
                if line.lower().startswith('data'):
                    self.name = line.split()[2]
		    txt = self.name 
                    re1='.*?'	# Non-greedy match on filler
		    re2='\\d+'	# Uninteresting: int
		    re3='.*?'	# Non-greedy match on filler
		    re4='\\d+'	# Uninteresting: int
		    re5='.*?'	# Non-greedy match on filler
		    re6='(\\d+)'	# Integer Number 1
		    re7='(-)'	# Any Single Character 1
		    re8='(\\d+)'	# Integer Number 2
		    re9='(-)'	# Any Single Character 2
		    re10='(\\d+)'	# Integer Number 3
		    re11='(-)'	# Any Single Character 3
		    re12='(\\d+)'	# Integer Number 4

		    rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12,re.IGNORECASE|re.DOTALL)
		    m = rg.search(txt)
		    if m:
			int1=float(m.group(1))*3600.*1000.
			c1=m.group(2)
			int2=float(m.group(3))*60.*1000.
			c2=m.group(4)
			int3=float(m.group(5))*1000.
			c3=m.group(6)
			int4=float(m.group(7))
		    self.time = int1+int2+int3+int4
                if line.lower().startswith('user'):
                    self.user = line.split()[1]
                if line.lower().startswith('spectrometer'):
                    self.spectrometer = line.split()[1]
                if line.lower().startswith('trigger'):
                    self.trigger_mode = int(line.split()[2])
                if line.lower().startswith('integration'):
                    self.integration_time = float(line.split()[3])
                if line.lower().startswith('scans'):
                    self.scans_to_average = int(line.split()[3])
                if line.lower().startswith('electric'):
                    self.electric_dark_correction_enabled = line.split()[4]
                if line.lower().startswith('nonlinearity'):
                    self.nonlinearity_correction_enabled = line.split()[3]
                if line.lower().startswith('boxcar'):
                    self.boxcar_width = int(line.split()[2])
                if line.lower().startswith('xaxis'):
                    self.xaxis_mode = line.split()[2]
                if line.lower().startswith('number'):
                    self.number_of_pixels = int(line.split()[5])
                if 'begin spectral data' in line.lower():
                    found_data = 1
        

     
    def add_data(self,line):
        """ Adding data to the wavelengths and counts lists from a given line"""
        try:
            self.wavelengths.append(float(line.split()[0]))
            self.counts.append(float(line.split()[1]))
        except ValueError:
            print 'Theres an error in the file!'
            print self.filename
            print line




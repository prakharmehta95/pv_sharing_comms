# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:46:26 2020

@authors: Sandro Schopfer, Prakhar Mehta

GENERATING PV PRODUCTION DATA FOR EACH BUILDING
"""
# %%

import scipy.io
import scipy.stats
import pandas as pd
import numpy as np
import os

# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)

parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, 'data')
results_dir = os.path.join(parent_dir, 'results')


deg2rad = np.pi/180.


# function to get random tilt values of PV panels
def TiltRV(N=10000):
    np.random.seed(11)
    return scipy.stats.gompertz.rvs(3e-2, loc=0, scale=12, size=N)

# function to get random orientations for PV panels - like the surface azimuth angle


def OrientRV(N=10000):
    lower = 0
    upper = 360.
    mu = 180.
    sigma = 50.

    np.random.seed(11)
    samples = scipy.stats.truncnorm.rvs(
        (lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=N)

    return samples


def myUnitStep(x):
    '''
    np.sign = 1 if +ve ; -1 if -ve
    returns 1 if +ve, 0 if -ve
    '''
    return (np.sign(x)+1.)/2.


def DirProj(pOrient, pTilt, alt, az):
    '''
    defines the cosine of the angle of incidence
    for the direct beam component of the solar irradiance

    '''
    k1 = 1/np.tan(alt*deg2rad)
    k2 = np.cos((az-pOrient)*deg2rad)
    Rb = np.cos(pTilt*deg2rad)+np.sin(pTilt*deg2rad)*myUnitStep(k1)*k1*myUnitStep(k2)*k2
    b0 = .05
    cosInc = Rb*np.sin(alt*deg2rad)

    # implemented for Zurich because less than 20 degrees meant no PV production
    # because of the mountains blocking sunlight
    # neglected for London
    iAM = np.zeros(np.size(alt))
    Rb[alt < 20] = 0
    iAM[cosInc > 0.0] = 1-b0*(1/cosInc[cosInc > 0.0]-1)  # only positive scalar product relevant
    iAM[iAM < 0.0] = 0

    # cosine of incidence
    return cosInc


def TiltFacDiff(pTilt):
    '''
    for the diffused component of solar irradiance
    equation 5.101
    (isotropic sky model Liu and Jordan 1960)
    '''
    return (1+np.cos(pTilt*deg2rad))/2.


def TiltFacDiffreflect(pTilt):
    '''
    for the reflected component of solar irradiance
    equation 5.100
    '''
    albedo = 0.1
    return albedo*(1-np.cos(pTilt*deg2rad))/2.


class PVmodule(object):
    __slot__ = [
        'VocSTC',  # open circuit voltage at Standard Testing Conditions (STC)
        'IscSTC',  # short circuit current at STC
        'Vmp',  # Max. Power point voltage
        'Imp',  # Max. Power point current
        'aIsc',  # temperature coeff. of ISC
        'aVoc',  # temperature coeff. of VOC
        'Amodule',  # Panel Area
        'nPanel',  # number of panels
        'Tilt',  # tilting angle in deg
        'Orient'  # Orientation in deg
    ]

    def __init__(s, VocSTC, IscSTC, Vmp, Imp, aIsc, aVoc, Amodule, nModule, Orient, Tilt):
        s.VocSTC = VocSTC
        s.IscSTC = IscSTC
        s.Vmpp = Vmp
        s.Impp = Imp
        s.aIsc = aIsc
        s.aVoc = aVoc
        s.Amodule = Amodule
        s.nPanel = nModule
        s.Tilt = Tilt
        s.Orient = Orient

    def IscTranslate(s, G, T):
        '''
        translate STC condition Isc to outdoor condition 
        equation 9.94a
        '''
        return s.IscSTC/(1.+s.aIsc*(25.0-T))*G/1000.0

    def VocTranslate(s, delta, G, T):
        '''
        translate STC condition Voc to outdoor condition 
        equation 9.94b
        G1 = 1000
        '''
        VocT = np.zeros(len(G))
        p = G > 0.0
        VocT[p] = s.VocSTC/(1.+s.aVoc*(25.-T[p]))/(1+delta*np.log(1000.0/G[p]))
        return VocT

    def ITranslate(s, It, G, T):
        return It*s.IscTranslate(G, T)/s.IscSTC

    def VTranslate(s, Vt, delta, G, T):
        return Vt*s.VocTranslate(delta, G, T)/s.VocSTC

    def VmpTranslate(s, delta, G, T):
        return s.Vmpp*s.VocTranslate(delta, G, T)/s.VocSTC

    def ImpTranslate(s, G, T):
        return s.Impp*s.IscTranslate(G, T)/s.IscSTC

    def Ppeak(s): return s.Vmpp*s.Impp

    def Pdc(s, G, T, delta=.085):  # meaning of delta ?
        '''
        p. 300: delta = the coefficient delta appears as a measure of deviation
        from the linear relation between power and irradiance.
        delta = 0.085 is for monocrystalline Si solar cells, 
        comes from Table 9.2
        Ct and Tmodule equations 9.11, 9.12
        Power = Vm*Im - equation 9.94e
        '''
        Ct = (45.-20.)/800.
        Tmodule = T+Ct*G
        return s.VmpTranslate(delta, G, Tmodule)*s.ImpTranslate(G, Tmodule)

    def TotPanelArea(s):
        return s.nPanel*s.Amodule  # %%


def CalcPV(Pload, Inputs, MeteoData, Pdc_ext=None):

    pvField = PVmodule(Inputs['VocSTC'],
                       Inputs['IscSTC'],
                       Inputs['Vmpp'],
                       Inputs['Impp'],
                       Inputs['aIsc']/100.,  # value is read in %
                       Inputs['aVoc']/100.,  # value is read in %
                       Inputs['Amodule'],
                       Inputs['nPV'],
                       Inputs['SolPVOrient'],
                       Inputs['SolPVTilt']
                       )
    n = Pload
    AnnualBalance = {
        'W_PV_DC': np.zeros(n),
        'W_PV_AC': np.zeros(n),
        'PV_Cap_kW': 0
    }

    # total irradiance on an inclined surface = (beam + diffuse + reflected components on horizontal surface) * geometric angles
    # DirProj returns the cosine of the angle of incidence on the PV panel (which is tilted and oriented)
    cosinc = DirProj(
        pvField.Orient,
        pvField.Tilt,
        MeteoData['Alt'],
        MeteoData['Az'])

    GtiltPV = MeteoData['GDir']*cosinc +\
        MeteoData['GhDiff']*TiltFacDiff(pvField.Tilt) +\
        ((MeteoData['GDir']+MeteoData['GhDiff'])*TiltFacDiffreflect(pvField.Tilt))

    PdcMod_ = pvField.Pdc(GtiltPV, MeteoData['Tamb'])  # pdc per module

    Pdc = PdcMod_*Inputs['nPV']/1000.  # kW

    if Pdc_ext is not None:
        Pdc = Pdc_ext

    dt = Inputs['dt']

    AnnualBalance['W_PV_DC'] = Pdc * dt
    AnnualBalance['W_PV_AC'] = Pdc * dt * Inputs['EtaInv']
    AnnualBalance['PV_Cap_kW'] = int(Inputs['Vmpp']*Inputs['Impp']*Inputs['nPV']/1000.0)  # kWp

    return AnnualBalance, cosinc


# %%

# weather data
weather_filename = 'London_Weather_TMY_spa.xlsx'
StationData = pd.read_excel(os.path.join(data_dir, weather_filename))

# demand data
demanddata_filename = 'london_cleaned.pickle'
dem = pd.read_pickle(os.path.join(data_dir, demanddata_filename))


# list of all buildings
flist = dem.columns
flist = flist.drop('ID')

Inputs = {
    'VocSTC': 37.8,
    'IscSTC': 9.11,
    'Vmpp': 30.7,
    'Impp': 8.47,
    'aIsc': -.31,  # %/K
    'aVoc': .06,  # %/K
    'Amodule': 1.63,
    'SolPVOrient': 180.0,  # deg
    'SolPVTilt': 25.0,  # deg
    'nPV': 100/26,  # number of panels, so that capacity = 1kW. 1 panel rated 260.029 W
    'EbatR': 0.01,
    'Nc': 6000,
    'DoD': 0.80,
    'SigmaCL': 0.9,
    'EtaInv': .95,
    'EtaCh': .95,
    'EtaDCh': 0.95,
    'dt': 0.5  # temproal resolution of time series in hours @Alejandro
}

MeteoData = {
    'GDir': StationData.DirNorm.values,
    'GhDiff': StationData.DifHorz.values,
    'Az': StationData.Azimuth.values,
    'Alt': StationData.Altitude.values,
    'Tamb': StationData.Tair.values
}

# random assignment of tilts and orientations
SolTiltRV = TiltRV(N=10000)
SolOrientRV = OrientRV(N=10000)

pvdata = pd.DataFrame(data=None)
sizes = []

tilt = []
orient = []
res_dict = {}
pv = pd.DataFrame(data=None)

for _i in range(len(flist)):

    name = flist[_i]
    Inputs['SolPVTilt'] = SolTiltRV[_i]
    tilt.append(Inputs['SolPVTilt'])
    Inputs['SolPVOrient'] = SolOrientRV[_i]
    orient.append(Inputs['SolPVOrient'])

    datadict, cosinc = CalcPV(len(StationData.Altitude), Inputs, MeteoData)
    pvdata[name] = ""
    pvdata[name] = datadict['W_PV_AC']
    sizes.append(datadict['PV_Cap_kW'])
    a = pvdata[name]
    pv[name] = a


res_dict['Tilts'] = tilt
res_dict['Orientations'] = orient
res_dict['W_PV_AC'] = pvdata
res_dict['Cap_kW'] = sizes

# saving data
np.save(os.path.join(data_dir, 'pv_gen_dict_20200815.npy'), res_dict)
pv.to_pickle(os.path.join(data_dir, 'PV_Gen_UK_20200815.pickle'))

import numpy as np
from scipy.special import lambertw
from scipy.special import sici, exp1




#==============================================================================
#==============================================================================
#==============================================================================
#FUNCTION DEFINITIONS==========================================================
#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
def comp_cdwave(F):
   gamma_em = np.euler_gamma
   return 4*(F*np.sin(2/F) - (F**2)*(np.sin(1/F)**2) - cosint(2/F) + np.log(2/F) - 1 + gamma_em )
#==============================================================================

#==============================================================================
def cosint(x):
   si, ci = sici(x)
   return ci
#==============================================================================


#==============================================================================
 #descriminator function between liquid and ice (i.e., omega defined in the
 #beginning of section 2e in Peters et al. 2022)
def omega(T,T1,T2):
    return ((T - T1)/(T2-T1))*np.heaviside((T - T1)/(T2-T1),1)*np.heaviside((1 - (T - T1)/(T2-T1)),1) + np.heaviside(-(1 - (T - T1)/(T2-T1)),1);
def domega(T,T1,T2):
    return (np.heaviside(T1-T,1) - np.heaviside(T2-T,1))/(T2-T1)
#==============================================================================

#==============================================================================
#FUNCTION THAT CALCULATES THE SATURATION MIXING RATIO
def compute_rsat(T,p,iceflag,T1,T2):
    
    #THIS FUNCTION COMPUTES THE SATURATION MIXING RATIO, USING THE INTEGRATED
    #CLAUSIUS CLAPEYRON EQUATION (eq. 7-12 in Peters et al. 2022).
    #https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1 

    #input arguments
    #T temperature (in K)
    #p pressure (in Pa)
    #iceflag (give mixing ratio with respect to liquid (0), combo liquid and
    #ice (2), or ice (3)
    #T1 warmest mixed-phase temperature
    #T2 coldest mixed-phase temperature
    
    #NOTE: most of my scripts and functions that use this function need
    #saturation mass fraction qs, not saturation mixing ratio rs.  To get
    #qs from rs, use the formula qs = (1 - qt)*rs, where qt is the total
    #water mass fraction

    #CONSTANTS
    Rd=287.04#%dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv
    cp=1005 #specific heat of dry air at constant pressure
    g=9.81 #gravitational acceleration
    xlv=2501000 #reference latent heat of vaporization at the triple point temperature
    xls=2834000 #reference latent heat of sublimation at the triple point temperature
    cpv=1870 #specific heat of water vapor at constant pressure
    cpl=4190 #specific heat of liquid water
    cpi=2106 #specific heat of ice
    ttrip=273.15; #triple point temperature
    eref=611.2 #reference pressure at the triple point temperature

    omeg = omega(T,T1,T2)
    if iceflag==0:
        term1=(cpv-cpl)/Rv
        term2=(xlv-ttrip*(cpv-cpl))/Rv
        esl=np.exp((T-ttrip)*term2/(T*ttrip))*eref*(T/ttrip)**(term1)
        qsat=epsilon*esl/(p-esl)
    elif iceflag==1: #give linear combination of mixing ratio with respect to liquid and ice (eq. 20 in Peters et al. 2022)
        term1=(cpv-cpl)/Rv
        term2=(xlv-ttrip*(cpv-cpl))/Rv
        esl_l=np.exp((T-ttrip)*term2/(T*ttrip))*eref*(T/ttrip)**(term1)
        qsat_l=epsilon*esl_l/(p-esl_l);
        term1=(cpv-cpi)/Rv
        term2=( xls-ttrip*(cpv-cpi))/Rv
        esl_i=np.exp((T-ttrip)*term2/(T*ttrip))*eref*(T/ttrip)**(term1);
        qsat_i=epsilon*esl_i/(p-esl_i)
        qsat=(1-omeg)*qsat_l + (omeg)*qsat_i
    elif iceflag==2: #only give mixing ratio with respect to ice
        term1=(cpv-cpi)/Rv
        term2=( xls-ttrip*(cpv-cpi))/Rv
        esl=np.exp((T-ttrip)*term2/(T*ttrip))*eref*(T/ttrip)**(term1)
        esl = min( esl , p*0.5 )
        qsat=epsilon*esl/(p-esl);
    return qsat
#==============================================================================


#==============================================================================
#LAPSE RATE FOR AN UNSATURATED PARCEL
def drylift(T,qv,T0,qv0,fracent):
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    cp=1005 #specific heat of dry air at constant pressure
    g=9.81 #gravitational acceleration
    cpv=1870 #specific heat of water vapor at constant pressure
    
    cpmv = (1 - qv)*cp + qv*cpv
    B = g*( (T-T0)/T0 + (Rv/Rd - 1)*(qv - qv0) )
    eps = -fracent*(T - T0)
    gamma_d = - (g + B)/cpmv + eps
    return gamma_d
#==============================================================================


#==============================================================================
#LIFTED CONDENSATION LEVEL USING THE ROMPS 2017 FORMULA
def compute_LCL(T,qv,p):
    #CONSTANTS
    Rd=287.04#%dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv
    cp=1005 #specific heat of dry air at constant pressure
    g=9.81 #gravitational acceleration
    xlv=2501000 #reference latent heat of vaporization at the triple point temperature
    xls=2834000 #reference latent heat of sublimation at the triple point temperature
    cpv=1870 #specific heat of water vapor at constant pressure
    cpl=4190 #specific heat of liquid water
    cpi=2106 #specific heat of ice
    ttrip=273.15; #triple point temperature
    eref=611.2 #reference pressure at the triple point temperature
    cv=cp-Rd
    cvv=cpv-Rv


    cpm = (1 - qv)*cp + qv*cpv
    Rm = (1 - qv)*Rd + qv*Rv
    
    a = cpm/Rm + ( cpl - cpv )/Rv
    b = -(xlv - (cpv - cpl)*ttrip)/(Rv*T)
    c = b/a
    
    r_sat = compute_rsat(T,p,0,273.15,253.15)
    q_sat = r_sat/(1 + r_sat)
    RH = qv/q_sat
    arg1 = RH**(1/a)
    arg2 = c*np.exp(1)**c
    arg3 = lambertw(arg1*arg2,k=-1)
    T_LCL = c*T/arg3
    P_LCL = p*(T_LCL/T)**(cpm/Rm)
    Z_LCL = (cpm/g)*(T - T_LCL)
    
    return Z_LCL
#==============================================================================





#==============================================================================
#LIFTED CONDENSATION LEVEL USING NUMERICAL INTEGRATION
def compute_LCL_NUMERICAL(T,qv,p,dz):
    #CONSTANTS
    #NOTE, WE ARE ASSUMING ZERO BUOYANCY (I.E., WELL MIXED PBL)
    Rd=287.04#%dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv
    cp=1005 #specific heat of dry air at constant pressure
    g=9.81 #gravitational acceleration
    xlv=2501000 #reference latent heat of vaporization at the triple point temperature
    xls=2834000 #reference latent heat of sublimation at the triple point temperature
    cpv=1870 #specific heat of water vapor at constant pressure
    cpl=4190 #specific heat of liquid water
    cpi=2106 #specific heat of ice
    ttrip=273.15; #triple point temperature
    eref=611.2 #reference pressure at the triple point temperature
    cv=cp-Rd
    cvv=cpv-Rv

    nfound_LCL = True
    
    zon = 0
    ind_hgt = 0
    Ton = T
    Qon = qv
    Pon = p
    while nfound_LCL:
        ind_hgt = ind_hgt+1
        Ton = Ton + dz*drylift(Ton,Qon,Ton,Qon,0)
        Pon = Pon - dz*(Pon*g)/(Rd*(1 + (Rv/Rd - 1)*Qon )*Ton )
        rsat = compute_rsat(Ton,Pon,0,273.15,253.15)
        qsat = rsat/(1 + rsat)
        if Qon >= qsat:
            nfound_LCL = False
    Z_LCL = ind_hgt*dz
    
    return Z_LCL
#==============================================================================








#==============================================================================
#LAPSE RATE FOR A SATURATED PARCEL
def moislif(T,qv,qvv,qvi,p0,T0,q0,qt,fracent,prate,T1,T2):
    
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv;
    cp=1005 #specific heat of dry air at constant pressure
    g=9.81 #gravitational acceleration
    xlv=2501000 #reference latent heat of vaporization at the triple point temperature
    xls=2834000 #reference latent heat of sublimation at the triple point temperature
    cpv=1870 #specific heat of water vapor at constant pressure
    cpl=4190 #specific heat of liquid water
    cpi=2106 #specific heat of ice
    ttrip=273.15 #triple point temperature
 
    qt=max(qt,0.0)
    qv=max(qv,0.0)
    
    OMEGA = omega(T,T1,T2)
    dOMEGA = domega(T,T1,T2)
    
    
    cpm = (1 - qt)*cp + qv*cpv + (1 - OMEGA)*(qt-qv)*cpl + OMEGA*(qt-qv)*cpi
    Lv = xlv + (T - ttrip)*(cpv - cpl)
    Li = (xls-xlv) + (T - ttrip)*(cpl - cpi);
    Rm0 = (1 - q0)*Rd + q0*Rv
    

    T_rho=T*(1 - qt + qv/epsilon)
    T_rho0=T0*( 1 - q0 + q0/epsilon )
    B = g*(T_rho - T_rho0)/(T_rho0)
    
    Qvsl = qvv/( epsilon - epsilon*qt + qv)
    Qvsi = qvi/( epsilon - epsilon*qt + qv)
    Q_M = (1 - OMEGA)*qvv/(1 - Qvsl) + OMEGA*qvi/(1 - Qvsi)
    L_M = Lv*(1 - OMEGA)*qvv/(1 - Qvsl) + (Lv + Li)*OMEGA*qvi/(1 - Qvsi)

    
    eps_T = -fracent*(T - T0)
    eps_qv = -fracent*(qv - q0)
    eps_qt = -fracent*(qt - q0)-prate*(qt-qv)
    term1 = -B
    
    term2 = - Q_M*(Lv + Li*OMEGA)*g/(Rm0*T0)
    
    term3 = -g
    term4 = (cpm - Li*(qt - qv)*dOMEGA)*eps_T
    term5 = (Lv + Li*OMEGA)*(eps_qv + (qv/(1-qt))*eps_qt)

    term6 = cpm
    term7 = -Li*(qt - qv)*dOMEGA
    term8 = (Lv + Li*OMEGA)*(-dOMEGA*(qvv - qvi) + (1/(Rv*(T**2)))*(L_M))
    gamma_m =( term1 + term2 + term3 + term4 + term5)/(term6 + term7 + term8)
    return gamma_m
#==============================================================================


#==============================================================================
#FUNCTION THAT LIFTS A PARCEL
def lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2):
    #[T_lif,Qv_lif,Qt_lif,B_lif]

    #this function computes lifted parcel properties using the unsaturated
    #and saturated lapse rate formulas from (Peters et al. 2022)
    #https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1 
    
    #input arguments
    #T0: sounding profile of temperature (in K)
    #p0: sounding profile of pressure (in Pa)
    #q0: sounding profile of water vapor mass fraction (in kg/kg)
    #start_loc: index of the parcel starting location (set to 1 for the
    #lowest: level in the sounding)
    #fracent: fractional entrainment rate (in m^-1)
    
    #output arguments
    #T_lif: lifted parcel temperature
    #Qv_lif: lifted parcel water vapor mass fraction
    #Qt_lif: lifted parcel total water mass fraction
    #B_lif: Lifted parcel buoyancy, computed using Eq. B6 in (Peters et al.
    #2022) (accounts for virtual temperature and loading effects)
    
    #prate: precipitation rate (in m^-1) large values make parcel more
    #pseudoadiabatic, small values make parcel more adiabatic.  I usually
    #just set it to 0 to get an adiabatic parce
    
    #z0: sounding profile of height above ground level (first height should
    #be 0 m)
    #T1 warmest mixed-phase temperature
    #T2 coldest mixed-phase temperature

    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv   
    g=9.81 #gravitational acceleration
    cp=1005
    xlv=2501000 #reference latent heat of vaporization at the triple point temperature
    
    #ESTIMATE THE MOIST STATIC ENERGY (MSE)
    MSE = cp*T0 + xlv*q0 + g*z0
    mn_hgt = np.where(MSE==np.min(MSE)) #FIND THE INDEX OF THE HEIGHT OF MINIMUM MSE
    
    #descriminator function between liquid and ice (i.e., omega defined in the
    #beginning of section 2e in Peters et al. 2022)

    
    T_lif=np.zeros(T0.shape)*np.nan #temperature of the lifted parcel
    Qv_lif=np.zeros(T0.shape)*np.nan #water vapor mass fraction of the lifted parcel
    Qt_lif=np.ones(T0.shape)*np.nan #total water mass fraction of the lifted parcel

    if start_loc>0:
        T_lif[0:start_loc+1]=T0[0:start_loc+1] #set initial values to that of the environment
        Qv_lif[0:start_loc+1]=q0[0:start_loc+1] #set initial values to that of the environment
        Qt_lif[0:start_loc+1]=Qv_lif[0:start_loc+1] #set initial values to that of the environment
    else:
        T_lif[0]=T0[0] #set initial values to that of the environment
        Qv_lif[0]=q0[0] #set initial values to that of the environment
        Qt_lif[0]=Qv_lif[0] #set initial values to that of the environment


    q_sat_prev=0
    B_run = 0
    iz=start_loc
    #
    #for iz in np.arange(start_loc+1,z0.shape[0]):
    #
    #
    #I REVISED THIS A BIT.  TO MAKE THE CODE FASTER, I HAVE THE CALCULATION CUT OUT WHEN THE INTEGRATED NEGATIVE BUOYANCY ("BRUN") 
    #BECOMES MORE NEGATIVE THAN THAN THE TOTAL INTEGRATED POSITIVE BUOYANCY.  I RESTRICT THIS TO ONLY HAPPEN AFTER WE HAVE PASSED 
    #THE HEIGHT OF MINIMUM MSE.  UNCOMMENT THE FOR LOOP ABOVE AND COMMENT OUT THE WHILE LOOP IF YOU JUST WANT TO INTEGRATE TO THE TOP OF THE SOUNDING.
    #THE +25 PART IN THE WHILE STATEMENT IS A PAD ON B_RUN (THE NEGATIVE CAPE HAS TO BE 25 J/KG LESS THAN THE POSITIVE CAPE TO KILL THE LOOP)
    #while iz<(z0.shape[0])-1 and (z0[iz]<z0[mn_hgt] or (B_run+25)>0):
    while iz<(z0.shape[0])-1 and (z0[iz]<z0[mn_hgt] or (B_run+250)>0):
        iz = iz + 1
        q_sat=(1-Qt_lif[iz-1])*compute_rsat(T_lif[iz-1],p0[iz-1],1,T1,T2)
        if Qv_lif[iz-1]<q_sat: #if we are unsaturated, go up at the unsaturated adiabatic lapse rate (eq. 19 in Peters et al. 2022)
            
        
        
            T_lif[iz] = T_lif[iz-1] + (z0[iz] - z0[iz-1])*drylift(T_lif[iz-1],Qv_lif[iz-1],T0[iz-1],q0[iz-1],fracent)
            Qv_lif[iz] = Qv_lif[iz-1] - (z0[iz] - z0[iz-1])*fracent*( Qv_lif[iz-1] - q0[iz-1] )
            Qt_lif[iz] = Qv_lif[iz]
            q_sat=(1-Qt_lif[iz])*compute_rsat(T_lif[iz],p0[iz],1,T1,T2)
            
            if Qv_lif[iz]>=q_sat: #if we hit saturation, split the vertical step into two stages.  The first stage advances at the saturated lapse rate to the saturation point, and the second stage completes the grid step at the moist lapse rate
                OMEGA = omega(T_lif[iz-1],T1,T2)
                dOMEGA = domega(T_lif[iz-1],T1,T2)
                satrat=(Qv_lif[iz]-q_sat_prev)/(q_sat-q_sat_prev)
                dz_dry=satrat*(z0[iz]-z0[iz-1])
                dz_wet=(1-satrat)*(z0[iz]-z0[iz-1])


                
                T_halfstep = T_lif[iz-1] + dz_dry*drylift(T_lif[iz-1],Qv_lif[iz-1],T0[iz-1],q0[iz-1],fracent)
                Qv_halfstep = Qv_lif[iz-1] - dz_dry*fracent*( Qv_lif[iz-1] - q0[iz-1] )
                Qt_halfstep = Qv_lif[iz]
                p_halfstep=p0[iz-1]*satrat + p0[iz]*(1-satrat)
                T0_halfstep=T0[iz-1]*satrat + T0[iz]*(1-satrat)
                Q0_halfstep=q0[iz-1]*satrat + q0[iz]*(1-satrat)

                T_lif[iz] = T_halfstep + dz_wet*moislif(T_halfstep,Qv_halfstep,(1-Qt_halfstep)*compute_rsat(T_halfstep,p_halfstep,0,T1,T2),(1-Qt_halfstep)*compute_rsat(T_halfstep,p_halfstep,2,T1,T2),p_halfstep,T0_halfstep,Q0_halfstep,Qt_halfstep,fracent,prate,T1,T2)
                
                
                Qt_lif[iz] = Qt_lif[iz-1] - (z0[iz] - z0[iz-1])*fracent*( Qt_halfstep - Q0_halfstep )
                Qv_lif[iz] = (1-Qt_lif[iz])*compute_rsat(T_lif[iz],p0[iz],1,T1,T2)

                if Qt_lif[iz]<Qv_lif[iz]:
                    Qv_lif[iz]=Qt_lif[iz]

            q_sat_prev=q_sat;
            
        else: #if we are already at saturation, just advance upward using the saturated lapse rate (eq. 24 in Peters et al. 2022)
            OMEGA = omega(T_lif[iz-1],T1,T2)
            dOMEGA = domega(T_lif[iz-1],T1,T2)

            T_lif[iz] = T_lif[iz-1] + (z0[iz] - z0[iz-1])*moislif(T_lif[iz-1],Qv_lif[iz-1],(1-Qt_lif[iz-1])*compute_rsat(T_lif[iz-1],p0[iz-1],0,T1,T2),(1-Qt_lif[iz-1])*compute_rsat(T_lif[iz-1],p0[iz-1],2,T1,T2),p0[iz-1],T0[iz-1],q0[iz-1],Qt_lif[iz-1],fracent,prate,T1,T2);
                     
             
            Qt_lif[iz] = Qt_lif[iz-1] - (z0[iz] - z0[iz-1])*(fracent*( Qt_lif[iz-1] - q0[iz-1] )  + prate*( Qt_lif[iz-1]-Qv_lif[iz-1]) )
            Qv_lif[iz] = (1-Qt_lif[iz])*compute_rsat(T_lif[iz],p0[iz],1,T1,T2)
            
            if Qt_lif[iz]<Qv_lif[iz]:
                Qv_lif[iz]=Qt_lif[iz]

        B_run = B_run + (g*T_lif[iz]*(1 + (Rv/Rd)*Qv_lif[iz] - Qt_lif[iz])/(T0[iz]*(1 + (Rv/Rd)*q0[iz] - q0[iz])) - g)*(z0[iz]-z0[iz-1])

    T_rho_lif = T_lif*(1 + (Rv/Rd)*Qv_lif - Qt_lif)
    T_0_lif = T0*(1 + (Rv/Rd - 1)*q0)
    #T_rho_lif=T_lif*(1 - Qt_lif + Qv_lif)/( 1 + (epsilon - 1)/( ( epsilon*(1 - Qt_lif)/Qv_lif - 1) ) )
    #T_0_lif=T0/( 1 + (epsilon - 1)/( ( epsilon*(1 - q0)/q0 - 1) ) )
    
    B_lif=g*(T_rho_lif - T_0_lif)/T_0_lif
    
    
    return T_lif,Qv_lif,Qt_lif,B_lif

#==============================================================================
#FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPE_AND_CIN(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2):
#[CAPE,CIN,LFC,EL]

    #this function computes CAPE and CIN
    
    #input arguments
    #T0: sounding profile of temperature (in K)
    #p0: sounding profile of pressure (in Pa)
    #q0: sounding profile of water vapor mass fraction (in kg/kg)
    #start_loc: index of the parcel starting location (set to 1 for the
    #lowest: level in the sounding)
    #fracent: fractional entrainment rate (in m^-1)
    
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv   
    g=9.81 #gravitational acceleration
    
    #compute lifted parcel buoyancy
    T_lif,Qv_lif,Qt_lif,B_lif=lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2)
    
    if np.nanmax(B_lif)>0:
        #CAPE will be the total integrated positive buoyancy
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]
        B_pos[np.where(B_pos<0)]=0
        dz = z0[1:z0.shape[0]] - z0[0:z0.shape[0]-1]
        CAPE = np.nansum( 0.5*B_pos[0:z0.shape[0]-1]*dz + 0.5*B_pos[1:z0.shape[0]]*dz )
        
        #CIN will be the total negative buoyancy below the height of maximum
        #buoyancy
        B_neg = np.zeros(B_lif.shape)
        B_neg[:] = B_lif[:]
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif==mx)
        imx=imx[0][0]
        B_neg[0:imx]=np.minimum( B_neg[0:imx], 0 )
        B_neg[imx:z0.shape[0]]= 0
        CIN = np.nansum( 0.5*B_neg[0:z0.shape[0]-1]*dz + 0.5*B_neg[1:z0.shape[0]]*dz )
        
        #LFC will be the last instance of negative buoyancy before the
        #continuous interval that contains the maximum in buoyancy
        fneg = np.where(B_lif<0)
        fneg=fneg[0]
        inn = np.where(fneg<imx)
        inn = inn[0]
        fneg = fneg[inn]
        if len(fneg)>0:
            LFC = 0.5*z0[np.max(fneg)] + 0.5*z0[np.max(fneg)+1]
        else:
            LFC = z0[start_loc]
        
        #EL will be last instance of positive buoyancy
        fpos = np.where(B_lif>0)
        fpos=fpos[0]
        EL = 0.5*z0[np.max(fpos)] + 0.5*z0[np.max(fpos)+1]
    else:
        CAPE = 0
        CIN = 0
        LFC = np.nan
        EL = np.nan

    return CAPE,CIN,LFC,EL


#==============================================================================
#FUNCTION THAT COMPUTES NCAPE
def compute_NCAPE(T0,p0,q0,z0,T1,T2,LFC,EL):

    Rd=287.04 # %DRY GAS CONSTANT
    Rv=461.5 # %GAS CONSTANT FOR WATEEER VAPRR
    epsilon=Rd/Rv # %RATO OF THE TWO
    cp=1005 #HEAT CAPACITY OF DRY AIR AT CONSTANT PRESSUREE
    gamma=Rd/cp #POTENTIAL TEMPERATURE EXPONENT
    g=9.81 #GRAVITATIONAL CONSTANT
    Gamma_d=g/cp #DRY ADIABATIC LAPSE RATE
    xlv=2501000 #LATENT HEAT OF VAPORIZATION AT TRIPLE POINT TEMPERATURE
    xls=2834000 #LATENT HEAT OF SUBLIMATION AT TRIPLE POINT TEMPERATURE
    cpv=1870 #HEAT CAPACITY OF WATER VAPOR AT CONSTANT PRESSURE
    cpl=4190 #HEAT CAPACITY OF LIQUID WATER
    cpi=2106 #HEAT CAPACITY OF ICE
    pref=611.65 #REFERENCE VAPOR PRESSURE OF WATER VAPOR AT TRIPLE POINT TEMPERATURE
    ttrip=273.15 #TRIPLE POINT TEMPERATURE
    
    #COMPUTE THE MOIST STATIC ENERGY
    MSE0 = cp*T0 + xlv*q0 + g*z0
    
    #COMPUTE THE SATURATED MOIST STATIC ENERGY
    rsat = compute_rsat(T0,p0,0,T1,T2)
    qsat = (1 - rsat)*rsat
    MSE0_star = cp*T0 + xlv*qsat + g*z0
    
    #COMPUTE MSE0_BAR
    MSE0bar=np.zeros(MSE0.shape)
    #for iz in np.arange(0,MSE0bar.shape[0],1):
     #   MSE0bar[iz]=np.mean(MSE0[1:iz])
        
    MSE0bar[0]=MSE0[0]
    for iz in np.arange(1,MSE0bar.shape[0],1):
        MSE0bar[iz] = 0.5*np.sum( (MSE0[0:iz] + MSE0[1:iz+1])*(z0[1:iz+1]-z0[0:iz]) )/(z0[iz]-z0[0])
    
    int_arg = - ( g/(cp*T0) )*( MSE0bar - MSE0_star)
    ddiff = abs(z0-LFC)
    mn = np.min(ddiff)
    ind_LFC = np.where(ddiff==mn)[0][0]
    ddiff = abs(z0-EL)
    mn = np.min(ddiff)
    ind_EL = np.where(ddiff==mn)[0][0]
    #ind_LFC=max(ind_LFC);
    #ind_EL=max(ind_EL);
    
    NCAPE = np.maximum(np.nansum( (0.5*int_arg[ind_LFC:ind_EL-1] + 0.5*int_arg[ind_LFC+1:ind_EL] )*(z0[ind_LFC+1:ind_EL] - z0[ind_LFC:ind_EL-1] ) ),0)
    return NCAPE,MSE0_star,MSE0bar


#==============================================================================
#FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_VSR(z0,u0,v0):
    #compute 0-1 km storm-relative flow (V_SR) using the storm motion
    #estimate of Bunkers et al. (2000)
    #https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
    
    f6000 = np.where(z0<=6000)[0]
    meanx=np.mean(u0[f6000])
    meany=np.mean(v0[f6000])
    
    f0500 = np.where(z0<=500)[0]
    lowx=np.mean(u0[f0500])
    lowy=np.mean(v0[f0500])
    
    f560 = np.where(np.logical_and(z0<=6000,z0>=5500))[0]
    highx=np.mean(u0[f560])
    highy=np.mean(v0[f560])
    BK_SHRx=highx-lowx
    BK_SHRy=highy-lowy
    BK_mag=np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx=BK_SHRx/BK_mag
    BK_diry=BK_SHRy/BK_mag
    BK_orthx=BK_diry*7.5
    BK_orthy=-BK_dirx*7.5


    SR_mean_u= u0 - meanx
    SR_mean_v= v0 - meany
    dudz=np.zeros(u0.shape)
    dvdz=np.zeros(v0.shape)
    dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dudz[0]=2*dudz[1]-dudz[2]
    dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dvdz[0]=2*dvdz[1]-dvdz[2]
    f1000 = np.where(z0<=1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000]*dvdz[f1000] + SR_mean_v[f1000]*dudz[f1000])*1000.0)
    
    
    propfac=min(SRH_mean/150,1)
    propfac = 1


    C_x=meanx+propfac*BK_orthx
    C_y=meany+propfac*BK_orthy
    
    u_sr = u0 - C_x
    v_sr = v0 - C_y
    
    f1000 = np.where(z0<=1000)[0]
    V_SR = np.nanmean(np.sqrt(  u_sr[f1000]**2 + v_sr[f1000]**2  ))
    return V_SR,C_x,C_y


#==============================================================================
def compute_ETILDE(CAPE,NCAPE,V_SR,EL,L):
    #THESE ARE A BUNCH OF CONSTANT PARAMTERS SET FOR THE ECAPE CALCULATION
    H=EL
    l=L/H
    sigma = 1.1
    alpha=0.8
    Pr=1/3 #PRANDTL NUMBER
    ksq=0.18 #VON KARMAN CONSTANT
    pitchfork=ksq*(alpha**2)*(np.pi**2)*L/(4*Pr*(sigma**2)*H)
    vsr_tilde = V_SR/np.sqrt(2*CAPE)
    N_tilde = NCAPE/CAPE
    
    #EQUATION SOLVES FOR THE NONDIMENSIONAL ECAPE (E_TILDE_A IN THE PAPER)
    E_tilde = vsr_tilde**2 + ( -1 - pitchfork - (pitchfork/(vsr_tilde**2 ))*N_tilde + \
                              np.sqrt((1 + pitchfork + (pitchfork/(vsr_tilde**2 ))*N_tilde)**2 + \
                                      (4*(pitchfork/(vsr_tilde**2 ))*(1 - pitchfork*N_tilde) ) ) )/( 2*pitchfork/(vsr_tilde**2) )
        
    E_tilde_ = E_tilde - vsr_tilde**2
        
    varepsilon = 2*((1 - E_tilde_)/(E_tilde_ + N_tilde))/(EL)  
    

    #eps = 2*ksq*L/(EL*Pr)
    
    #Rm2 = ( (alpha*np.pi/(sigma) )**2 )*( E_tilde/vsr_tilde + 1)
    #Radius =  EL*Rm2**(-1/2)
    #varepsilon = 2*ksq*L/(Pr*Radius**2 )
    
    #Radius=Radius/2
    
    #varepsilon = 0.65*eps*(alpha**2)*(np.pi**2)*E_tilde/(4*(sigma**2)*EL*(vsr_tilde**2 ) ) #THIS IS THE FRACTIONAL ENTRAINMENT RATE
    Radius = np.sqrt(2*ksq*L/(Pr*varepsilon))

    return E_tilde,varepsilon,Radius

#==============================================================================
def CI_model(T0,p0,q0,z0,u0,v0,T1,T2,radrng,itmax,L,prate_global):
    
    #THIS FUNCTION EXECUTES THE "PROGRESSIVE ROOTING" TOY MODEL DESCRIBED BY PETERS ET AL. 2022A
    #https://journals.ametsoc.org/view/journals/atsc/79/6/JAS-D-21-0145.1.xml
    
    #NOTE, A VAREITY OF THINGS HAVE CHANGED SINCE THAT PUBLICATION.  I WILL TRY TO 
    #POINT SPECIFIC EQUATIONS HERE TO EQUATION NUMBERS IN THE PUBLCIATION.  I WILL PROBABLY
    #CREATE A TECHNICAL DOCUMENT TO DESCRIBE THESE CHANGES SOMETIME SOON.  STAY TUNED...
    
    #THE FUNCTION TAKES AS INPUT:
        #T0, profile of temperature (K)
        #p0, profile of pressure (Pa)
        #q0, profile of specific humidity (kg/kg)
        #z0, profile of height above ground level (m)
        #u0, profile of u wind (m/s)
        #v0, profile of v wind (m/s)
        #T1, temperature at which freezing begins in parcel calculations (I usually set to 273.15 K)
        #T2, temperature at which freezing ends in the parcel calculation (K).  This will control the temperature
            #range over which mixed-phase occurs.  I usually set to 253.15 k
        #radrng, a vector containing the initial radii we are going to test.  A reasonable
            #choice here is a range from 100 m to 6000 m at intervals of 100 m (np.arange(100,6000,100))
        #itmax, the number of iterations (I usually set to 20)
        #L, the mixing length (I usually set to 250 m)
        #prate_global, the precipitation loss inverse length scale (km^(-1)).  Larger values make the
            #parcel more pseudoadiabatic, smaller values make it more adiabatic.
    
    #STANDARD THERMODYNAMIC CONSTANTS
    Rd=287.04 # %DRY GAS CONSTANT
    Rv=461.5 # %GAS CONSTANT FOR WATEEER VAPRR
    epsilon=Rd/Rv # %RATO OF THE TWO
    cp=1005 #HEAT CAPACITY OF DRY AIR AT CONSTANT PRESSUREE
    gamma=Rd/cp #POTENTIAL TEMPERATURE EXPONENT
    g=9.81 #GRAVITATIONAL CONSTANT
    Gamma_d=g/cp #DRY ADIABATIC LAPSE RATE
    xlv=2501000 #LATENT HEAT OF VAPORIZATION AT TRIPLE POINT TEMPERATURE
    xls=2834000 #LATENT HEAT OF SUBLIMATION AT TRIPLE POINT TEMPERATURE
    cpv=1870 #HEAT CAPACITY OF WATER VAPOR AT CONSTANT PRESSURE
    cpl=4190 #HEAT CAPACITY OF LIQUID WATER
    cpi=2106 #HEAT CAPACITY OF ICE
    pref=611.65 #REFERENCE VAPOR PRESSURE OF WATER VAPOR AT TRIPLE POINT TEMPERATURE
    ttrip=273.15 #TRIPLE POINT TEMPERATURE
    

    #PARAMTERS UNIQUE TO THE CI MODEL
    alpha=0.8 #ASSUMED RATIO OF HORIZONTALLY AVERAGED W TO HORIZONTAL MAX OF W AT A GIVEN LEVEL
    Pr=1/3 #PRANDTL NUMBER
    ksq=0.18 #VON KARMAN CONSTANT
    start_loc = 0 #STARTING HEIGHT OF THE AIR PARCEL WE ARE LIFTING
    sig = 0.5 #RATIO OF THE HEIGHT OF WMAX TO EQUILBIRIUM LEVEL HEIGHT (SHOULD PROBABLY SET THIS TO 1)
    rfac = 1/4 #RELAXATION FACTOR FOR MODEL INTEGRATION.  SMALLER VALUE GIVES A SMOOTHER SOLUTION
    
    #WE WILL NEED THE DENSITY PROFILE TO COMPUTE THE STORM-RELATIVE WIND LATER
    rho0 = p0/(Rd*T0*(1 + (Rv/Rd - 1)*q0))

    #TIME SERIES OF QUANTITIES OUTPUTTED FROM THE CI MODEL
    R_TS = np.zeros((radrng.shape[0],itmax)) #RADIUS OF THE UPDRAFT
    H_TS = np.zeros((radrng.shape[0],itmax)) #EL HEIGHT
    W_TS = np.zeros((radrng.shape[0],itmax)) #MAX VERTICAL VELOCITY
    VSR_TS = np.zeros((radrng.shape[0],itmax)) #STORM-RELATIVE FLOW
    
    #INITIAL CONDITION ON RADIUS: SET TO R0
    R_TS[:,0]=radrng 
    
    
    #dudz=np.zeros(u0.shape)
    #dvdz=np.zeros(v0.shape)
    #dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    #dudz[0]=2*dudz[1]-dudz[2]
    #dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    #dvdz[0]=2*dvdz[1]-dvdz[2]
    #SHR_mag = np.sqrt(dudz**2 + dvdz**2)
    
    #IN THE FUTRE, WE'LL PROBABLY WANT TO COMPUTE THE DENSITY WEIGHTED STORM-RELATIVE FLOW, LIKE IN THE ECAPE THEORY
    #UDCAPE,UDCIN,UDLFC,UDEL=compute_CAPE_AND_CIN(T0,p0,q0,start_loc,0,prate_global,z0,T1,T2)

    #PARAMETERS FOR CI MODEL
    for it in np.arange(0,itmax-1,1): #LOOP THROUGH THE SPECIFIED NUMBER OF ITERATIONS
        for ir in np.arange(0,radrng.shape[0],1): #LOOP THROUGH EACH OF THE STARTING RADII
            R_on = R_TS[ir,it] #STORE THE RADIUS (IN M)
            #
            fracent = 2*ksq*L/(Pr*(R_on**2)) #USE RADIUS TO COMPUTE FRACTION ENTRAINMENT RATE WITH EQ. XX IN XX
            
            #WHEN COMPUTING THE VERTICAL PROFILE OF KINETIC ENERGY, THE LOWER BOUNDARY CONDITION IS THAT A PARCEL
            #BEGINS WITH THE KINETIC ENERGY OF THE INFLOW.  THIS MEANS WE HAVE TO GIVE THE VERTICAl VELOCITY
            #FUNCTION THE STORM RELATIVE WIND.
            if it == 0:
                #AT THE FIRST TIME STEP, WE WONT HAVE THE STORM RELATIVE WIND YET SO WE'LL MAKE AN AD-HOC ESTIMATE
                #V_SR = 15*R_on/5000 #5.0  
                V_SR = 20*R_on/5000 #5.0  
            else:
                #AT LATER TIMES, WE JUST USE THE STORM RELATIVE FLOW FROM THE PREVIOUS TIME STEP
                V_SR = VSR_TS[ir,it-1]
                    
            #GET THE MAXIMUM VERTICAL VELOCITY PROFILE FOR A RISING CLOUD THERMAL
            CAPE,LFC,EL,B_pos=compute_w(T0,p0,q0,start_loc,fracent,prate_global,z0,T1,T2,R_on,u0,v0,V_SR)
            #NOW THE VERTICAL VELOCITY AT THE BASE OF THE THERMAL, WHICH WILL EXPERIENCE A HIGHER ENTRAINMENT RATE
            CAPE2,LFC2,EL2,null=compute_w(T0,p0,q0,start_loc,fracent*9/4,prate_global,z0,T1,T2,R_on,u0,v0,V_SR)
            
            #WE WILL NEED THE PROFILE OF POSITIVE BUOYANCY TO ESTIMATE STORM MOTION LATER.  
            #ZERO OUT THE NEGATIVE BUOYANCY
            B_pos = np.maximum(B_pos,0)

            #IF WE ACTUALLY HAVE ANY POSITIVE BUOYANCY, WE'LL ADVANCE THE MODEL
            if ~np.isnan(EL):
                
                #GET THE 0-1 KM STORM-RELATIVE FLOW
                V_SR = compute_VSR_DIFF(z0,u0,v0,rho0,EL,B_pos)
                #V_SR = compute_VSR(z0,u0,v0)
                     
                #ADVANCE TO THE NEXT RADIUS USING EQ XX IN XX
                R_next = 1.7*( (EL2/EL)**2 )*2*V_SR*(EL-LFC)*sig/(np.pi*alpha*np.sqrt(2*CAPE))
                R_next = (rfac)*R_next + (1-rfac)*R_on #RELAXATION PROCEEDURE
                
            else: #OTHERWISE SET THE RADIUS AT THE NEXT TIME TO ZERO
                R_next = 0
                
            if EL<LFC: #THIS HAPPENS SOMETIMES.  SET TO ZERO IF EL IS LESS THAN LFC
                R_next = 0
                
            #STORE TIME SERIES'
            R_TS[ir,it+1] = R_next
            H_TS[ir,it]=EL
            W_TS[ir,it]=np.sqrt(2*CAPE)
            VSR_TS[ir,it]=V_SR
        R_TS[np.where(np.isnan(R_TS))]=0
    
    return R_TS,H_TS,W_TS,VSR_TS
            
    
#==============================================================================
#FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_w(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2,Radius,u0,v0,V_SR):
#[CAPE,CIN,LFC,EL]

    #this function computes CAPE and CIN
    
    #input arguments
    #T0: sounding profile of temperature (in K)
    #p0: sounding profile of pressure (in Pa)
    #q0: sounding profile of water vapor mass fraction (in kg/kg)
    #start_loc: index of the parcel starting location (set to 1 for the
    #lowest: level in the sounding)
    #fracent: fractional entrainment rate (in m^-1)
    
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv # %RATO OF THE TWO
    g=9.81 #gravitational acceleration
    c_d = 0.2 #DRAG COEFICIENT ON A SPHERE
    Lambda=0.6 #RATIO OF ASCENT RATE OF THERMAL TO ITS MAX W
    alpha=0.8 #ASSUMED RATIO OF HORIZONTALLY AVERAGED W TO HORIZONTAL MAX OF W AT A GIVEN LEVEL
    
    #COMPUTE A VERTICAL PROFILE OF THE MAGNITUDE OF VERTICAL WIND SHEAR
    dz = np.zeros(u0.shape)
    dz[0:u0.shape[0]-1]=z0[1:u0.shape[0]]-z0[0:u0.shape[0]-1]
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(u0.shape)
    dudz[0:dudz.shape[0]-1]=(u0[1:dudz.shape[0]]-u0[0:dudz.shape[0]-1])/dz[0:dudz.shape[0]-1]
    dvdz[0:dudz.shape[0]-1]=(v0[1:dudz.shape[0]]-v0[0:dudz.shape[0]-1])/dz[0:dudz.shape[0]-1]  
    S = np.sqrt( dudz**2 + dvdz**2)                                            
    
    #COMPUTE THE LIFTED PARCEL BUOYANCY
    T_lif,Qv_lif,Qt_lif,B_lif=lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2)
    
    #CALCULATE THE LIFTED CONDENSATION LEVEL
    qdiff = abs(Qt_lif - Qv_lif) #FIGURE OUT THE FIRST HEIGHT WHERE QV STARTS DEVIATING FROM QT, IMPLYING CONDENSATION
    if np.logical_and(~np.isnan(qdiff[1]),np.nanmax(qdiff)>0):
        lcl_ind = np.where(qdiff>0)[0][0]
        LCL = z0[lcl_ind]
    else:
        LCL = 1000
        lcl_ind = np.where(abs(LCL-z0)==np.amin(abs(LCL-z0)))[0][0]
    
    #IF WE HAVE SOME POSITIVE BUOYANCY, PROCEED
    if np.nanmax(B_lif)>0:
        #MAKE A NEW MATRIX THAT WILL ONLY CONTAIN THE POSITIVE PART OF BUOYANCY
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]

        #GET RID OF ALL NEGATIVE BUOYANCY BELOW THE LCL
        B_pos[0:lcl_ind]=0
        wpos = np.where(B_pos>0)[0]
        if len(wpos)>0:
            wpos=wpos[0] #WPOS CONTAINS INDEX OF LCL.  SET TO 0 IF THERE IS NO POSTIVE BUOYANCY
        else:
            wpos=lcl_ind
        B_pos[0:wpos]=0
        dz = z0[1:z0.shape[0]] - z0[0:z0.shape[0]-1]      
  
        #LFC WILL BE THE LAST INSTANCE OF NEGATIVE BUOYANCY BEFORE THE PARCEL REACHES ITS CONTINUOUS INTERVAL OF POSITIVE BUOY
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif==mx)
        imx=imx[0][0]
        
        fneg = np.where(B_lif<0)
        fneg=fneg[0]
        inn = np.where(fneg<imx)
        
        inn = inn[0]
        fneg = fneg[inn]
        if len(inn)>0:
            LFC = 0.5*z0[np.max(fneg)] + 0.5*z0[np.max(fneg)+1]
        else:
            LFC = z0[start_loc]
        
        #EL WILL BE THE LAST INSTANCE OF POSITIVE BUOYANCY
        fpos = np.where(B_lif>0)
        fpos=fpos[0]
        EL = 0.5*z0[np.max(fpos)] + 0.5*z0[np.max(fpos)+1]
        
        #INTIALIZE PROFILE OF SQUARED VERTICAL VELOCITY (I.E., VERTICAL KINETIC ENERGY)
        WSQ_prof = np.zeros(B_pos.shape[0])
        WSQ_prof[start_loc]=(V_SR**2)/2 #LOWER BOUNADRY CONDITION ON VERTICAL KE IS THE KE OF INFLOW
        uprime_prof = np.zeros(B_pos.shape[0]) #INITIALIZE UPRIME PROFILE
        for iz in np.arange(0,WSQ_prof.shape[0]-1,1): #VERTICALLY INTEGRATE
            B_on = B_pos[iz] #STORE THE CURRENT BUOYANCY
            ebuoy_fac = 1/(1 + 2*(alpha**2)*(Radius**2)/((EL-LFC)**2 ) ) #SCALE FACTOR THAT ACCOUNTS FOR EFFECITVE BUOYANCY
            #ns_drag = -2.5*c_d*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            ns_drag = -c_d*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            s_drag = -( c_d/Radius )*(1 - Lambda)/(Lambda**2) #COEFICIENT ON THE SHEARED PART OF DRAG
            sh_drag = (  1/(0.5*np.sqrt(2*WSQ_prof[iz-1])) )*(3*c_d/(8*Radius)) #SHEARED DRAG TERM
            
            
            if np.sqrt(2*WSQ_prof[iz-1])<1: #IF WE HAVE VERY SMALL VERTICAL VELOICTY (LESS THAN 1 M/S, WE NEED TO ZERO OUT THE SHEAR DRAG TERM OR THINGS BLOW UP)
                sh_drag = 0
                
            #NOW VERTICALLY INTEGRATE THE UPRIME AND WSQ EQUATIONS TOGETHER, FOLLOWING EQ. XX AND XX IN XX RESPECTIVELY
            uprime_prof[iz+1] = uprime_prof[iz-1] + ( z0[iz+1]-z0[iz] )*(-sh_drag*uprime_prof[iz]**2 + S[iz] )
            WSQ_prof[iz+1] = WSQ_prof[iz] + ( z0[iz+1]-z0[iz] )*(ebuoy_fac*B_on + ns_drag*WSQ_prof[iz] + s_drag*uprime_prof[iz]*np.sqrt(2*WSQ_prof[iz]))
          
        #WE WILL OUTPUT THE MAXIMUM KE AS THE "CAPE" ARGUMENT                                                                                                          
        CAPE = np.nanmax(WSQ_prof)
        
        #SET THE EL TO THE HEIGHT OF MAXIMUM VERTICAL VELOCITY
        mxval = np.nanmax(WSQ_prof)
        fnval=np.where(WSQ_prof==mxval)
        LFC = LCL
        if fnval[0].shape[0]>0:
            EL = z0[fnval[0][0]]
        else:
            EL = np.nan
    else:
        #IF WE HAVE NO POSITIVE BUOYANCY, SET EVERYTHING TO 0S AND NANS
        CAPE = 0      
        LFC = np.nan
        EL = np.nan
        B_pos = np.zeros(T0.shape)
        

    return CAPE,LFC,EL,B_pos


#==============================================================================
#FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_VSR_DIFF(z0,u0,v0,rho0,EL,B_pos):
    #compute 0-1 km storm-relative flow (V_SR) using the storm motion
    #estimate of Bunkers et al. (2000)
    #https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
    
    zdiff = ( z0 - EL )**2
    ind_top = np.where(zdiff==np.min(zdiff))[0][0]
    inds_avg=np.arange(0,ind_top,1)
    
    meanx = np.nanmean(B_pos[inds_avg]*rho0[inds_avg]*u0[inds_avg])/np.nanmean(B_pos[inds_avg]*rho0[inds_avg])
    meany = np.nanmean(B_pos[inds_avg]*rho0[inds_avg]*v0[inds_avg])/np.nanmean(B_pos[inds_avg]*rho0[inds_avg])
    
    #meanx = np.nanmean(rho0[inds_avg]*u0[inds_avg])/np.nanmean(rho0[inds_avg])
    #meany = np.nanmean(rho0[inds_avg]*v0[inds_avg])/np.nanmean(rho0[inds_avg])
    
    f6000 = np.where(z0<=6000)[0]
    #meanx=np.mean(u0[f6000])
    #meany=np.mean(v0[f6000])
    
    f0500 = np.where(z0<=500)[0]
    lowx=np.mean(u0[f0500])
    lowy=np.mean(v0[f0500])
    
    f560 = np.where(np.logical_and(z0<=6000,z0>=5500))[0]
    highx=np.mean(u0[f560])
    highy=np.mean(v0[f560])
    BK_SHRx=highx-lowx
    BK_SHRy=highy-lowy
    BK_mag=np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx=BK_SHRx/BK_mag
    BK_diry=BK_SHRy/BK_mag
    BK_orthx=BK_diry*7.5
    BK_orthy=-BK_dirx*7.5


    SR_mean_u= u0 - meanx
    SR_mean_v= v0 - meany
    dudz=np.zeros(u0.shape)
    dvdz=np.zeros(v0.shape)
    dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dudz[0]=2*dudz[1]-dudz[2]
    dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dvdz[0]=2*dvdz[1]-dvdz[2]
    f1000 = np.where(z0<=1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000]*dvdz[f1000] + SR_mean_v[f1000]*dudz[f1000])*1000.0)
    
    
    propfac=min(SRH_mean/150,1)


    C_x=meanx+propfac*BK_orthx
    C_y=meany+propfac*BK_orthy
    
    u_sr = u0 - C_x
    v_sr = v0 - C_y
    
    f1000 = np.where(z0<=1000)[0]
    V_SR = np.nanmean(np.sqrt(  u_sr[f1000]**2 + v_sr[f1000]**2  ))
    return V_SR



#==============================================================================
#FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_OMEGA_AND_SRH(z0,u0,v0,C_x,C_y,rho0,EL):
    #compute 0-1 km storm-relative flow (V_SR) using the storm motion
    #estimate of Bunkers et al. (2000)
    #https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2
    
    zdiff = ( z0 - EL )**2
    ind_top = np.where(zdiff==np.min(zdiff))[0][0]
    inds_avg=np.arange(0,ind_top,1)
    
    meanx = np.nanmean(rho0[inds_avg]*u0[inds_avg])/np.nanmean(rho0[inds_avg])
    meany = np.nanmean(rho0[inds_avg]*v0[inds_avg])/np.nanmean(rho0[inds_avg])
    
    f6000 = np.where(z0<=6000)[0]
    #meanx=np.mean(u0[f6000])
    #meany=np.mean(v0[f6000])
    
    f0500 = np.where(z0<=500)[0]
    lowx=np.mean(u0[f0500])
    lowy=np.mean(v0[f0500])
    
    f560 = np.where(np.logical_and(z0<=6000,z0>=5500))[0]
    highx=np.mean(u0[f560])
    highy=np.mean(v0[f560])
    BK_SHRx=highx-lowx
    BK_SHRy=highy-lowy
    BK_mag=np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx=BK_SHRx/BK_mag
    BK_diry=BK_SHRy/BK_mag
    BK_orthx=BK_diry*7.5
    BK_orthy=-BK_dirx*7.5


    SR_mean_u= u0 - meanx
    SR_mean_v= v0 - meany
    dudz=np.zeros(u0.shape)
    dvdz=np.zeros(v0.shape)
    dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dudz[0]=2*dudz[1]-dudz[2]
    dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dvdz[0]=2*dvdz[1]-dvdz[2]
    f1000 = np.where(z0<=1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000]*dvdz[f1000] + SR_mean_v[f1000]*dudz[f1000])*1000.0)
    
    
    #propfac=min(SRH_mean/150,2)
    propfac=min(SRH_mean/250,2)
    #propfac=1

    if C_x<0:
        C_x=meanx+propfac*BK_orthx
        C_y=meany+propfac*BK_orthy
        
    u_sr = u0 - C_x
    v_sr = v0 - C_y
    sr_mag = np.sqrt(  u_sr**2 + v_sr**2 )
    dudz=np.zeros(u0.shape)
    dvdz=np.zeros(v0.shape)
    dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dudz[0]=2*dudz[1]-dudz[2]
    dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    dvdz[0]=2*dvdz[1]-dvdz[2]
    f1000 = np.where(z0<=1000)[0]
    SRH = abs(np.mean(-u_sr[f1000]*dvdz[f1000] + v_sr[f1000]*dudz[f1000])*1000.0)
    OMEGA = np.mean( (-u_sr[f1000]*dvdz[f1000] + v_sr[f1000]*dudz[f1000])/sr_mag[f1000] )
        
    
    
    f1000 = np.where(z0<=1000)[0]
    V_SR = np.nanmean(np.sqrt(  u_sr[f1000]**2 + v_sr[f1000]**2  ))
    return V_SR,C_x,C_y,SRH,OMEGA




#==============================================================================
#FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPE_CONTS(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2):
#[CAPE,CIN,LFC,EL]

    #this function computes CAPE and CIN
    
    #input arguments
    #T0: sounding profile of temperature (in K)
    #p0: sounding profile of pressure (in Pa)
    #q0: sounding profile of water vapor mass fraction (in kg/kg)
    #start_loc: index of the parcel starting location (set to 1 for the
    #lowest: level in the sounding)
    #fracent: fractional entrainment rate (in m^-1)
    
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv   
    g=9.81 #gravitational acceleration
    
    #compute lifted parcel buoyancy
    T_lif,Qv_lif,Qt_lif,B_lif=lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2)
    
    #compute lifted parcel buoyancy
    T_lif_p,Qv_lif_p,Qt_lif_p,B_lif_p=lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,0.01,z0,T1,T2)
    
    #compute thermal buoyancy 
    B_therm = g*(T_lif - T0)/T0
    
    B_therm_p = g*(T_lif_p - T0)/T0
    B_cond = B_therm - B_therm_p
    
    #water vapor buoyancy contribution
    B_vap = g*(Rv/Rd - 1)*(Qv_lif - q0)
    
    #condensate loading contribution
    B_load = -g*(Qt_lif - Qv_lif)
    
    
    
    if np.nanmax(B_lif)>0:
        #CAPE will be the total integrated positive buoyancy
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]
        
        B_therm[np.where(B_pos<0)]=0
        B_cond[np.where(B_pos<0)]=0
        B_vap[np.where(B_pos<0)]=0
        B_load[np.where(B_pos<0)]=0
        
        B_pos[np.where(B_pos<0)]=0
        dz = z0[1:z0.shape[0]] - z0[0:z0.shape[0]-1]
        CAPE = np.nansum( 0.5*B_pos[0:z0.shape[0]-1]*dz + 0.5*B_pos[1:z0.shape[0]]*dz )
        CAPE_therm = np.nansum( 0.5*B_therm[0:z0.shape[0]-1]*dz + 0.5*B_therm[1:z0.shape[0]]*dz )
        CAPE_cond = np.nansum( 0.5*B_cond[0:z0.shape[0]-1]*dz + 0.5*B_cond[1:z0.shape[0]]*dz )
        CAPE_vap = np.nansum( 0.5*B_vap[0:z0.shape[0]-1]*dz + 0.5*B_vap[1:z0.shape[0]]*dz )
        CAPE_load = np.nansum( 0.5*B_load[0:z0.shape[0]-1]*dz + 0.5*B_load[1:z0.shape[0]]*dz )
        
        #CIN will be the total negative buoyancy below the height of maximum
        #buoyancy
        B_neg = np.zeros(B_lif.shape)
        B_neg[:] = B_lif[:]
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif==mx)
        imx=imx[0][0]
        B_neg[0:imx]=np.minimum( B_neg[0:imx], 0 )
        B_neg[imx:z0.shape[0]]= 0
        CIN = np.nansum( 0.5*B_neg[0:z0.shape[0]-1]*dz + 0.5*B_neg[1:z0.shape[0]]*dz )
        
        #LFC will be the last instance of negative buoyancy before the
        #continuous interval that contains the maximum in buoyancy
        fneg = np.where(B_lif<0)
        fneg=fneg[0]
        inn = np.where(fneg<imx)
        inn = inn[0]
        fneg = fneg[inn]
        if len(fneg)>0:
            LFC = 0.5*z0[np.max(fneg)] + 0.5*z0[np.max(fneg)+1]
        else:
            LFC = z0[start_loc]
        
        #EL will be last instance of positive buoyancy
        fpos = np.where(B_lif>0)
        fpos=fpos[0]
        EL = 0.5*z0[np.max(fpos)] + 0.5*z0[np.max(fpos)+1]
    else:
        CAPE = 0
        CIN = 0
        LFC = np.nan
        EL = np.nan
        CAPE_therm = 0
        CAPE_cond = 0
        CAPE_vap = 0
        CAPE_load = 0

    return CAPE,CAPE_therm,CAPE_cond,CAPE_vap,CAPE_load



#==============================================================================
#FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPES_DRAG(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2,Radius,V_SR):
#[CAPE,CIN,LFC,EL]

    #this function computes CAPE and CIN
    
    #input arguments
    #T0: sounding profile of temperature (in K)
    #p0: sounding profile of pressure (in Pa)
    #q0: sounding profile of water vapor mass fraction (in kg/kg)
    #start_loc: index of the parcel starting location (set to 1 for the
    #lowest: level in the sounding)
    #fracent: fractional entrainment rate (in m^-1)
    
    #CONSTANTS
    Rd=287.04 #dry gas constant
    Rv=461.5 #water vapor gas constant
    epsilon=Rd/Rv   
    g=9.81 #gravitational acceleration
    cp=1005 #HEAT CAPACITY OF DRY AIR AT CONSTANT PRESSUREE
    alpha = 0.8
    #c_d = 0.2
    c_d = 0.2
    
    th0 = T0*(1000*100/p0)**(Rd/cp)
    
    #compute lifted parcel buoyancy
    T_lif,Qv_lif,Qt_lif,B_lif=lift_parcel_adiabatic(T0,p0,q0,start_loc,fracent,prate,z0,T1,T2)
    
    #CALCULATE THE LIFTED CONDENSATION LEVEL
    qdiff = abs(Qt_lif - Qv_lif) #FIGURE OUT THE FIRST HEIGHT WHERE QV STARTS DEVIATING FROM QT, IMPLYING CONDENSATION
    if np.logical_and(~np.isnan(qdiff[1]),np.nanmax(qdiff)>0):
        lcl_ind = np.where(qdiff>0)[0][0]
        LCL = z0[lcl_ind]
    else:
        LCL = 1000
        lcl_ind = np.where(abs(LCL-z0)==np.amin(abs(LCL-z0)))[0][0]
    
    #IF WE HAVE SOME POSITIVE BUOYANCY, PROCEED
    if np.nanmax(B_lif)>0:
        #MAKE A NEW MATRIX THAT WILL ONLY CONTAIN THE POSITIVE PART OF BUOYANCY
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]

        #GET RID OF ALL NEGATIVE BUOYANCY BELOW THE LCL
        B_pos[0:lcl_ind]=0
        wpos = np.where(B_pos>0)[0]
        if len(wpos)>0:
            wpos=wpos[0] #WPOS CONTAINS INDEX OF LCL.  SET TO 0 IF THERE IS NO POSTIVE BUOYANCY
        else:
            wpos=lcl_ind
        B_pos[0:wpos]=0
        dz = z0[1:z0.shape[0]] - z0[0:z0.shape[0]-1]      
  
        #LFC WILL BE THE LAST INSTANCE OF NEGATIVE BUOYANCY BEFORE THE PARCEL REACHES ITS CONTINUOUS INTERVAL OF POSITIVE BUOY
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif==mx)
        imx=imx[0][0]
        
        fneg = np.where(B_lif<0)
        fneg=fneg[0]
        inn = np.where(fneg<imx)
        
        inn = inn[0]
        fneg = fneg[inn]
        if len(inn)>0:
            LFC = 0.5*z0[np.max(fneg)] + 0.5*z0[np.max(fneg)+1]
        else:
            LFC = z0[start_loc]
        
        #EL WILL BE THE LAST INSTANCE OF POSITIVE BUOYANCY
        fpos = np.where(B_lif>0)
        fpos=fpos[0]
        EL = 0.5*z0[np.max(fpos)] + 0.5*z0[np.max(fpos)+1]
        
        #INTIALIZE PROFILE OF SQUARED VERTICAL VELOCITY (I.E., VERTICAL KINETIC ENERGY)
        WSQ_prof = np.zeros(B_pos.shape[0])
        WSQ_prof[start_loc]=(V_SR**2)/2 #LOWER BOUNADRY CONDITION ON VERTICAL KE IS THE KE OF INFLOW
        for iz in np.arange(0,WSQ_prof.shape[0]-1,1): #VERTICALLY INTEGRATE
            B_on = B_pos[iz] #STORE THE CURRENT BUOYANCY
            ebuoy_fac = 1/(1 + 2*(alpha**2)*(Radius**2)/((EL-LFC)**2 ) ) #SCALE FACTOR THAT ACCOUNTS FOR EFFECITVE BUOYANCY
            #ns_drag = -2.5*c_d*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            
            N = np.max( ( g/th0[iz] )*( th0[iz+1]-th0[iz] )/( z0[iz+1]-z0[iz] ), 0 )
            N = ( np.minimum(z0[iz]/Radius,1)*1/2 + np.minimum( np.maximum(( z0[iz] - Radius)/Radius,0), 1)*(2/3-1/2))*N
            
            F = np.sqrt( np.max( WSQ_prof/2 ,0) )/( np.sqrt(N)*Radius )
            
            if N>0:
                ns_drag = -(c_d + comp_cdwave(F))*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            else:
                ns_drag = -(c_d)*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            
            if np.sqrt(2*WSQ_prof[iz-1])<1: #IF WE HAVE VERY SMALL VERTICAL VELOICTY (LESS THAN 1 M/S, WE NEED TO ZERO OUT THE SHEAR DRAG TERM OR THINGS BLOW UP)
                sh_drag = 0
                
            #NOW VERTICALLY INTEGRATE THE UPRIME AND WSQ EQUATIONS TOGETHER, FOLLOWING EQ. XX AND XX IN XX RESPECTIVELY
            WSQ_prof[iz+1] = WSQ_prof[iz] + ( z0[iz+1]-z0[iz] )*(ebuoy_fac*B_on + ns_drag*WSQ_prof[iz])
          
        #WE WILL OUTPUT THE MAXIMUM KE AS THE "CAPE" ARGUMENT                                                                                                          
        CAPE = np.nanmax(WSQ_prof)
        
        #SET THE EL TO THE HEIGHT OF MAXIMUM VERTICAL VELOCITY
        mxval = np.nanmax(WSQ_prof)
        fnval=np.where(WSQ_prof==mxval)
        LFC = LCL
        if fnval[0].shape[0]>0:
            EL = z0[fnval[0][0]]
        else:
            EL = np.nan
    else:
        #IF WE HAVE NO POSITIVE BUOYANCY, SET EVERYTHING TO 0S AND NANS
        CAPE = 0      
        LFC = np.nan
        EL = np.nan
        B_pos = np.zeros(T0.shape)

    return CAPE

#==============================================================================
#==============================================================================
#=======================================1/======================================
#END FUNCTION DEFINITIONS======================================================
#==============================================================================
#==============================================================================
#==============================================================================



# ============================================================

# LATEST RRFS | REGIONAL DOMAIN
# ORIGINAL PETERS ECAPE + LCL-LFC RH + 0-3/0-6 KM BULK SHEAR
# + LATEST RRFS SOUNDING AT TEST POINT
# ============================================================
#
# OUTPUTS
# -------
# 1) Regional-domain map:
#       Surface-Based ECAPE fill
#       0-1 km mean storm-relative wind contours
#       0-1 km bulk-shear vector barbs (blue)
#       0-3 km bulk-shear vector barbs (red)
#       0-6 km bulk-shear vector barbs (black)
#
# 2) Latest-RRFS sounding:
#       Temperature / dewpoint
#       undiluted SB parcel
#       diagnostic text for CAPE / ECAPE / NCAPE / VSR / shear
#
# ECAPE METHOD
# ------------
# Uses the original functions supplied by the Peters code:
#
#   compute_CAPE_AND_CIN()
#   compute_NCAPE()
#   compute_VSR()
#   compute_ETILDE(..., L=250 m)
#
#   ECAPE = E_tilde * CAPE
#
# RRFS INPUT METHOD
# -----------------
# Uses:
#   - pressure-level HGT / TMP / SPFH / UGRD / VGRD
#   - actual RRFS surface pressure
#   - 2 m T / SPFH
#   - terrain height
#   - 10 m U/V wind
#
# Each RRFS sounding is converted to AGL and interpolated to the
# same 100-m vertical grid used in the original GET_ECAPE.py.
#
# ============================================================


# ============================================================
# DEPENDENCIES
# ============================================================
#
# Required packages:
#
# !pip -q install \
#     cfgrib \
#     eccodes \
#     scipy \
#     xarray \
#     metpy \
#     cartopy \
#     matplotlib \
#     tqdm
#
# ============================================================


import os
import re
import warnings
import requests
import json
import time
import boto3
import zipfile
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import geopandas as gpd

from shapely.ops import unary_union
from shapely.prepared import prep

from datetime import datetime, timedelta, timezone
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

import metpy.calc as mpcalc
from metpy.units import units

from matplotlib.colors import ListedColormap, BoundaryNorm


warnings.filterwarnings("ignore")


# ============================================================
# SITE MAP ASSETS
# ============================================================

ASSET_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "assets"
)


# ============================================================
# EXTRACT LBF CWA SHAPEFILE
# ============================================================

zip_path = os.path.join(
    ASSET_DIR,
    "c_18mr25.zip"
)

if os.path.exists(
    zip_path
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            ASSET_DIR
        )

COUNTY_SHP = os.path.join(
    ASSET_DIR,
    "cb_2018_us_county_500k.shp"
)

STATE_SHP = os.path.join(
    ASSET_DIR,
    "cb_2018_us_state_500k.shp"
)

LBF_CWA_SHP = os.path.join(
    ASSET_DIR,
    "c_18mr25.shp"
)

LOGO_PATH = os.path.join(
    ASSET_DIR,
    "NOAANWSLogos.png"
)


# ============================================================
# MAP SHAPEFILE HELPERS
# ============================================================

def add_shapefile_outline(
    ax,
    shp_path,
    extent,
    edgecolor="black",
    linewidth=1.0,
    zorder=10
):

    if not os.path.exists(
        shp_path
    ):

        print(
            "Missing shapefile:",
            shp_path
        )

        return

    gdf = (
        gpd
        .read_file(
            shp_path
        )
        .to_crs(
            epsg=4326
        )
    )

    lon_min, lon_max, lat_min, lat_max = extent

    gdf = gdf.cx[
        lon_min - 1:
        lon_max + 1,

        lat_min - 1:
        lat_max + 1
    ]

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder
    )


def get_lbf_cwa_geom(
    cwa_shp_path
):

    if not os.path.exists(
        cwa_shp_path
    ):

        print(
            "Missing LBF CWA shapefile:",
            cwa_shp_path
        )

        return None

    reader = shpreader.Reader(
        cwa_shp_path
    )

    recs = list(
        reader.records()
    )

    geoms = [

        r.geometry

        for r in recs

        if (
            str(
                r.attributes.get(
                    "CWA",
                    ""
                )
            )
            .upper()
            ==
            "LBF"
        )

        or

        (
            str(
                r.attributes.get(
                    "WFO",
                    ""
                )
            )
            .upper()
            ==
            "LBF"
        )

    ]

    if not geoms:

        geoms = [
            r.geometry
            for r in recs
        ]

    return unary_union(
        geoms
    )


def add_counties_clipped_to_cwa(
    ax,
    counties_shp_path,
    cwa_geom,
    linewidth=1.0,
    edgecolor="black",
    zorder=14
):

    if (
        cwa_geom is None
        or
        not os.path.exists(
            counties_shp_path
        )
    ):

        return

    reader = shpreader.Reader(
        counties_shp_path
    )

    cwa_prepared = prep(
        cwa_geom
    )

    clipped = []

    for record in reader.records():

        geom = record.geometry

        if cwa_prepared.intersects(
            geom
        ):

            inter = geom.intersection(
                cwa_geom
            )

            if not inter.is_empty:

                clipped.append(
                    inter
                )

    ax.add_geometries(
        clipped,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder
    )


LBF_CWA_GEOM = get_lbf_cwa_geom(
    LBF_CWA_SHP
)



# ============================================================
# R2 / SITE SETUP
# ============================================================

BUCKET = os.environ["AWS_BUCKET"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)


def upload_to_r2(
    local_file,
    remote_key,
    content_type="image/png"
):

    s3.upload_file(
        local_file,
        BUCKET,
        remote_key,
        ExtraArgs={
            "ContentType": content_type
        }
    )

    print(
        "Uploaded to R2:",
        remote_key
    )


def upload_runs_json(
    init_dt,
    cycle_str,
    max_fhr
):

    old_runs = []

    try:

        obj = s3.get_object(
            Bucket=BUCKET,
            Key=(
                f"runs/cams/hrrr/ecape/"
                f"runs.json"
            )
        )

        old_data = json.loads(
            obj["Body"]
            .read()
            .decode("utf-8")
        )

        old_runs = old_data.get(
            "runs",
            []
        )

    except Exception:

        old_runs = []

    new_run = {

        "id":
            cycle_str,

        "label":
            init_dt.strftime(
                "%Y-%m-%d %Hz"
            ),

        "max_fhr":
            max_fhr,

    }

    combined = [
        new_run
    ]

    for run in old_runs:

        if isinstance(
            run,
            str
        ):

            rid = run

            try:

                rhour = int(
                    rid
                    .split("_")[1]
                    .replace(
                        "z",
                        ""
                    )
                )

            except Exception:

                rhour = 0

            combined.append({

                "id":
                    rid,

                "label":
                    rid.replace(
                        "_",
                        " "
                    ),

                "max_fhr":
                    (
                        48
                        if
                        rhour
                        in
                        LONG_CYCLE_HOURS
                        else
                        18
                    ),

            })

        elif (
            run.get("id")
            !=
            cycle_str
        ):

            combined.append(
                run
            )

    runs_json = {
        "runs":
            combined[:4]
    }

    local_runs = os.path.join(
        OUTDIR_BASE,
        "runs.json"
    )

    with open(
        local_runs,
        "w"
    ) as f:

        json.dump(
            runs_json,
            f,
            indent=2
        )

    upload_to_r2(
        local_runs,
        (
            f"runs/cams/hrrr/ecape/"
            f"runs.json"
        ),
        content_type="application/json"
    )


# ============================================================
# SETTINGS
# ============================================================

# Regional map extent
REGIONAL_EXTENT = [
    -107.5,
    -93.0,
    38.5,
    44.2,
]

# Original Peters / GET_ECAPE vertical settings
DZ_M = 100.0
TOP_M = 20000.0
MIXING_LENGTH_M = 250.0

T1 = 273.15
T2 = 253.15

FHR = 0

VALID_RRFS_CYCLES = [
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21
]

LONG_CYCLE_HOURS = [
    0,
    6,
    12,
    18
]

MAX_FHR_LONG = 60
MAX_FHR_SHORT = 18

# RRFS parallel files normally begin at F001.
START_FHR = 1

CYCLE_DELAY_MINUTES = 45

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_SECONDS = 15

FHR_ATTEMPTS = 3
FHR_RETRY_SECONDS = 12

# ------------------------------------------------------------
# MAP CALCULATION STRIDE
#
# RRFS grid spacing is ~3 km.
#
# 4 = ECAPE calculated roughly every 12 km.
#     Good first Colab test.
#
# Change to 3 later for ~9 km.
# Change to 2 later for ~6 km.
# ------------------------------------------------------------

ECAPE_GRID_STRIDE = 4

# Smooth the interpolated ECAPE field very lightly.
ECAPE_SMOOTH_SIGMA = 0.65

# ------------------------------------------------------------
# SHEAR BARBS
#
# Barbs are plotted from the sampled ECAPE grid.
# ------------------------------------------------------------

BARB_SKIP = 8
# ------------------------------------------------------------
# MEAN LCL-LFC RELATIVE HUMIDITY CONTOURS
#
# Mean environmental relative humidity between the
# surface-based parcel LCL and LFC.
# Units: percent.
# ------------------------------------------------------------

LCL_LFC_RH_LEVELS = [40, 50, 60, 70, 80, 90]

RH_COLORS = [
    "#8c6d31",  # 40%
    "#a6611a",  # 50%
    "#dfc27d",  # 60%
    "#80cdc1",  # 70%
    "#35978f",  # 80%
    "#01665e",  # 90%
]

# ------------------------------------------------------------
# SITE OUTPUT
# ------------------------------------------------------------

SECTION_KEY = "cams"
MODEL_KEY = "rrfs"
PRODUCT_KEY = "ecape"

R2_PRODUCT_PATH = (
    f"runs/"
    f"{SECTION_KEY}/"
    f"{MODEL_KEY}/"
    f"{PRODUCT_KEY}"
)

OUTDIR_BASE = os.path.join(
    "site",
    "runs",
    SECTION_KEY,
    MODEL_KEY,
    PRODUCT_KEY
)

os.makedirs(
    OUTDIR_BASE,
    exist_ok=True
)

OUTDIR = None




# ============================================================
# RRFS NOMADS PARALLEL DATA
# ============================================================

RRFS_NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/rrfs/para"
)

DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "rrfs_ecape_subsets"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

# Pressure levels requested from RRFS prslev.
# Missing individual levels are skipped automatically.
RRFS_PRESSURE_LEVELS_HPA = [
    1000, 975, 950, 925, 900, 875, 850, 825,
    800, 775, 750, 725, 700, 675, 650, 625,
    600, 575, 550, 525, 500, 475, 450, 425,
    400, 375, 350, 325, 300, 275, 250, 225,
    200, 175, 150, 125, 100, 75, 50
]

# Cache .idx inventories so dozens of pressure-level requests
# do not redownload the same index file.
_RRFS_IDX_CACHE = {}

# ============================================================
# YOUR ECAPE COLORMAP
# ============================================================

ECAPE_BOUNDS = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
    1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
    2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300,
    3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100,
    4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
    5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700,
    5800, 5900, 6000, 6500, 7000, 7500, 8000, 8500,
    9000, 9500, 10000, 10500
]

ECAPE_COLORS = [
    "#ffffff",
    "#f0f0f0",
    "#e1e1e1",
    "#d2d2d2",
    "#c3c3c3",
    "#a5a5a5",
    "#969696",
    "#878787",
    "#787878",
    "#696969",
    "#3b5269",
    "#475f74",
    "#546c7f",
    "#60798a",
    "#6d8695",
    "#7993a1",
    "#86a0ac",
    "#92adb7",
    "#9fbac2",
    "#abc7ce",
    "#e6de99",
    "#e4d289",
    "#e3c679",
    "#e1b96a",
    "#dfae5a",
    "#dfa24b",
    "#dd963c",
    "#dc8a2f",
    "#da7e24",
    "#d9731c",
    "#d3491f",
    "#cb4323",
    "#c23d27",
    "#b9362b",
    "#b13131",
    "#a82b37",
    "#9f253d",
    "#971f44",
    "#8e1a4a",
    "#861550",
    "#700e89",
    "#7b1c93",
    "#872b9e",
    "#923aa8",
    "#9e4ab2",
    "#a95bbd",
    "#b56ac7",
    "#c07ad1",
    "#cc8adc",
    "#d79ae6",
    "#e6bfc3",
    "#dfb1b7",
    "#d9a4ad",
    "#d297a1",
    "#cc8a95",
    "#c57c8a",
    "#be707e",
    "#b86272",
    "#b25667",
    "#ac485b",
    "#844049",
    "#8a4953",
    "#91545c",
    "#985e66",
    "#9e6970",
    "#a57279",
    "#ab7d83",
    "#b2878c",
    "#b99295"
]

ECAPE_CMAP = ListedColormap(
    ECAPE_COLORS,
    name="ecape_bins"
)

ECAPE_NORM = BoundaryNorm(
    ECAPE_BOUNDS,
    ECAPE_CMAP.N,
    clip=True
)

ECAPE_TICKS = [
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    5500,
    6000,
    8500,
]


# ============================================================
# CITY LABELS
# ============================================================

# City/station labels are intentionally not plotted on this product.



# ============================================================
# BASIC HELPERS
# ============================================================

def rrfs_grib_url(
    init_dt,
    fhr,
    product="2dfld"
):

    ymd = init_dt.strftime(
        "%Y%m%d"
    )

    hh = init_dt.strftime(
        "%H"
    )

    if product == "2dfld":

        fname = (
            f"rrfs.t{hh}z."
            f"2dfld.3km."
            f"f{int(fhr):03d}."
            f"conus.grib2"
        )

    elif product == "prslev":

        fname = (
            f"rrfs.t{hh}z."
            f"prslev.3km."
            f"f{int(fhr):03d}."
            f"conus.grib2"
        )

    else:

        raise ValueError(
            "product must be "
            "'2dfld' or 'prslev'"
        )

    return (
        f"{RRFS_NOMADS_BASE}/"
        f"rrfs.{ymd}/"
        f"{hh}/"
        f"{fname}"
    )


def rrfs_idx_url(
    init_dt,
    fhr,
    product="2dfld"
):

    return (
        rrfs_grib_url(
            init_dt,
            fhr,
            product=product
        )
        +
        ".idx"
    )


def find_latest_available_rrfs_cycle(
    max_back_hours=48
):

    now = (
        datetime.now(
            timezone.utc
        )
        -
        timedelta(
            minutes=CYCLE_DELAY_MINUTES
        )
    )

    print("")
    print("=" * 72)
    print(
        "SEARCHING FOR LATEST "
        "RRFS PARALLEL CYCLE"
    )
    print("=" * 72)

    for back in range(
        max_back_hours + 1
    ):

        dt_aware = (
            now
            -
            timedelta(
                hours=back
            )
        )

        if (
            dt_aware.hour
            not in
            VALID_RRFS_CYCLES
        ):

            continue

        dt = datetime(
            dt_aware.year,
            dt_aware.month,
            dt_aware.day,
            dt_aware.hour
        )

        test_idx = rrfs_idx_url(
            dt,
            1,
            product="2dfld"
        )

        print(
            f"Checking RRFS "
            f"{dt:%Y%m%d %HZ}..."
        )

        try:

            r = requests.get(
                test_idx,
                timeout=15
            )

            if (
                r.status_code == 200
                and
                len(
                    r.text
                )
                > 100
            ):

                print(
                    "Latest RRFS cycle found: "
                    f"{dt:%Y%m%d %HZ}"
                )

                return dt

        except Exception as error:

            print(
                "  cycle check failed:",
                error
            )

    raise RuntimeError(
        "Could not find a recent "
        "RRFS parallel cycle."
    )



def to_lon180(
    lon
):

    return (
        (
            np.asarray(
                lon
            )
            + 180.0
        )
        % 360.0
    ) - 180.0


def get_lat_lon(
    da
):

    if (
        "latitude" in da.coords
        and
        "longitude" in da.coords
    ):

        lat = np.asarray(
            da.latitude.values
        )

        lon = to_lon180(
            da.longitude.values
        )

    elif (
        "lat" in da.coords
        and
        "lon" in da.coords
    ):

        lat = np.asarray(
            da.lat.values
        )

        lon = to_lon180(
            da.lon.values
        )

    else:

        raise RuntimeError(
            "Could not find latitude/longitude."
        )

    return (
        lat,
        lon
    )


def parse_idx_lines(
    lines
):

    parsed = []

    for line in lines:

        parts = line.split(
            ":"
        )

        if len(
            parts
        ) < 3:

            continue

        try:

            msg_num = int(
                parts[0]
            )

            start_byte = int(
                parts[1]
            )

        except Exception:

            continue

        parsed.append(
            {
                "line": line,
                "msg_num": msg_num,
                "start": start_byte,
            }
        )

    for i in range(
        len(
            parsed
        )
    ):

        if (
            i
            <
            len(
                parsed
            )
            - 1
        ):

            parsed[i][
                "end"
            ] = (
                parsed[
                    i + 1
                ][
                    "start"
                ]
                - 1
            )

        else:

            parsed[i][
                "end"
            ] = None

    return parsed


def read_idx_cached(
    idx_url,
    timeout=30
):

    if (
        idx_url
        in
        _RRFS_IDX_CACHE
    ):

        return (
            _RRFS_IDX_CACHE[
                idx_url
            ]
        )

    last_error = None

    for attempt in range(
        1,
        DOWNLOAD_ATTEMPTS + 1
    ):

        try:

            r = requests.get(
                idx_url,
                timeout=timeout
            )

            r.raise_for_status()

            lines = (
                r.text
                .strip()
                .splitlines()
            )

            if (
                len(
                    lines
                )
                == 0
            ):

                raise RuntimeError(
                    "RRFS IDX response "
                    "was empty."
                )

            parsed = (
                parse_idx_lines(
                    lines
                )
            )

            _RRFS_IDX_CACHE[
                idx_url
            ] = parsed

            return parsed

        except Exception as error:

            last_error = error

            print(
                f"IDX attempt "
                f"{attempt}/"
                f"{DOWNLOAD_ATTEMPTS} "
                f"failed: {error}"
            )

            if (
                attempt
                <
                DOWNLOAD_ATTEMPTS
            ):

                time.sleep(
                    DOWNLOAD_RETRY_SECONDS
                )

    raise RuntimeError(
        f"Could not retrieve IDX: "
        f"{idx_url}. "
        f"Last error: "
        f"{last_error}"
    )


def find_idx_match(
    parsed,
    all_terms,
    label
):

    terms = [
        str(
            term
        ).lower()
        for term
        in all_terms
    ]

    for item in parsed:

        line_lower = (
            item[
                "line"
            ]
            .lower()
        )

        if all(
            term
            in line_lower
            for term
            in terms
        ):

            print(
                f"Matched {label}:"
            )

            print(
                item[
                    "line"
                ]
            )

            return item

    return None


def download_byte_range(
    grib_url,
    start_byte,
    end_byte,
    outpath
):

    if end_byte is None:

        range_value = (
            f"bytes={start_byte}-"
        )

    else:

        range_value = (
            f"bytes={start_byte}-"
            f"{end_byte}"
        )

    headers = {
        "Range":
            range_value
    }

    last_error = None

    for attempt in range(
        1,
        DOWNLOAD_ATTEMPTS + 1
    ):

        try:

            r = requests.get(
                grib_url,
                headers=headers,
                stream=True,
                timeout=120
            )

            if (
                r.status_code
                not in
                (
                    200,
                    206
                )
            ):

                r.raise_for_status()

            with open(
                outpath,
                "wb"
            ) as f:

                for chunk in (
                    r.iter_content(
                        chunk_size=
                            1024
                            *
                            1024
                    )
                ):

                    if chunk:

                        f.write(
                            chunk
                        )

            if (
                not os.path.exists(
                    outpath
                )
                or
                os.path.getsize(
                    outpath
                )
                < 100
            ):

                raise RuntimeError(
                    "Downloaded GRIB "
                    "message was empty "
                    "or incomplete."
                )

            return outpath

        except Exception as error:

            last_error = error

            if os.path.exists(
                outpath
            ):

                try:

                    os.remove(
                        outpath
                    )

                except Exception:

                    pass

            if (
                attempt
                <
                DOWNLOAD_ATTEMPTS
            ):

                time.sleep(
                    DOWNLOAD_RETRY_SECONDS
                )

    raise RuntimeError(
        "Byte-range download "
        f"failed: {last_error}"
    )


def open_subset_grib(
    path,
    label
):

    attempts = [
        {},
        {
            "typeOfLevel":
                "heightAboveGround"
        },
        {
            "typeOfLevel":
                "surface"
        },
        {
            "typeOfLevel":
                "isobaricInhPa"
        },
        {
            "typeOfLevel":
                "isobaricInPa"
        },
        {
            "typeOfLevel":
                "atmosphere"
        },
        {
            "typeOfLevel":
                "entireAtmosphere"
        },
    ]

    errors = []

    for filter_keys in (
        attempts
    ):

        try:

            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys":
                        filter_keys,
                    "errors":
                        "ignore",
                }
            )

            if (
                len(
                    ds.data_vars
                )
                == 0
            ):

                ds.close()

                continue

            candidates = [
                name
                for name
                in ds.data_vars
                if name
                not in
                {
                    "latitude",
                    "longitude",
                    "valid_time",
                    "step",
                    "time",
                }
            ]

            if not candidates:

                ds.close()

                continue

            var = (
                candidates[0]
            )

            da = (
                ds[
                    var
                ]
                .load()
            )

            ds.close()

            return da

        except Exception as error:

            errors.append(
                str(
                    error
                )
            )

    raise RuntimeError(
        f"Could not open subset "
        f"for {label}. "
        f"Last errors: "
        f"{errors[-3:]}"
    )


def rrfs_idx_field(
    init_dt,
    fhr,
    term_sets,
    label,
    product="2dfld",
    required=True
):

    grib_url = (
        rrfs_grib_url(
            init_dt,
            fhr,
            product=product
        )
    )

    idx_url = (
        grib_url
        +
        ".idx"
    )

    parsed = (
        read_idx_cached(
            idx_url
        )
    )

    match = None

    for terms in (
        term_sets
    ):

        candidate = (
            find_idx_match(
                parsed,
                terms,
                label
            )
        )

        if (
            candidate
            is not None
        ):

            match = candidate

            break

    if match is None:

        if required:

            raise RuntimeError(
                "Could not find required "
                f"RRFS field {label} "
                f"in {idx_url}. "
                f"Search terms: "
                f"{term_sets}"
            )

        return None

    safe_label = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        label
    ).strip(
        "_"
    )

    outname = (
        f"rrfs_"
        f"{product}_"
        f"{init_dt:%Y%m%d_%H}z_"
        f"f{int(fhr):03d}_"
        f"{safe_label}_"
        f"{match['msg_num']}."
        f"grib2"
    )

    outpath = os.path.join(
        DATA_DIR,
        outname
    )

    try:

        download_byte_range(
            grib_url,
            match[
                "start"
            ],
            match[
                "end"
            ],
            outpath
        )

        return (
            open_subset_grib(
                outpath,
                label
            )
        )

    finally:

        if os.path.exists(
            outpath
        ):

            try:

                os.remove(
                    outpath
                )

            except Exception:

                pass


def ensure_2d_field(
    da,
    label
):

    arr = np.asarray(
        da.values,
        dtype=float
    )

    arr = np.squeeze(
        arr
    )

    if (
        arr.ndim
        != 2
    ):

        raise RuntimeError(
            f"{label} is not 2D "
            f"after squeeze. "
            f"Shape={arr.shape}, "
            f"dims="
            f"{getattr(da, 'dims', None)}"
        )

    return arr


def rrfs_pressure_stack(
    init_dt,
    fhr,
    short_name,
    label
):

    arrays = []
    levels_found = []
    base_lat = None
    base_lon = None

    for level in (
        RRFS_PRESSURE_LEVELS_HPA
    ):

        da = rrfs_idx_field(
            init_dt,
            fhr,
            [
                [
                    short_name,
                    f"{level} mb"
                ]
            ],
            f"{label} {level} mb",
            product="prslev",
            required=False
        )

        if (
            da
            is None
        ):

            continue

        try:

            arr = (
                ensure_2d_field(
                    da,
                    f"{label} "
                    f"{level} mb"
                )
            )

            lat_on, lon_on = (
                get_lat_lon(
                    da
                )
            )

        except Exception as error:

            print(
                f"Skipping {label} "
                f"{level} mb: "
                f"{error}"
            )

            continue

        if (
            base_lat
            is None
        ):

            base_lat = (
                np.asarray(
                    lat_on
                )
            )

            base_lon = (
                np.asarray(
                    lon_on
                )
            )

        arrays.append(
            arr
        )

        levels_found.append(
            float(
                level
            )
        )

    if (
        len(
            arrays
        )
        < 4
    ):

        raise RuntimeError(
            f"Too few RRFS pressure "
            f"levels found for "
            f"{label}: "
            f"{levels_found}"
        )

    values = np.stack(
        arrays,
        axis=0
    )

    return xr.DataArray(
        values,
        dims=(
            "isobaricInhPa",
            "y",
            "x"
        ),
        coords={
            "isobaricInhPa":
                np.asarray(
                    levels_found,
                    dtype=float
                ),
            "latitude":
                (
                    (
                        "y",
                        "x"
                    ),
                    base_lat
                ),
            "longitude":
                (
                    (
                        "y",
                        "x"
                    ),
                    base_lon
                ),
        },
        name=short_name.lower()
    )


def rrfs_field(
    init_dt,
    product,
    search,
    label,
    preferred=None
):

    del preferred

    if product == "prs":

        field_map = {
            "pressure-level height":
                "HGT",
            "pressure-level temperature":
                "TMP",
            "pressure-level specific humidity":
                "SPFH",
            "pressure-level U wind":
                "UGRD",
            "pressure-level V wind":
                "VGRD",
        }

        if (
            label
            not in
            field_map
        ):

            raise RuntimeError(
                "Unsupported RRFS "
                f"pressure field: "
                f"{label}"
            )

        return (
            rrfs_pressure_stack(
                init_dt,
                FHR,
                field_map[
                    label
                ],
                label
            )
        )

    surface_map = {
        "2 m temperature": (
            [
                [
                    "TMP",
                    "2 m above ground"
                ],
                [
                    "TMP",
                    "2 m"
                ],
            ],
            "2dfld"
        ),

        "2 m specific humidity": (
            [
                [
                    "SPFH",
                    "2 m above ground"
                ],
                [
                    "SPFH",
                    "2 m"
                ],
            ],
            "2dfld"
        ),

        "surface pressure": (
            [
                [
                    "PRES",
                    "surface"
                ],
            ],
            "2dfld"
        ),

        "terrain height": (
            [
                [
                    "HGT",
                    "surface"
                ],
            ],
            "2dfld"
        ),

        "10 m U wind": (
            [
                [
                    "UGRD",
                    "10 m above ground"
                ],
                [
                    "UGRD",
                    "10 m"
                ],
            ],
            "2dfld"
        ),

        "10 m V wind": (
            [
                [
                    "VGRD",
                    "10 m above ground"
                ],
                [
                    "VGRD",
                    "10 m"
                ],
            ],
            "2dfld"
        ),
    }

    if (
        label
        not in
        surface_map
    ):

        raise RuntimeError(
            "Unsupported RRFS "
            f"surface field: "
            f"{label} | "
            f"{search}"
        )

    term_sets, rrfs_product = (
        surface_map[
            label
        ]
    )

    return (
        rrfs_idx_field(
            init_dt,
            FHR,
            term_sets,
            label,
            product=rrfs_product,
            required=True
        )
    )



def find_pressure_coord(
    da
):

    preferred = [
        "isobaricInhPa",
        "isobaricInPa",
        "pressure",
        "level",
    ]

    for name in (
        preferred
    ):

        if name in (
            da.coords
        ):

            return name

    for name in (
        da.dims
    ):

        lname = (
            name.lower()
        )

        if (
            "isobaric" in lname
            or
            "pressure" in lname
        ):

            return name

    raise RuntimeError(
        "Could not identify pressure "
        f"coordinate for {da.name}."
    )


def pressure_levels_hpa(
    da,
    pcoord
):

    levels = np.asarray(
        da[
            pcoord
        ].values,
        dtype=float
    )

    if (
        np.nanmax(
            levels
        )
        > 2000.0
    ):

        levels = (
            levels
            / 100.0
        )

    return levels


def select_pressure_levels(
    da,
    pcoord,
    levels_hpa
):

    raw = np.asarray(
        da[
            pcoord
        ].values,
        dtype=float
    )

    if (
        np.nanmax(
            raw
        )
        > 2000.0
    ):

        request = (
            np.asarray(
                levels_hpa,
                dtype=float
            )
            * 100.0
        )

    else:

        request = np.asarray(
            levels_hpa,
            dtype=float
        )

    return da.sel({
        pcoord: request
    })


def move_vertical_first(
    da,
    array,
    vertical_coord
):

    axis = (
        da.get_axis_num(
            vertical_coord
        )
    )

    if axis != 0:

        array = np.moveaxis(
            array,
            axis,
            0
        )

    return array


def subset_xy(
    da,
    yslice,
    xslice
):

    indexers = {}

    if "y" in (
        da.dims
    ):

        indexers[
            "y"
        ] = yslice

    if "x" in (
        da.dims
    ):

        indexers[
            "x"
        ] = xslice

    return da.isel(
        **indexers
    )


def get_xy_slice(
    da,
    extent,
    pad=2
):

    west, east, south, north = (
        extent
    )

    lat, lon = get_lat_lon(
        da
    )

    mask = (
        np.isfinite(
            lat
        )
        &
        np.isfinite(
            lon
        )
        &
        (
            lon >= west
        )
        &
        (
            lon <= east
        )
        &
        (
            lat >= south
        )
        &
        (
            lat <= north
        )
    )

    iy, ix = np.where(
        mask
    )

    if len(
        iy
    ) == 0:

        raise RuntimeError(
            "No RRFS grid points "
            "inside regional domain."
        )

    y0 = max(
        int(
            iy.min()
        )
        - pad,
        0
    )

    y1 = min(
        int(
            iy.max()
        )
        + pad + 1,
        lat.shape[
            0
        ]
    )

    x0 = max(
        int(
            ix.min()
        )
        - pad,
        0
    )

    x1 = min(
        int(
            ix.max()
        )
        + pad + 1,
        lon.shape[
            1
        ]
    )

    return (
        slice(
            y0,
            y1
        ),
        slice(
            x0,
            x1
        )
    )


def nearest_index(
    lat,
    lon,
    target_lat,
    target_lon
):

    d2 = (
        (
            lat
            - target_lat
        ) ** 2
        +
        (
            lon
            - target_lon
        ) ** 2
    )

    flat = np.nanargmin(
        d2
    )

    return np.unravel_index(
        flat,
        lat.shape
    )


# ============================================================
# BUILD ONE ORIGINAL-PETERS PROFILE
# ============================================================

def build_profile_at_point(
    data,
    j,
    i
):

    surface_p_hpa = (
        data[
            "ps"
        ][j, i]
        / 100.0
    )

    surface_t_k = (
        data[
            "t2"
        ][j, i]
    )

    surface_q = (
        data[
            "q2"
        ][j, i]
    )

    surface_z_msl = (
        data[
            "terrain"
        ][j, i]
    )

    surface_u = (
        data[
            "u10"
        ][j, i]
    )

    surface_v = (
        data[
            "v10"
        ][j, i]
    )

    p_hpa = (
        data[
            "plevels"
        ].copy()
    )

    z_msl = (
        data[
            "hgt"
        ][:, j, i]
        .astype(
            float
        )
    )

    t_k = (
        data[
            "tmp"
        ][:, j, i]
        .astype(
            float
        )
    )

    q = (
        data[
            "q"
        ][:, j, i]
        .astype(
            float
        )
    )

    u = (
        data[
            "u"
        ][:, j, i]
        .astype(
            float
        )
    )

    v = (
        data[
            "v"
        ][:, j, i]
        .astype(
            float
        )
    )

    surface_values = np.asarray(
        [
            surface_p_hpa,
            surface_t_k,
            surface_q,
            surface_z_msl,
            surface_u,
            surface_v,
        ]
    )

    if not np.all(
        np.isfinite(
            surface_values
        )
    ):

        return None

    good = (
        np.isfinite(
            p_hpa
        )
        &
        np.isfinite(
            z_msl
        )
        &
        np.isfinite(
            t_k
        )
        &
        np.isfinite(
            q
        )
        &
        np.isfinite(
            u
        )
        &
        np.isfinite(
            v
        )
        &
        (
            p_hpa
            < surface_p_hpa - 0.5
        )
        &
        (
            z_msl
            > surface_z_msl
        )
        &
        (
            t_k > 150.0
        )
        &
        (
            q >= 0.0
        )
    )

    if (
        np.count_nonzero(
            good
        )
        < 20
    ):

        return None

    p_hpa = (
        p_hpa[
            good
        ]
    )

    z_msl = (
        z_msl[
            good
        ]
    )

    t_k = (
        t_k[
            good
        ]
    )

    q = (
        q[
            good
        ]
    )

    u = (
        u[
            good
        ]
    )

    v = (
        v[
            good
        ]
    )

    # Add actual RRFS surface.
    p_hpa = np.concatenate([
        [
            surface_p_hpa
        ],
        p_hpa
    ])

    z_msl = np.concatenate([
        [
            surface_z_msl
        ],
        z_msl
    ])

    t_k = np.concatenate([
        [
            surface_t_k
        ],
        t_k
    ])

    q = np.concatenate([
        [
            surface_q
        ],
        q
    ])

    u = np.concatenate([
        [
            surface_u
        ],
        u
    ])

    v = np.concatenate([
        [
            surface_v
        ],
        v
    ])

    # Sort upward.
    order = np.argsort(
        z_msl
    )

    p_hpa = p_hpa[
        order
    ]

    z_msl = z_msl[
        order
    ]

    t_k = t_k[
        order
    ]

    q = q[
        order
    ]

    u = u[
        order
    ]

    v = v[
        order
    ]

    # Enforce monotonic vertical profile.
    keep = [
        0
    ]

    last_z = (
        z_msl[
            0
        ]
    )

    last_p = (
        p_hpa[
            0
        ]
    )

    for idx in range(
        1,
        len(
            z_msl
        )
    ):

        if (
            z_msl[
                idx
            ]
            > last_z + 0.1
            and
            p_hpa[
                idx
            ]
            < last_p - 0.1
        ):

            keep.append(
                idx
            )

            last_z = (
                z_msl[
                    idx
                ]
            )

            last_p = (
                p_hpa[
                    idx
                ]
            )

    keep = np.asarray(
        keep,
        dtype=int
    )

    p_hpa = p_hpa[
        keep
    ]

    z_msl = z_msl[
        keep
    ]

    t_k = t_k[
        keep
    ]

    q = q[
        keep
    ]

    u = u[
        keep
    ]

    v = v[
        keep
    ]

    z_agl = (
        z_msl
        - surface_z_msl
    )

    if (
        np.nanmax(
            z_agl
        )
        < 16000.0
    ):

        return None

    # --------------------------------------------------------
    # Original code uses 100-m grid 0-20 km.
    # --------------------------------------------------------

    z0 = np.arange(
        0.0,
        TOP_M,
        DZ_M,
        dtype=float
    )

    z_interp = (
        z0.copy()
    )

    z_interp[
        0
    ] = 1.0e-8

    try:

        T0 = interp1d(
            z_agl,
            t_k,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False
        )(
            z_interp
        )

        q0 = interp1d(
            z_agl,
            q,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False
        )(
            z_interp
        )

        u0 = interp1d(
            z_agl,
            u,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False
        )(
            z_interp
        )

        v0 = interp1d(
            z_agl,
            v,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False
        )(
            z_interp
        )

        p0 = np.exp(
            interp1d(
                z_agl,
                np.log(
                    p_hpa
                    * 100.0
                ),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False
            )(
                z_interp
            )
        )

    except Exception:

        return None

    q0 = np.clip(
        q0,
        0.0,
        0.05
    )

    if not (
        np.all(
            np.isfinite(
                T0
            )
        )
        and
        np.all(
            np.isfinite(
                p0
            )
        )
        and
        np.all(
            np.isfinite(
                q0
            )
        )
        and
        np.all(
            np.isfinite(
                u0
            )
        )
        and
        np.all(
            np.isfinite(
                v0
            )
        )
    ):

        return None

    return {
        "T0": T0,
        "p0": p0,
        "q0": q0,
        "z0": z0,
        "u0": u0,
        "v0": v0,
        "surface_z_msl": surface_z_msl,
        "source_p_hpa": p_hpa,
        "source_z_agl": z_agl,
        "source_t_k": t_k,
        "source_q": q,
        "source_u": u,
        "source_v": v,
    }


# ============================================================
# LCL-LFC MEAN ENVIRONMENTAL RELATIVE HUMIDITY
# ============================================================

def calculate_lcl_lfc_mean_rh(
    profile,
    lfc_m_agl
):
    """
    Calculate mean environmental RH (%) from the surface-based
    LCL to the original-Peters surface-based LFC.

    The profile is already on a uniform 100-m AGL grid, so a
    simple mean over the layer is also a height-weighted mean.
    """

    try:

        p0 = np.asarray(
            profile["p0"],
            dtype=float
        )

        T0 = np.asarray(
            profile["T0"],
            dtype=float
        )

        q0 = np.asarray(
            profile["q0"],
            dtype=float
        )

        z0 = np.asarray(
            profile["z0"],
            dtype=float
        )

        P = p0 * units.Pa
        T = T0 * units.K
        Q = q0 * units("kg/kg")

        Td = mpcalc.dewpoint_from_specific_humidity(
            P,
            T,
            Q
        )

        lcl_p, _ = mpcalc.lcl(
            P[0],
            T[0],
            Td[0]
        )

        lcl_p_pa = float(
            lcl_p.to("Pa").magnitude
        )

        # Pressure decreases with height, while np.interp expects
        # an increasing x-coordinate. Reverse both arrays.
        lcl_m_agl = float(
            np.interp(
                lcl_p_pa,
                p0[::-1],
                z0[::-1]
            )
        )

        if (
            not np.isfinite(lfc_m_agl)
            or
            not np.isfinite(lcl_m_agl)
        ):
            return np.nan, np.nan

        layer_bottom = min(
            lcl_m_agl,
            lfc_m_agl
        )

        layer_top = max(
            lcl_m_agl,
            lfc_m_agl
        )

        if layer_top <= layer_bottom:
            return np.nan, lcl_m_agl

        rh = mpcalc.relative_humidity_from_specific_humidity(
            P,
            T,
            Q
        ).to("dimensionless").magnitude * 100.0

        rh = np.clip(
            rh,
            0.0,
            100.0
        )

        mask = (
            np.isfinite(rh)
            &
            (z0 >= layer_bottom)
            &
            (z0 <= layer_top)
        )

        if np.count_nonzero(mask) < 1:
            return np.nan, lcl_m_agl

        mean_rh = float(
            np.nanmean(
                rh[mask]
            )
        )

        return mean_rh, lcl_m_agl

    except Exception:
        return np.nan, np.nan


# ============================================================
# ORIGINAL PETERS ECAPE CALCULATION
# ============================================================

def calculate_original_ecape(
    profile,
    return_details=False
):

    T0 = profile[
        "T0"
    ]

    p0 = profile[
        "p0"
    ]

    q0 = profile[
        "q0"
    ]

    z0 = profile[
        "z0"
    ]

    u0 = profile[
        "u0"
    ]

    v0 = profile[
        "v0"
    ]

    try:

        CAPE, CIN, LFC, EL = (
            compute_CAPE_AND_CIN(
                T0,
                p0,
                q0,
                0,
                0,
                0,
                z0,
                T1,
                T2
            )
        )

        if (
            not np.isfinite(
                CAPE
            )
            or
            CAPE <= 0.0
            or
            not np.isfinite(
                EL
            )
            or
            EL <= 0.0
        ):

            if return_details:
                return (
                    0.0,
                    None
                )

            return 0.0

        NCAPE, _, _ = (
            compute_NCAPE(
                T0,
                p0,
                q0,
                z0,
                T1,
                T2,
                LFC,
                EL
            )
        )

        V_SR, C_x, C_y = (
            compute_VSR(
                z0,
                u0,
                v0
            )
        )

        E_tilde, varepsilon, Radius = (
            compute_ETILDE(
                CAPE,
                NCAPE,
                V_SR,
                EL,
                MIXING_LENGTH_M
            )
        )

        ECAPE = (
            E_tilde
            * CAPE
        )

        if not np.isfinite(
            ECAPE
        ):

            ECAPE = np.nan

        # Mean environmental RH between the surface parcel
        # LCL and the original-Peters surface-based LFC.
        lcl_lfc_rh, LCL = calculate_lcl_lfc_mean_rh(
            profile,
            LFC
        )

        # --------------------------------------------------------
        # 0-1 KM BULK SHEAR VECTOR
        # --------------------------------------------------------

        if (
            np.nanmax(
                z0
            )
            >=
            1000.0
        ):

            u1 = np.interp(
                1000.0,
                z0,
                u0
            )

            v1 = np.interp(
                1000.0,
                z0,
                v0
            )

            shear01_u = (
                u1
                -
                u0[
                    0
                ]
            )

            shear01_v = (
                v1
                -
                v0[
                    0
                ]
            )

        else:

            shear01_u = np.nan
            shear01_v = np.nan


        # --------------------------------------------------------
        # 0-3 KM BULK SHEAR VECTOR
        # --------------------------------------------------------

        if (
            np.nanmax(
                z0
            )
            >=
            3000.0
        ):

            u3 = np.interp(
                3000.0,
                z0,
                u0
            )

            v3 = np.interp(
                3000.0,
                z0,
                v0
            )

            shear03_u = (
                u3
                -
                u0[
                    0
                ]
            )

            shear03_v = (
                v3
                -
                v0[
                    0
                ]
            )

        else:

            shear03_u = np.nan
            shear03_v = np.nan


        # --------------------------------------------------------
        # 0-6 KM BULK SHEAR VECTOR
        # --------------------------------------------------------

        if (
            np.nanmax(
                z0
            )
            >=
            6000.0
        ):

            u6 = np.interp(
                6000.0,
                z0,
                u0
            )

            v6 = np.interp(
                6000.0,
                z0,
                v0
            )

            shear06_u = (
                u6
                -
                u0[
                    0
                ]
            )

            shear06_v = (
                v6
                -
                v0[
                    0
                ]
            )

        else:

            shear06_u = np.nan
            shear06_v = np.nan


        if return_details:

            details = {
                "CAPE": CAPE,
                "CIN": CIN,
                "LFC": LFC,
                "EL": EL,
                "NCAPE": NCAPE,
                "V_SR": V_SR,
                "C_x": C_x,
                "C_y": C_y,
                "E_tilde": E_tilde,
                "varepsilon": varepsilon,
                "Radius": Radius,
                "LCL": LCL,
                "lcl_lfc_rh": lcl_lfc_rh,
                "shear01_u": shear01_u,
                "shear01_v": shear01_v,
                "shear03_u": shear03_u,
                "shear03_v": shear03_v,
                "shear06_u": shear06_u,
                "shear06_v": shear06_v,
            }

            return (
                ECAPE,
                details
            )

        return ECAPE

    except Exception:

        if return_details:

            return (
                np.nan,
                None
            )

        return np.nan





# ============================================================
# SITE FORECAST-HOUR PROCESSOR
# ============================================================

def process_forecast_hour(
    fhr,
    init_dt,
    cycle_date,
    cycle_hour,
    cycle_str
):

    global FHR
    global OUTDIR

    FHR = int(
        fhr
    )

    valid_dt = (
        init_dt
        +
        timedelta(
            hours=FHR
        )
    )

    OUTDIR = os.path.join(
        OUTDIR_BASE,
        cycle_str,
        "regional"
    )

    os.makedirs(
        OUTDIR,
        exist_ok=True
    )

    print("")
    print("=" * 72)
    print(
        f"PROCESSING RRFS ECAPE "
        f"{cycle_str} "
        f"F{FHR:03d}"
    )
    print(
        f"VALID: "
        f"{valid_dt:%Y-%m-%d %HZ}"
    )
    print("=" * 72)

    # LOAD RRFS FIELDS
    # ============================================================

    t2_da = rrfs_field(
        init_dt,
        "sfc",
        ":TMP:2 m",
        "2 m temperature",
        [
            "t2m",
            "t"
        ]
    )

    q2_da = rrfs_field(
        init_dt,
        "sfc",
        ":SPFH:2 m",
        "2 m specific humidity",
        [
            "sh2",
            "q",
            "spfh"
        ]
    )

    ps_da = rrfs_field(
        init_dt,
        "sfc",
        ":PRES:surface",
        "surface pressure",
        [
            "sp",
            "pres"
        ]
    )

    terrain_da = rrfs_field(
        init_dt,
        "sfc",
        ":HGT:surface",
        "terrain height",
        [
            "orog",
            "gh",
            "hgt"
        ]
    )

    u10_da = rrfs_field(
        init_dt,
        "sfc",
        ":UGRD:10 m",
        "10 m U wind",
        [
            "u10",
            "u"
        ]
    )

    v10_da = rrfs_field(
        init_dt,
        "sfc",
        ":VGRD:10 m",
        "10 m V wind",
        [
            "v10",
            "v"
        ]
    )

    hgt_da = rrfs_field(
        init_dt,
        "prs",
        r":HGT:[0-9]+ mb:",
        "pressure-level height",
        [
            "gh",
            "hgt"
        ]
    )

    tmp_da = rrfs_field(
        init_dt,
        "prs",
        r":TMP:[0-9]+ mb:",
        "pressure-level temperature",
        [
            "t",
            "tmp"
        ]
    )

    q_da = rrfs_field(
        init_dt,
        "prs",
        r":SPFH:[0-9]+ mb:",
        "pressure-level specific humidity",
        [
            "q",
            "spfh"
        ]
    )

    u_da = rrfs_field(
        init_dt,
        "prs",
        r":UGRD:[0-9]+ mb:",
        "pressure-level U wind",
        [
            "u",
            "ugrd"
        ]
    )

    v_da = rrfs_field(
        init_dt,
        "prs",
        r":VGRD:[0-9]+ mb:",
        "pressure-level V wind",
        [
            "v",
            "vgrd"
        ]
    )


    # ============================================================
    # COMMON PRESSURE LEVELS
    # ============================================================

    ph = find_pressure_coord(
        hgt_da
    )

    pt = find_pressure_coord(
        tmp_da
    )

    pq = find_pressure_coord(
        q_da
    )

    pu = find_pressure_coord(
        u_da
    )

    pv = find_pressure_coord(
        v_da
    )


    hgt_levels = pressure_levels_hpa(
        hgt_da,
        ph
    )

    tmp_levels = pressure_levels_hpa(
        tmp_da,
        pt
    )

    q_levels = pressure_levels_hpa(
        q_da,
        pq
    )

    u_levels = pressure_levels_hpa(
        u_da,
        pu
    )

    v_levels = pressure_levels_hpa(
        v_da,
        pv
    )


    common_levels = (
        hgt_levels.copy()
    )

    for levels in [
        tmp_levels,
        q_levels,
        u_levels,
        v_levels,
    ]:

        common_levels = np.intersect1d(
            common_levels,
            levels
        )


    common_levels = (
        common_levels[
            (
                common_levels
                <= 1000.0
            )
            &
            (
                common_levels
                >= 50.0
            )
        ]
    )

    common_levels = np.sort(
        common_levels
    )[::-1]


    print()
    print(
        "Common pressure levels:",
        len(
            common_levels
        )
    )


    hgt_da = select_pressure_levels(
        hgt_da,
        ph,
        common_levels
    )

    tmp_da = select_pressure_levels(
        tmp_da,
        pt,
        common_levels
    )

    q_da = select_pressure_levels(
        q_da,
        pq,
        common_levels
    )

    u_da = select_pressure_levels(
        u_da,
        pu,
        common_levels
    )

    v_da = select_pressure_levels(
        v_da,
        pv,
        common_levels
    )


    # ============================================================
    # SUBSET ALL DATA TO REGIONAL DOMAIN BEFORE LOADING ARRAYS
    # ============================================================

    yslice, xslice = get_xy_slice(
        t2_da,
        REGIONAL_EXTENT,
        pad=2
    )


    t2_da = subset_xy(
        t2_da,
        yslice,
        xslice
    )

    q2_da = subset_xy(
        q2_da,
        yslice,
        xslice
    )

    ps_da = subset_xy(
        ps_da,
        yslice,
        xslice
    )

    terrain_da = subset_xy(
        terrain_da,
        yslice,
        xslice
    )

    u10_da = subset_xy(
        u10_da,
        yslice,
        xslice
    )

    v10_da = subset_xy(
        v10_da,
        yslice,
        xslice
    )

    hgt_da = subset_xy(
        hgt_da,
        yslice,
        xslice
    )

    tmp_da = subset_xy(
        tmp_da,
        yslice,
        xslice
    )

    q_da = subset_xy(
        q_da,
        yslice,
        xslice
    )

    u_da = subset_xy(
        u_da,
        yslice,
        xslice
    )

    v_da = subset_xy(
        v_da,
        yslice,
        xslice
    )


    lat, lon = get_lat_lon(
        t2_da
    )


    # ============================================================
    # CONVERT ARRAYS
    # ============================================================

    def vertical_array(
        da,
        pcoord
    ):

        arr = np.asarray(
            da.values,
            dtype=float
        )

        return move_vertical_first(
            da,
            arr,
            pcoord
        )


    data = {
        "lat": lat,
        "lon": lon,

        "t2": np.asarray(
            t2_da.values,
            dtype=float
        ),

        "q2": np.asarray(
            q2_da.values,
            dtype=float
        ),

        "ps": np.asarray(
            ps_da.values,
            dtype=float
        ),

        "terrain": np.asarray(
            terrain_da.values,
            dtype=float
        ),

        "u10": np.asarray(
            u10_da.values,
            dtype=float
        ),

        "v10": np.asarray(
            v10_da.values,
            dtype=float
        ),

        "plevels": common_levels,

        "hgt": vertical_array(
            hgt_da,
            ph
        ),

        "tmp": vertical_array(
            tmp_da,
            pt
        ),

        "q": vertical_array(
            q_da,
            pq
        ),

        "u": vertical_array(
            u_da,
            pu
        ),

        "v": vertical_array(
            v_da,
            pv
        ),
    }


    # ============================================================
    # CALCULATE SAMPLED REGIONAL ECAPE GRID
    # ============================================================

    ny, nx = (
        lat.shape
    )

    sample_j = np.arange(
        0,
        ny,
        ECAPE_GRID_STRIDE
    )

    sample_i = np.arange(
        0,
        nx,
        ECAPE_GRID_STRIDE
    )


    calc_lat = lat[
        np.ix_(
            sample_j,
            sample_i
        )
    ]

    calc_lon = lon[
        np.ix_(
            sample_j,
            sample_i
        )
    ]


    ecape_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    lcl_lfc_rh_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear01_u_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear01_v_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear03_u_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear03_v_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear06_u_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )

    shear06_v_sample = np.full(
        calc_lat.shape,
        np.nan,
        dtype=float
    )


    total_profiles = (
        len(
            sample_j
        )
        *
        len(
            sample_i
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "CALCULATING REGIONAL ECAPE"
    )

    print(
        "=" * 72
    )

    print(
        "Sample stride:",
        ECAPE_GRID_STRIDE
    )

    print(
        "Profiles:",
        total_profiles
    )


    progress = tqdm(
        total=total_profiles,
        desc="Regional ECAPE"
    )


    for jj, j in enumerate(
        sample_j
    ):

        for ii, i in enumerate(
            sample_i
        ):

            profile = (
                build_profile_at_point(
                    data,
                    j,
                    i
                )
            )

            if profile is not None:

                ecape_value, details = (
                    calculate_original_ecape(
                        profile,
                        return_details=True
                    )
                )

                ecape_sample[
                    jj,
                    ii
                ] = (
                    ecape_value
                )

                if details is not None:

                    lcl_lfc_rh_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "lcl_lfc_rh"
                        ]
                    )

                    shear01_u_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear01_u"
                        ]
                    )

                    shear01_v_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear01_v"
                        ]
                    )

                    shear03_u_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear03_u"
                        ]
                    )

                    shear03_v_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear03_v"
                        ]
                    )

                    shear06_u_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear06_u"
                        ]
                    )

                    shear06_v_sample[
                        jj,
                        ii
                    ] = (
                        details[
                            "shear06_v"
                        ]
                    )

            progress.update(
                1
            )


    progress.close()


    finite = np.isfinite(
        ecape_sample
    )

    print()
    print(
        "Valid ECAPE profiles:",
        int(
            np.count_nonzero(
                finite
            )
        )
    )

    if np.any(
        finite
    ):

        print(
            "Maximum ECAPE:",
            f"{np.nanmax(ecape_sample):.0f} J/kg"
        )

        print(
            "Median ECAPE:",
            f"{np.nanmedian(ecape_sample):.0f} J/kg"
        )


    # ============================================================
    # INTERPOLATE ECAPE BACK TO FULL RRFS REGIONAL GRID
    # ============================================================

    points = np.column_stack(
        (
            calc_lon[
                finite
            ],
            calc_lat[
                finite
            ]
        )
    )

    values = (
        ecape_sample[
            finite
        ]
    )


    ecape_full = griddata(
        points,
        values,
        (
            lon,
            lat
        ),
        method="linear"
    )


    # Fill edges with nearest neighbor.
    missing = np.isnan(
        ecape_full
    )

    if np.any(
        missing
    ):

        nearest = griddata(
            points,
            values,
            (
                lon,
                lat
            ),
            method="nearest"
        )

        ecape_full = np.where(
            missing,
            nearest,
            ecape_full
        )


    # Very light cosmetic smoothing.
    ecape_full = gaussian_filter(
        ecape_full,
        sigma=ECAPE_SMOOTH_SIGMA
    )

    ecape_full = np.maximum(
        ecape_full,
        0.0
    )


    # ============================================================
    # INTERPOLATE MEAN LCL-LFC RH TO FULL RRFS REGIONAL GRID
    # ============================================================

    rh_finite = np.isfinite(
        lcl_lfc_rh_sample
    )

    if np.count_nonzero(
        rh_finite
    ) >= 3:

        rh_points = np.column_stack(
            (
                calc_lon[
                    rh_finite
                ],
                calc_lat[
                    rh_finite
                ]
            )
        )

        rh_values = lcl_lfc_rh_sample[
            rh_finite
        ]

        lcl_lfc_rh_full = griddata(
            rh_points,
            rh_values,
            (
                lon,
                lat
            ),
            method="linear"
        )

        rh_missing = np.isnan(
            lcl_lfc_rh_full
        )

        if np.any(
            rh_missing
        ):

            rh_nearest = griddata(
                rh_points,
                rh_values,
                (
                    lon,
                    lat
                ),
                method="nearest"
            )

            lcl_lfc_rh_full = np.where(
                rh_missing,
                rh_nearest,
                lcl_lfc_rh_full
            )

        # Light smoothing for contour readability only.
        lcl_lfc_rh_full = gaussian_filter(
            lcl_lfc_rh_full,
            sigma=0.55
        )

        lcl_lfc_rh_full = np.clip(
            lcl_lfc_rh_full,
            0.0,
            100.0
        )

    else:

        lcl_lfc_rh_full = np.full_like(
            ecape_full,
            np.nan,
            dtype=float
        )


    # ============================================================
    # PLOT REGIONAL MAP
    #
    # This map framework intentionally matches the other CAMS
    # products:
    #   * figsize=(14, 10)
    #   * regional site extent
    #   * state/county shapefiles
    #   * LBF CWA county emphasis + black/white CWA outline
    #   * NOAA/NWS logo + office label
    #   * creator credit
    #   * bottom colorbar via make_axes_locatable
    #
    # Meteorological fields remain:
    #   Fill:     Surface-Based ECAPE
    #   Contour:  0-1 km mean VSR
    #   Barbs:    0-3 km bulk shear vector
    # ============================================================

    plt.close(
        "all"
    )

    plt.rcParams[
        "contour.negative_linestyle"
    ] = "solid"

    fig = plt.figure(
        figsize=(
            14,
            10
        )
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        REGIONAL_EXTENT,
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        zorder=0
    )


    # ========================================================
    # ECAPE FILL
    # ========================================================

    pm = ax.contourf(
        lon,
        lat,
        ecape_full,
        levels=ECAPE_BOUNDS,
        cmap=ECAPE_CMAP,
        norm=ECAPE_NORM,
        extend="neither",
        transform=ccrs.PlateCarree(),
        zorder=5
    )


    # ========================================================
    # MEAN LCL-LFC RH CONTOURS
    # ========================================================

    if np.any(
        np.isfinite(
            lcl_lfc_rh_full
        )
    ):

        rh_contours = ax.contour(
            lon,
            lat,
            lcl_lfc_rh_full,
            levels=LCL_LFC_RH_LEVELS,
            colors=RH_COLORS,
            linewidths=0.9,
            linestyles="dashed",
            transform=ccrs.PlateCarree(),
            zorder=22
        )

        rh_labels = ax.clabel(
            rh_contours,
            levels=LCL_LFC_RH_LEVELS,
            inline=True,
            inline_spacing=6,
            fmt=lambda value: f"{int(value)}%",
            fontsize=8
        )

        for label in rh_labels:

            label.set_zorder(
                30
            )

            label.set_path_effects(
                [
                    pe.withStroke(
                        linewidth=3.0,
                        foreground="white"
                    )
                ]
            )


    # ========================================================
    # 0-6 KM BULK-SHEAR BARBS -- BLACK
    # ========================================================

    ax.barbs(
        calc_lon[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        calc_lat[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        shear06_u_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        shear06_v_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        length=5.2,
        linewidth=0.75,
        color="black",
        transform=ccrs.PlateCarree(),
        zorder=23
    )


    # ========================================================
    # 0-3 KM BULK-SHEAR BARBS -- RED
    #
    # Plot on the same sampled locations, but slightly smaller
    # than the 0-6 km vectors so both layers remain readable.
    # ========================================================

    ax.barbs(
        calc_lon[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        calc_lat[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        shear03_u_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        shear03_v_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        length=4.5,
        linewidth=0.75,
        color="red",
        transform=ccrs.PlateCarree(),
        zorder=24
    )


    # ========================================================
    # 0-1 KM BULK-SHEAR BARBS -- BLUE
    #
    # ========================================================

    ax.barbs(
        calc_lon[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        calc_lat[
            ::BARB_SKIP,
            ::BARB_SKIP
        ],
        shear01_u_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        shear01_v_sample[
            ::BARB_SKIP,
            ::BARB_SKIP
        ]
        *
        1.94384,
        length=4.0,
        linewidth=0.70,
        color="blue",
        transform=ccrs.PlateCarree(),
        zorder=25
    )


    # ========================================================
    # STATES / COUNTIES
    #
    # Same appearance as the other site products.
    # ========================================================

    add_shapefile_outline(
        ax,
        STATE_SHP,
        REGIONAL_EXTENT,
        edgecolor="black",
        linewidth=1.4,
        zorder=13
    )

    add_shapefile_outline(
        ax,
        COUNTY_SHP,
        REGIONAL_EXTENT,
        edgecolor="lightgray",
        linewidth=0.35,
        zorder=12
    )


    # ========================================================
    # LBF CWA
    # ========================================================

    if LBF_CWA_GEOM is not None:

        add_counties_clipped_to_cwa(
            ax,
            COUNTY_SHP,
            LBF_CWA_GEOM,
            linewidth=1.0,
            edgecolor="black",
            zorder=13
        )

        # Black outer border.
        ax.add_geometries(
            [
                LBF_CWA_GEOM
            ],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=3.5,
            zorder=14
        )

        # White inner border.
        ax.add_geometries(
            [
                LBF_CWA_GEOM
            ],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="white",
            linewidth=1.8,
            zorder=15
        )


    # ========================================================
    # TITLES
    # ========================================================


    valid_title = (
        f"F{FHR:03d} Valid: "
        f"{valid_dt:%a %Y-%m-%d %Hz}"
    )

    init_title = (
        f"Init: "
        f"{init_dt:%a %Y-%m-%d %Hz} RRFS"
    )

    # Colored main title: 0-3 km shear is red
    title_y = 1.042

    title1 = ax.text(
        0.0,
        title_y,
        "RRFS | Surface-Based ECAPE, LCL-LFC RH, 0-1 km + ",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    bbox1 = title1.get_window_extent(renderer=renderer)
    x2 = (bbox1.x1 - axes_bbox.x0) / axes_bbox.width

    title2 = ax.text(
        x2,
        title_y,
        "0-3 km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="red"
    )

    fig.canvas.draw()
    bbox2 = title2.get_window_extent(renderer=renderer)
    x3 = (bbox2.x1 - axes_bbox.x0) / axes_bbox.width

    ax.text(
        x3,
        title_y,
        " + 0-6 km Bulk Shear",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

    ax.text(
        0.0,
        1.005,
        valid_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        1.0,
        1.005,
        init_title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )


    # ========================================================
    # COLORBAR
    #
    # Same bottom placement as the other site maps.
    # Fixed labels and no max-extension arrow.
    # ========================================================

    divider = make_axes_locatable(
        ax
    )

    cax = divider.append_axes(
        "bottom",
        size="3%",
        pad=0.25,
        axes_class=plt.Axes
    )

    cbar = plt.colorbar(
        pm,
        cax=cax,
        orientation="horizontal",
        ticks=ECAPE_TICKS,
        drawedges=True,
        extend="neither"
    )

    cbar.set_label(
        "Surface-Based ECAPE (J kg$^{-1}$)",
        fontsize=10,
        weight="bold"
    )

    cbar.ax.xaxis.set_label_position(
        "top"
    )

    cbar.ax.tick_params(
        axis="x",
        which="both",
        labelsize=8,
        length=0
    )


    # ========================================================
    # NOAA / NWS LOGO
    # ========================================================

    if os.path.exists(
        LOGO_PATH
    ):

        logo = mpimg.imread(
            LOGO_PATH
        )

        logo_ax = ax.inset_axes(
            [
                0.82,
                0.84,
                0.165,
                0.155
            ],
            transform=ax.transAxes,
            zorder=50
        )

        logo_ax.imshow(
            logo
        )

        logo_ax.axis(
            "off"
        )


    # ========================================================
    # OFFICE LABEL
    # ========================================================

    ax.text(
        0.902,
        0.835,
        "NWS North Platte, NE",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=51,
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="white"
            )
        ]
    )


    # ========================================================
    # CREATOR CREDIT
    # ========================================================

    ax.text(
        0.01,
        0.015,
        "Plot created by: Matthew Labenz",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        color="black",
        zorder=40,
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="white"
            )
        ]
    )


    map_path = os.path.join(
        OUTDIR,
        (
            f"rrfs_regional_"
            f"f{FHR:03d}.png"
        )
    )

    plt.savefig(
        map_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print()
    print(
        "Saved map:",
        map_path
    )

    remote_key = (
        f"{R2_PRODUCT_PATH}/"
        f"{cycle_str}/"
        f"regional/"
        f"{os.path.basename(map_path)}"
    )

    upload_to_r2(
        map_path,
        remote_key
    )


    # ============================================================

# ============================================================
# MAIN
# ============================================================

init_dt = (
    find_latest_available_rrfs_cycle()
)

cycle_date = (
    init_dt.strftime(
        "%Y%m%d"
    )
)

cycle_hour = (
    init_dt.hour
)

cycle_str = (
    f"{cycle_date}_"
    f"{cycle_hour:02d}z"
)

if (
    cycle_hour
    in
    LONG_CYCLE_HOURS
):

    MAX_FHR = (
        MAX_FHR_LONG
    )

else:

    MAX_FHR = (
        MAX_FHR_SHORT
    )

upload_runs_json(
    init_dt,
    cycle_str,
    MAX_FHR
)

successful_fhrs = []
failed_fhrs = []

for fhr in range(
    START_FHR,
    MAX_FHR + 1
):

    success = False
    last_error = None

    for attempt in range(
        1,
        FHR_ATTEMPTS + 1
    ):

        try:

            process_forecast_hour(
                fhr,
                init_dt,
                cycle_date,
                cycle_hour,
                cycle_str
            )

            successful_fhrs.append(
                fhr
            )

            success = True

            break

        except Exception as e:

            last_error = e

            print(
                f"F{fhr:03d} attempt "
                f"{attempt}/"
                f"{FHR_ATTEMPTS} failed: "
                f"{e}"
            )

            plt.close(
                "all"
            )

            if (
                attempt
                <
                FHR_ATTEMPTS
            ):

                time.sleep(
                    FHR_RETRY_SECONDS
                )

    if not success:

        failed_fhrs.append(
            (
                fhr,
                str(
                    last_error
                )
            )
        )

        print(
            f"Skipping RRFS ECAPE "
            f"F{fhr:03d} after "
            f"{FHR_ATTEMPTS} attempts."
        )


print("")
print("=" * 72)
print("RRFS ECAPE SITE PROCESSING SUMMARY")
print("=" * 72)

print(
    "Successful:",
    ", ".join(
        f"F{x:03d}"
        for x
        in successful_fhrs
    )
    if successful_fhrs
    else
    "none"
)

if failed_fhrs:

    print("Failed:")

    for (
        fhr,
        error
    ) in failed_fhrs:

        print(
            f"  F{fhr:03d}: "
            f"{error}"
        )

else:

    print(
        "Failed: none"
    )

print(
    "R2 product path:",
    R2_PRODUCT_PATH
)

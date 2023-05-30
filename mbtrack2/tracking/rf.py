# -*- coding: utf-8 -*-
"""
This module handles radio-frequency (RF) cavitiy elements. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from mbtrack2.tracking.element import Element

class RFCavity(Element):
    """
    Perfect RF cavity class for main and harmonic RF cavities.
    Use cosine definition.
    
    Parameters
    ----------
    ring : Synchrotron object
    m : int
        Harmonic number of the cavity
    Vc : float
        Amplitude of cavity voltage [V]
    theta : float
        Phase of Cavity voltage
    """
    def __init__(self, ring, m, Vc, theta):
        self.ring = ring
        self.m = m 
        self.Vc = Vc
        self.theta = theta
        
    @Element.parallel    
    def track(self,bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        bunch["delta"] += self.Vc / self.ring.E0 * np.cos(
                self.m * self.ring.omega1 * bunch["tau"] + self.theta )
        
    def value(self, val):
        return self.Vc / self.ring.E0 * np.cos( 
                self.m * self.ring.omega1 * val + self.theta )
    
    
class CavityResonator():
    """Cavity resonator class for active or passive RF cavity with beam
    loading or HOM, based on [1,2].
    
    Use cosine definition.
    
    If used with mpi, beam.mpi.share_distributions must be called before the 
    track method call.
    
    Parameters
    ----------
    ring : Synchrotron object
    m : int or float
        Harmonic number of the cavity.
    Rs : float
        Shunt impedance of the cavities in [Ohm], defined as 0.5*Vc*Vc/Pc.
        If Ncav = 1, used for the total shunt impedance.
        If Ncav > 1, used for the shunt impedance per cavity.
    Q : float
        Quality factor of the cavity.
    QL : float
        Loaded quality factor of the cavity.
    detune : float
        Detuing of the cavity in [Hz], defined as (fr - m*ring.f1).
    Ncav : int, optional
        Number of cavities.
    Vc : float, optinal
        Total cavity voltage in [V].
    theta : float, optional
        Total cavity phase in [rad].
    n_bin : int, optional
        Number of bins used for the beam loading computation. 
        Only used if MPI is not used, otherwise n_bin must be specified in the 
        beam.mpi.share_distributions method.
        The default is 75.
        
    Attributes
    ----------
    beam_phasor : complex
        Beam phasor in [V].
    beam_phasor_record : array of complex
        Last beam phasor value of each bunch in [V].
    generator_phasor : complex
        Generator phasor in [V].
    cavity_phasor : complex
        Cavity phasor in [V].
    cavity_phasor_record : array of complex
        Last cavity phasor value of each bunch in [V].
    cavity_voltage : float
        Cavity total voltage in [V].
    cavity_phase : float
        Cavity total phase in [rad].
    loss_factor : float
        Cavity loss factor in [V/C].
    Rs_per_cavity : float
        Shunt impedance of a single cavity in [Ohm], defined as 0.5*Vc*Vc/Pc.
    beta : float
        Coupling coefficient of the cavity.
    fr : float
        Resonance frequency of the cavity in [Hz].
    wr : float
        Angular resonance frequency in [Hz.rad].
    psi : float
        Tuning angle in [rad].
    filling_time : float
        Cavity filling time in [s].
    Pc : float
        Power dissipated in the cavity walls in [W].
    Pg : float
        Generator power in [W].
    Vgr : float
        Generator voltage at resonance in [V].
    theta_gr : float
        Generator phase at resonance in [rad].
    Vg : float
        Generator voltage in [V].
    theta_g : float
        Generator phase in [rad].
    tracking : bool
        True if the tracking has been initialized.
    bunch_index : int
        Number of the tracked bunch in the current core.
    distance : array
        Distance between bunches.
    valid_bunch_index : array
    
    Methods
    -------
    Vbr(I0)
        Return beam voltage at resonance in [V].
    Vb(I0)
        Return beam voltage in [V].
    Pb(I0)
        Return power transmitted to the beam in [W].
    Pr(I0)
        Return power reflected back to the generator in [W].
    Z(f)
        Cavity impedance in [Ohm] for a given frequency f in [Hz].
    set_optimal_coupling(I0)
        Set coupling to optimal value.
    set_optimal_detune(I0)
        Set detuning to optimal conditions.
    set_generator(I0)
        Set generator parameters.
    plot_phasor(I0)
        Plot phasor diagram.
    is_DC_Robinson_stable(I0)
        Check DC Robinson stability.
    plot_DC_Robinson_stability()
        Plot DC Robinson stability limit.
    init_tracking(beam)
        Initialization of the tracking.
    track(beam)
        Tracking method.
    phasor_decay(time)
        Compute the beam phasor decay during a given time span.
    phasor_evol(profile, bin_length, charge_per_mp)
        Compute the beam phasor evolution during the crossing of a bunch.
    VRF(z, I0)
        Return the total RF voltage.
    dVRF(z, I0)
        Return derivative of total RF voltage.
    ddVRF(z, I0)
        Return the second derivative of total RF voltage.
    deltaVRF(z, I0)
        Return the generator voltage minus beam loading voltage.
    init_FB(gain_A, gain_P, delay)
        Initialize and switch on amplitude and phase feedback.
    track_FB()
        Tracking method for the amplitude and phase feedback.
    
    References
    ----------
    [1] Wilson, P. B. (1994). Fundamental-mode rf design in e+ e− storage ring 
    factories. In Frontiers of Particle Beams: Factories with e+ e-Rings 
    (pp. 293-311). Springer, Berlin, Heidelberg.
    
    [2] Yamamoto, Naoto, Alexis Gamelin, and Ryutaro Nagaoka. "Investigation 
    of Longitudinal Beam Dynamics With Harmonic Cavities by Using the Code 
    Mbtrack." IPAC’19, Melbourne, Australia, 2019.
    """
    def __init__(self, ring, m, Rs, Q, QL, detune, Ncav=1, Vc=0, theta=0, 
                 n_bin=75):
        self.ring = ring
        self.m = m
        self.Ncav = Ncav
        if Ncav != 1:
            self.Rs_per_cavity = Rs
        else:
            self.Rs = Rs
        self.Q = Q
        self.QL = QL
        self.detune = detune
        self.Vc = Vc
        self.theta = theta
        self.beam_phasor = np.zeros(1, dtype=np.complex)
        self.beam_phasor_record = np.zeros((self.ring.h), dtype=np.complex)
        self.tracking = False
        self.Vg = 0
        self.theta_g = 0
        self.Vgr = 0
        self.theta_gr = 0
        self.Pg = 0
        self.generator_phasor_record = np.zeros((self.ring.h), dtype=np.complex)
        self.ig_phasor_record = np.zeros((self.ring.h), dtype=np.complex)
        self.DirectFB_ig_phasor = np.zeros((self.ring.h), dtype=np.complex)
        self._Ig2Vg_mat = np.zeros((self.ring.h,self.ring.h),dtype=np.complex)
        self.n_bin = int(n_bin)
        self.FB = False
        self._PIctrlVcLoop = False
        self._DirectFB = False
        self._TunerLoop = False
        self._Ig2Vg_switch = False
        
    def init_tracking(self, beam):
        """
        Initialization of the tracking.

        Parameters
        ----------
        beam : Beam object

        """
        if beam.mpi_switch:
            self.bunch_index = beam.mpi.bunch_num # Number of the tracked bunch in this processor
            
        self.distance = beam.distance_between_bunches
        self.valid_bunch_index = beam.bunch_index
        self.tracking = True
        self.nturn = 0
    
    def track(self, beam):
        """
        Track a Beam object through the CavityResonator object.
        
        Can be used with or without mpi.
        If used with mpi, beam.mpi.share_distributions must be called before.
        
        The beam phasor is given at t=0 (synchronous particle) of the first 
        non empty bunch.

        Parameters
        ----------
        beam : Beam object

        """
        
        if self.tracking is False:
            self.init_tracking(beam)
        
        for index, bunch in enumerate(beam):
            
            if beam.filling_pattern[index]:
                
                if beam.mpi_switch:
                    # get rank of bunch n° index
                    rank = beam.mpi.bunch_to_rank(index)
                    # mpi -> get shared bunch profile for current bunch
                    center = beam.mpi.tau_center[rank]
                    profile = beam.mpi.tau_profile[rank]
                    bin_length = center[1]-center[0]
                    charge_per_mp = beam.mpi.charge_per_mp_all[rank]
                    if index == self.bunch_index:
                        sorted_index = beam.mpi.tau_sorted_index
                else:
                    # no mpi -> get bunch profile for current bunch
                    if len(bunch) != 0:
                        (bins, sorted_index, profile, center) = bunch.binning(n_bin=self.n_bin)
                        bin_length = center[1]-center[0]
                        charge_per_mp = bunch.charge_per_mp
                        self.bunch_index = index
                    else:
                        # Update filling pattern
                        beam.update_filling_pattern()
                        beam.update_distance_between_bunches()
                        # save beam phasor value
                        self.beam_phasor_record[index] = self.beam_phasor
                        # phasor decay to be at t=0 of the next bunch
                        self.phasor_decay(self.ring.T1, ref_frame="beam")
                        continue
                
                energy_change = bunch["tau"]*0
                
                # remove part of beam phasor decay to be at the start of the binning (=bins[0])
                self.phasor_decay(center[0] - bin_length/2, ref_frame="beam")
                
                if index != self.bunch_index:
                    self.phasor_evol(profile, bin_length, charge_per_mp, ref_frame="beam")
                else:
                    # modify beam phasor
                    for i, center0 in enumerate(center):
                        mp_per_bin = profile[i]
                        
                        if mp_per_bin == 0:
                            self.phasor_decay(bin_length, ref_frame="beam")
                            continue
                        
                        ind = (sorted_index == i)
                        phase = self.m * self.ring.omega1 * (center0 + self.ring.T1* (index + self.ring.h * self.nturn))
                        #Since 2023/02, the generator_phasor_record is used instead of Vg and theta_G.
                        #    Vgene = self.Vg*np.cos(phase + self.theta_g)
                        Vgene = (self.generator_phasor_record[index]*np.exp(1j*phase)).real
                        Vbeam = np.real(self.beam_phasor)
                        Vtot = Vgene + Vbeam - charge_per_mp*self.loss_factor*mp_per_bin
                        energy_change[ind] = Vtot / self.ring.E0
    
                        self.beam_phasor -= 2*charge_per_mp*self.loss_factor*mp_per_bin
                        self.phasor_decay(bin_length, ref_frame="beam")
                
                # phasor decay to be at t=0 of the current bunch (=-1*bins[-1])
                self.phasor_decay(-1 * (center[-1] + bin_length/2), ref_frame="beam")
                
                if index == self.bunch_index:
                    # apply kick
                    bunch["delta"] += energy_change
            
            # save beam phasor value
            self.beam_phasor_record[index] = self.beam_phasor
            
            if self.FB and (self.FB_index == index):
                self.track_FB()

            # phasor decay to be at t=0 of the next bunch
            self.phasor_decay(self.ring.T1, ref_frame="beam")

        ## Processes for the function which asuumes Digital low level RF system
        # Calculate and refresh ig compoment from PI control Vc loop
        if self._Ig2Vg_switch:
            if self._PIctrlVcLoop:
                self.track_PIctrlVcLoop()
        # Calculate ig compoment from Direct FB loop
            if self._DirectFB:
                self.track_DirectFB()
        # After all digital process, convert Ig to Vg 
            self._Ig2Vg()

        # Tuner loop, which changes the detuning frequency of the cavity.
        if self._TunerLoop:
            self.track_TunerLoop()
                
        self.nturn += 1
        
    def init_phasor_track(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using a
        tracking like method.
        
        Follow the same steps as the track method but in the "rf" reference 
        frame and without any modifications on the beam.

        Parameters
        ----------
        beam : Beam object

        """        
        if self.tracking is False:
            self.init_tracking(beam)
            
        n_turn = int(self.filling_time/self.ring.T0*10)
        
        for i in range(n_turn):
            for j, bunch in enumerate(beam.not_empty):
                
                index = self.valid_bunch_index[j]
                
                if beam.mpi_switch:
                    # get shared bunch profile for current bunch
                    center = beam.mpi.tau_center[j]
                    profile = beam.mpi.tau_profile[j]
                    bin_length = center[1]-center[0]
                    charge_per_mp = beam.mpi.charge_per_mp_all[j]
                else:
                    if i == 0:
                        # get bunch profile for current bunch
                        (bins, sorted_index, profile, center) = bunch.binning(n_bin=self.n_bin)
                        if j == 0:
                            self.profile_save = np.zeros((len(beam),len(profile),))
                            self.center_save = np.zeros((len(beam),len(center),))
                        self.profile_save[j,:] = profile
                        self.center_save[j,:] = center
                    else:
                        profile = self.profile_save[j,:]
                        center = self.center_save[j,:]
                        
                    bin_length = center[1]-center[0]
                    charge_per_mp = bunch.charge_per_mp
                
                self.phasor_decay(center[0] - bin_length/2, ref_frame="rf")
                self.phasor_evol(profile, bin_length, charge_per_mp, ref_frame="rf")
                self.phasor_decay(-1 * (center[-1] + bin_length/2), ref_frame="rf")
                self.phasor_decay( (self.distance[index] * self.ring.T1), ref_frame="rf")
            
    def phasor_decay(self, time, ref_frame="beam"):
        """
        Compute the beam phasor decay during a given time span, assuming that 
        no particles are crossing the cavity during the time span.

        Parameters
        ----------
        time : float
            Time span in [s], can be positive or negative.
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".

        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m*self.ring.omega1)
        self.beam_phasor = self.beam_phasor * np.exp((-1/self.filling_time +
                                  1j*delta)*time)
        
    def phasor_evol(self, profile, bin_length, charge_per_mp, ref_frame="beam"):
        """
        Compute the beam phasor evolution during the crossing of a bunch using 
        an analytic formula [1].
        
        Assume that the phasor decay happens before the beam loading.

        Parameters
        ----------
        profile : array
            Longitudinal profile of the bunch in [number of macro-particle].
        bin_length : float
            Length of a bin in [s].
        charge_per_mp : float
            Charge per macro-particle in [C].
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".
            
        References
        ----------
        [1] mbtrack2 manual.
            
        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m*self.ring.omega1)
            
        n_bin = len(profile)
        
        # Phasor decay during crossing time
        deltaT = n_bin*bin_length
        self.phasor_decay(deltaT, ref_frame)
        
        # Phasor evolution due to induced voltage by marco-particles
        k = np.arange(0, n_bin)
        var = np.exp( (-1/self.filling_time + 1j*delta) * 
                      (n_bin-k) * bin_length )
        sum_tot = np.sum(profile * var)
        sum_val = -2 * sum_tot * charge_per_mp * self.loss_factor
        self.beam_phasor += sum_val
        
    def init_phasor(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using an
        analytic formula [1].
        
        No modifications on the Beam object.

        Parameters
        ----------
        beam : Beam object
            
        References
        ----------
        [1] mbtrack2 manual.

        """
        
        # Initialization
        if self.tracking is False:
            self.init_tracking(beam)
        
        N = self.n_bin - 1
        delta = (self.wr - self.m*self.ring.omega1)
        n_turn = int(self.filling_time/self.ring.T0*10)
        
        T = np.ones(self.ring.h)*self.ring.T1
        bin_length = np.zeros(self.ring.h)
        charge_per_mp = np.zeros(self.ring.h)
        profile = np.zeros((N, self.ring.h))
        center = np.zeros((N, self.ring.h))
        
        # Gather beam distribution data
        for j, bunch in enumerate(beam.not_empty):
            index = self.valid_bunch_index[j]
            if beam.mpi_switch:
                beam.mpi.share_distributions(beam, n_bin=self.n_bin)
                center[:,index] = beam.mpi.tau_center[j]
                profile[:,index] = beam.mpi.tau_profile[j]
                bin_length[index] = center[1, index]-center[0, index]
                charge_per_mp[index] = beam.mpi.charge_per_mp_all[j]
            else:
                (bins, sorted_index, profile[:, index], center[:, index]) = bunch.binning(n_bin=self.n_bin)
                bin_length[index] = center[1, index]-center[0, index]
                charge_per_mp[index] = bunch.charge_per_mp
            T[index] -= (center[-1, index] + bin_length[index]/2)
            if index != 0:
                T[index - 1] += (center[0, index] - bin_length[index]/2)
        T[self.ring.h - 1] += (center[0, 0] - bin_length[0]/2)

        # Compute matrix coefficients
        k = np.arange(0, N)
        Tkj = np.zeros((N, self.ring.h))
        for j in range(self.ring.h):
            sum_t = np.array([T[n] + N*bin_length[n] for n in range(j+1,self.ring.h)])
            Tkj[:,j] = (N-k)*bin_length[j] + T[j] + np.sum(sum_t)
            
        var = np.exp( (-1/self.filling_time + 1j*delta) * Tkj )
        sum_tot = np.sum((profile*charge_per_mp) * var)
        
        # Use the formula n_turn times
        for i in range(n_turn):
            # Phasor decay during one turn
            self.phasor_decay(self.ring.T0, ref_frame="rf")
            # Phasor evolution due to induced voltage by marco-particles during one turn
            sum_val = -2 * sum_tot * self.loss_factor
            self.beam_phasor += sum_val
        
        # Replace phasor at t=0 (synchronous particle) of the first non empty bunch.
        idx0 = self.valid_bunch_index[0]
        self.phasor_decay(center[-1,idx0] + bin_length[idx0]/2, ref_frame="rf")
    
    @property
    def generator_phasor(self):
        """Generator phasor in [V]"""
        return self.Vg*np.exp(1j*self.theta_g)
    
    @property
    def cavity_phasor(self):
        """Cavity total phasor in [V]"""
        return self.generator_phasor + self.beam_phasor
    
    @property
    def cavity_phasor_record(self):
        """Last cavity phasor value of each bunch in [V]"""
        #if self._PIctrlVcLoop:
        return self.generator_phasor_record + self.beam_phasor_record
        #else:
        #    return self.generator_phasor + self.beam_phasor_record
    
    @property
    def cavity_voltage(self):
        """Cavity total voltage in [V]"""
        return np.abs(self.cavity_phasor)
    
    @property
    def cavity_phase(self):
        """Cavity total phase in [rad]"""
        return np.angle(self.cavity_phasor)
    
    @property
    def beam_voltage(self):
        """Beam loading voltage in [V]"""
        return np.abs(self.beam_phasor)
    
    @property
    def beam_phase(self):
        """Beam loading phase in [rad]"""
        return np.angle(self.beam_phasor)
    
    @property
    def loss_factor(self):
        """Cavity loss factor in [V/C]"""
        return self.wr*self.Rs/(2 * self.Q)

    @property
    def m(self):
        """Harmonic number of the cavity"""
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        
    @property
    def Ncav(self):
        """Number of cavities"""
        return self._Ncav

    @Ncav.setter
    def Ncav(self, value):
        self._Ncav = value
        
    @property
    def Rs_per_cavity(self):
        """Shunt impedance of a single cavity in [Ohm], defined as 
        0.5*Vc*Vc/Pc."""
        return self._Rs_per_cavity

    @Rs_per_cavity.setter
    def Rs_per_cavity(self, value):
        self._Rs_per_cavity = value

    @property
    def Rs(self):
        """Shunt impedance [ohm]"""
        return self.Rs_per_cavity * self.Ncav

    @Rs.setter
    def Rs(self, value):
        self.Rs_per_cavity = value / self.Ncav
        
    @property
    def RL(self):
        """Loaded shunt impedance [ohm]"""
        return self.Rs / (1 + self.beta)

    @property
    def Q(self):
        """Quality factor"""
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def QL(self):
        """Loaded quality factor"""
        return self._QL

    @QL.setter
    def QL(self, value):
        self._QL = value
        self._beta = self.Q/self.QL - 1

    @property
    def beta(self):
        """Coupling coefficient"""
        return self._beta

    @beta.setter
    def beta(self, value):
        self.QL = self.Q/(1 + value)

    @property
    def detune(self):
        """Cavity detuning [Hz] - defined as (fr - m*f1)"""
        return self._detune

    @detune.setter
    def detune(self, value):
        self._detune = value
        self._fr = self.detune + self.m*self.ring.f1
        self._wr = self.fr*2*np.pi
        self._psi = np.arctan(self.QL*(self.fr/(self.m*self.ring.f1) -
                                       (self.m*self.ring.f1)/self.fr))

    @property
    def fr(self):
        """Resonance frequency of the cavity in [Hz]"""
        return self._fr

    @fr.setter
    def fr(self, value):
        self.detune = value - self.m*self.ring.f1

    @property
    def wr(self):
        """Angular resonance frequency in [Hz.rad]"""
        return self._wr

    @wr.setter
    def wr(self, value):
        self.detune = (value - self.m*self.ring.f1)*2*np.pi

    @property
    def psi(self):
        """Tuning angle in [rad]"""
        return self._psi

    @psi.setter
    def psi(self, value):
        delta = (self.ring.f1*self.m*np.tan(value)/self.QL)**2 + 4*(self.ring.f1*self.m)**2
        fr = (self.ring.f1*self.m*np.tan(value)/self.QL + np.sqrt(delta))/2
        self.detune = fr - self.m*self.ring.f1
        self._init_Ig2Vg()
        
    @property
    def filling_time(self):
        """Cavity filling time in [s]"""
        return 2*self.QL/self.wr
    
    @property
    def Pc(self):
        """Power dissipated in the cavity walls in [W]"""
        return self.Vc**2 / (2 * self.Rs)
    
    @property
    def Pg_record(self):
        """
        Return Pg from Ig

        Eq.27 of N.Yamamoto et al., PRAB 21, 012001 (2018)
        """
        return self.Rs/8/self.beta*np.abs(self.ig_phasor_record)**2
    
    def Pb(self, I0):
        """
        Return power transmitted to the beam in [W] - near Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power transmitted to the beam in [W].

        """
        return I0 * self.Vc * np.cos(self.theta)
    
    def Pr(self, I0):
        """
        Power reflected back to the generator in [W].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power reflected back to the generator in [W].

        """
        return self.Pg - self.Pb(I0) - self.Pc

    def Vbr(self, I0):
        """
        Return beam voltage at resonance in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage at resonance in [V].

        """
        return 2*I0*self.Rs/(1+self.beta)
    
    def Vb(self, I0):
        """
        Return beam voltage in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage in [V].

        """
        return self.Vbr(I0)*np.cos(self.psi)
    
    def Z(self, f):
        """Cavity impedance in [Ohm] for a given frequency f in [Hz]"""
        #return self.Rs/(1 + 1j*self.QL*(self.fr/f - f/self.fr))
        return self.Rs/self.Q*self.QL/(1 + 1j*self.QL*(self.fr/f - f/self.fr))
    
    def set_optimal_detune(self, I0):
        """
        Set detuning to optimal conditions - second Eq. (4.2.1) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.psi = np.arctan(-self.Vbr(I0)/self.Vc*np.sin(self.theta))
        
    def set_optimal_coupling(self, I0):
        """
        Set coupling to optimal value - Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.beta = 1 + (2 * I0 * self.Rs * np.cos(self.theta) / 
                         self.Vc)
                
    def set_generator(self, I0):
        """
        Set generator parameters (Pg, Vgr, theta_gr, Vg and theta_g) for a 
        given current and set of parameters.

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        
        # Generator power [W] - Eq. (4.1.2) [1] corrected with factor (1+beta)**2 instead of (1+beta**2)
        self.Pg = self.Vc**2*(1+self.beta)**2/(2*self.Rs*4*self.beta*np.cos(self.psi)**2)*(
            (np.cos(self.theta) + 2*I0*self.Rs/(self.Vc*(1+self.beta))*np.cos(self.psi)**2 )**2 + 
            (np.sin(self.theta) + 2*I0*self.Rs/(self.Vc*(1+self.beta))*np.cos(self.psi)*np.sin(self.psi) )**2)
        # Generator voltage at resonance [V] - Eq. (3.2.2) [1]
        self.Vgr = 2*self.beta**(1/2)/(1+self.beta)*(2*self.Rs*self.Pg)**(1/2)
        # Generator phase at resonance [rad] - from Eq. (4.1.1)
        self.theta_gr = np.arctan((self.Vc*np.sin(self.theta) + self.Vbr(I0)*np.cos(self.psi)*np.sin(self.psi))/
                    (self.Vc*np.cos(self.theta) + self.Vbr(I0)*np.cos(self.psi)**2)) - self.psi
        # Generator voltage [V]
        self.Vg = self.Vgr*np.cos(self.psi)
        # Generator phase [rad]
        self.theta_g = self.theta_gr + self.psi

        self.beam_phasor = self.Vc*np.exp(1j*self.theta) - self.generator_phasor
        self.beam_phasor_record = np.ones(self.ring.h)*self.beam_phasor
        self.generator_phasor_record = np.ones(self.ring.h)*self.generator_phasor
        self.ig_phasor_record = np.ones(self.ring.h)*self._Vg2Ig(self.generator_phasor)

    def plot_phasor(self, I0):
        """
        Plot phasor diagram showing the vector addition of generator and beam 
        loading voltage.

        Parameters
        ----------
        I0 : float
            Beam current in [A].
            
        Returns
        -------
        Figure.

        """

        def make_legend_arrow(legend, orig_handle,
                              xdescent, ydescent,
                              width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
            return p

        fig = plt.figure()
        ax= fig.add_subplot(111, polar=True)
        ax.set_rmax(max([1.2,self.Vb(I0)/self.Vc*1.2,self.Vg/self.Vc*1.2]))
        arr1 = ax.arrow(self.theta, 0, 0, 1, alpha = 0.5, width = 0.015,
                         edgecolor = 'black', lw = 2)

        arr2 = ax.arrow(self.psi + np.pi, 0, 0,self.Vb(I0)/self.Vc, alpha = 0.5, width = 0.015,
                         edgecolor = 'red', lw = 2)

        arr3 = ax.arrow(self.theta_g, 0, 0,self.Vg/self.Vc, alpha = 0.5, width = 0.015,
                         edgecolor = 'blue', lw = 2)

        ax.set_rticks([])  # less radial ticks
        plt.legend([arr1,arr2,arr3], ['Vc','Vb','Vg'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})
        
        return fig
        
    def is_DC_Robinson_stable(self, I0):
        """
        Check DC Robinson stability - Eq. (6.1.1) [1]

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        bool

        """
        return 2*self.Vc*np.sin(self.theta) + self.Vbr(I0)*np.sin(2*self.psi) > 0
    
    def plot_DC_Robinson_stability(self, detune_range = [-1e5,1e5]):
        """
        Plot DC Robinson stability limit.

        Parameters
        ----------
        detune_range : list or array, optional
            Range of tuning to plot in [Hz].

        Returns
        -------
        Figure.

        """
        old_detune = self.psi
        
        x = np.linspace(detune_range[0],detune_range[1],1000)
        y = []
        for i in range(0,x.size):
            self.detune = x[i]
            y.append(-self.Vc*(1+self.beta)/(self.Rs*np.sin(2*self.psi))*np.sin(self.theta)) # droite de stabilité
            
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(x,y)
        ax.set_xlabel("Detune [Hz]")
        ax.set_ylabel("Threshold current [A]")
        ax.set_title("DC Robinson stability limit")
        
        self.psi = old_detune
        
        return fig
        
    def VRF(self, z, I0, F = 1, PHI = 0):
        """Total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return self.Vg*np.cos(self.ring.k1*self.m*z + self.theta_g) - self.Vb(I0)*F*np.cos(self.ring.k1*self.m*z + self.psi - PHI)
    
    def dVRF(self, z, I0, F = 1, PHI = 0):
        """Return derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*self.ring.k1*self.m*np.sin(self.ring.k1*self.m*z + self.theta_g) + self.Vb(I0)*F*self.ring.k1*self.m*np.sin(self.ring.k1*self.m*z + self.psi - PHI)
    
    def ddVRF(self, z, I0, F = 1, PHI = 0):
        """Return the second derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.theta_g) + self.Vb(I0)*F*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.psi - PHI)
        
    def deltaVRF(self, z, I0, F = 1, PHI = 0):
        """Return the generator voltage minus beam loading voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.theta_g) - self.Vb(I0)*F*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.psi - PHI)
    
    def init_FB(self, gain_A, gain_P, delay, FB_index=0):
        """
        Initialize and switch on amplitude and phase feedback.
        
        Objective amplitude and phase are self.Vc and self.theta.

        Parameters
        ----------
        gain_A : float
            Amplitude (voltage) gain of the feedback.
        gain_P : float
            Phase gain of the feedback.
        delay : int
            Feedback delay in unit of turns.
        FB_index : int, optional
            Bunch index at which the feedback is applied.
            Default is 0.

        Returns
        -------
        None.

        """
        self.gain_A = gain_A
        self.gain_P = gain_P
        self.delay = int(delay)
        self.FB = True
        self.volt_delay = np.ones(self.delay)*self.Vc
        self.phase_delay = np.ones(self.delay)*self.theta
        self.FB_index = int(FB_index)
        
    def track_FB(self):
        """
        Tracking method for the amplitude and phase feedback.

        Returns
        -------
        None.

        """
        diff_A = self.volt_delay[-1] - self.Vc
        diff_P = self.phase_delay[-1] - self.theta
        self.Vg -= self.gain_A*diff_A
        self.theta_g -= self.gain_P*diff_P
        self.generator_phasor_record = np.ones(self.ring.h)*self.generator_phasor
        self.volt_delay = np.roll(self.volt_delay, 1)
        self.phase_delay = np.roll(self.phase_delay, 1)
        self.volt_delay[0] = self.cavity_voltage
        self.phase_delay[0] = self.cavity_phase

    @property
    def Vref(self):
        """
        Target (Reference) Vc value in complex

        Vref can be specified as an IQ-complex style or list [Amp, Phase] by using '=' operator.
        If not specified, the complex value calculated from .Vc and .theta is used.

        For main cavity, 
        if only the amplitude is given, the synchronus phase is used as a target phase value.
        """
        return self._Vref

    @Vref.setter
    def Vref(self, value):
        if isinstance(value,complex):
            self._Vref = value
        elif isinstance(value,list):
            self._Vref = value[0]*np.exp(1j*value[1])
        else: 
        # At following line, it is assumed that the CavityResonator is a MC, would not work for an active HC then.
            self._Vref = value*np.exp(1j*np.arccos(self.ring.U0/value))
        self._PIctrlVcLoop_Rot = np.exp(-1j*np.angle(self._Vref))
        self._PIctrlVcLoop_VrefRot = self._Vref*self._PIctrlVcLoop_Rot
        if self._PIctrlVcLoop_FF:
            self.PIctrlVcLoop_FFconst = np.mean(self.PIctrlVcLoop_ig_phasor)

    @property
    def DirectFB_phaseShift(self):
        return self.psi - self._DirectFB_phaseShift
    
    @DirectFB_phaseShift.setter
    def DirectFB_phaseShift(self,value):
        self._DirectFB_phaseShift = self.psi - value 

    @property
    def DirectFB_psi(self):
        """
        Return detune value with Direct RF feedback

        Fig.4 of K. Akai, PRAB 25, 102002 (2022)
        """
        return np.angle(np.mean(self.cavity_phasor_record)) - np.angle(np.mean(self.ig_phasor_record))

    @property
    def DirectFB_detune(self):
        """
        Return detune value with Direct RF feedback

        Fig.4 of K. Akai, PRAB 25, 102002 (2022)
        """
        return self.DirectFB_psi/np.pi*180
    
    @property
    def DirectFB_alpha(self):
        fac = np.abs(np.mean(self.DirectFB_ig_phasor)/np.mean(self.ig_phasor_record))
        return 20*np.log10(fac)
    
    @property
    def DirectFB_gamma(self):
        fac = np.abs(np.mean(self.DirectFB_ig_phasor)/np.mean(self.ig_phasor_record))
        return fac/(1-fac)
    
    @property
    def DirectFB_Rs(self):
        return self.Rs/(1+self.DirectFB_gamma*np.cos(self.DirectFB_psi))

    def _Vg2Ig(self,Vg):
        """
        Return Ig from Vg
        assuming constant Vg

        Eq.25 of N.Yamamoto et al., PRAB 21, 012001 (2018)
        assuming the dVg/dt = 0
        """
        fac = 1/self.filling_time/self.loss_factor # 1/R_L
        ig = fac*Vg*( 1 - 1j*np.tan(self.psi) )
        #dVg_over_dt = dVg/self.ring.T1
        #fac = 1/self.loss_factor
        #ig = fac*(dVg_over_dt+Vg/self.filling_time*( 1 - 1j*np.tan(self.psi) ))
        return ig

    def _Ig2Vg_old(self):
        """
        Return Vg from Ig

        Eq.26 of N.Yamamoto et al., PRAB 21, 012001 (2018)
        """
        for index in range(self.ring.h):
            self.generator_phasor_record[index] = (self.generator_phasor_record[index-1]
                    *np.exp(-1/self.filling_time*(1 - 1j*np.tan(self.psi))*self.ring.T1)
                    + self.ig_phasor_record[index]*self.loss_factor*self.ring.T1)
        self.Vg=np.mean(np.abs(self.generator_phasor_record))
        self.theta_g=np.mean(np.angle(self.generator_phasor_record))

    def _Ig2Vg(self):
        """
            expect faster processing than _Ig2Vg_old
        """
        self.generator_phasor_record = (self._Ig2Vg_vec*self.generator_phasor_record[-1] +
                    self._Ig2Vg_mat.dot(self.ig_phasor_record)*self.loss_factor*self.ring.T1)
        self.Vg=np.mean(np.abs(self.generator_phasor_record))
        self.theta_g=np.mean(np.angle(self.generator_phasor_record))

    def _init_Ig2Vg(self):
        """
            shoud be called before first time of Ig2Vg conversion
            and after cavity parameter change.
        """
        k = np.arange(0, self.ring.h)
        ###
        self._Ig2Vg_vec = np.exp(-1/self.filling_time*(1 - 1j*np.tan(self.psi))*self.ring.T1*(k+1))
        ####
        tempV = np.exp(-1/self.filling_time*self.ring.T1*k*(1 - 1j*np.tan(self.psi)))
        for idx in np.arange(self.ring.h):
            self._Ig2Vg_mat[idx:,idx]=tempV[:self.ring.h-idx]

    def init_PIctrlVcLoop(self,gain,sample_num,every,delay,Vref=None,IQ=True,FF=True):
        """
        Vc loop via Generator current (ig) by using PI controller,
        PI (Proportional,Integral) controller : 
            Ref. https://en.wikipedia.org/wiki/PID_controller
        assumption :  delay >> every ~ sample_num

        NOTE:Should be called after generator parameters are set.

        Adjusting ig(Vg) parameter to keep the Vc constant (to target(Vref) values).
        The method "track_" is
           1) Monitoring the Vc phasor
           Mean Vc value between specified bunch number (sample_num) is monitored
           with specified interval period (every).
           2) Changing the ig phasor
           The ig phasor is changed according the difference of the specified
           reference values (Vref) with specified gain (gain).
           By using ig instead of Vg, the cavity response can be taken account.
           3) ig changes are reflected to Vg after the specifed delay (delay) of the system

        Vc,IQ-->(rot)-->(-)-->(V->I,fac)-->PI-->Ig,IQ {--> Vg,IQ}
                         |
                         Ref

        Example
        -------
            PF; E0=2.5GeV, C=187m, frf=500MHz
                QL=11800, fs0=23kHz
                ==> gain=[0.5,1e4],sample_num=8,every=7(13ns),delay=500(1us)
                             is one reasonable parameter set. 

        Parameters
        -------
        gain : float or float list
            Pgain and Igain of the feedback system
            For IQ feedback, same gain set is used for I and Q.
            In case of normal conducting cavity (QL~1e4),
                the Pgain of ~1.0 and Igain of ~1e4(5) are usually used.
            In case of super conducting cavity (QL > 1e6),
                the Pgain of ~100 can be used.
            In a "bad" parameter set, unstable oscillation of Vc is caused.
            So, parameter scan of Vcloop should be made. 
        sample_num : int
            Sample number to monitor Vc 
            The averaged Vc value in sample_num is monitored.
            Units are in bucket numbers.
        every : int
            interval to monitor and change Vc
            Units are in bucket numbers.
        delay : int
            Loop delay of the assumed system. 
            Units are in bucket numbers.
        Vref : float, complex
            target (reference) value of the feedback
            if None, .Vc and .theta is set as the reference
        IQ : bool
            reserved parameter for IQ/AP switching. (to be...)
        FF : bool
            True is recommended to prevent a Vc drop in the beginning of the tracking.
            In case of small Pgain (QL ~ 1e4), Vc drop may cause baem loss due to static Robinson.
        """
        self._IQ = False
        if IQ:
            self._IQ = True

        if isinstance(gain,list): # if IQ is true, len(gain) should be 2
            self.PIctrlVcLoop_g = gain
        else:
            self.PIctrlVcLoop_g = [gain,0]

        if delay > 0:
            self.PIctrlVcLoop_delay = int(delay)
        else:
            self.PIctrlVcLoop_delay = 1
        if every > 0:
            self.PIctrlVcLoop_every = int(every)
        else:
            self.PIctrlVcLoop_every = 1
        record_size = int(np.ceil(self.PIctrlVcLoop_delay/self.PIctrlVcLoop_every))
        if record_size < 1:
            raise ValueError("Bad parameter set : delay or sample_every")
        self.PIctrlVcLoop_sample = int(sample_num)

        # init lists for FB process
        #self.generator_phasor_record = np.ones(self.ring.h)*self.generator_phasor
        if self._DirectFB is False:
            self.PIctrlVcLoop_ig_phasor = np.ones(self.ring.h,dtype=np.complex)*self._Vg2Ig(self.generator_phasor)
            self.ig_phasor_record = self.PIctrlVcLoop_ig_phasor
        self.PIctrlVcLoop_vc_previous = np.ones(self.PIctrlVcLoop_sample)*self.cavity_phasor # 
        self.PIctrlVcLoop_diff_record = np.zeros(record_size,dtype=np.complex)
        self.PIctrlVcLoop_I_record = 0+0j

        
        if FF:
            self._PIctrlVcLoop_FF = True
            #self.PIctrlVcLoop_FFconst = self.PIctrlVcLoop_ig_phasor[0]
        else:
            self._PIctrlVcLoop_FF = False
            self.PIctrlVcLoop_FFconst = 0

        if Vref is not None:
            self.Vref = Vref
        else: 
            self.Vref = self.Vc*np.exp(1j*self.theta)

        self.PIctrlVcLoop_sample_list = range(0,self.ring.h,self.PIctrlVcLoop_every)
        self._PIctrlVcLoop = True
        self._Ig2Vg_switch = True

        # Pre caclulation for Ig2Vg
        self._init_Ig2Vg()

    def track_PIctrlVcLoop(self):
        """
        Tracking method for the Cavity PI control feedback.

        Returns
        -------
        None.

        """
        vc_list = np.concatenate([self.PIctrlVcLoop_vc_previous,self.cavity_phasor_record]) #This line is slowing down the process.
        #self.PIctrlVcLoop_ig_phasor=np.ones(self.ring.h,dtype=np.complex)*self.PIctrlVcLoop_ig_phasor[-1]
        self.PIctrlVcLoop_ig_phasor.fill(self.PIctrlVcLoop_ig_phasor[-1])
        
        for index in self.PIctrlVcLoop_sample_list: 
            # 2) updating Ig using last item of the list
            diff = self.PIctrlVcLoop_diff_record[-1]- self.PIctrlVcLoop_FFconst
            self.PIctrlVcLoop_I_record += diff/self.ring.f1
            fb_value = self.PIctrlVcLoop_g[0]*diff + self.PIctrlVcLoop_g[1]*self.PIctrlVcLoop_I_record
            self.PIctrlVcLoop_ig_phasor[index:] = self._Vg2Ig(fb_value) + self.PIctrlVcLoop_FFconst
            # Shift the record
            self.PIctrlVcLoop_diff_record = np.roll(self.PIctrlVcLoop_diff_record,1)
            # 1) recording diff as a first item of the list
            mean_vc = np.mean(vc_list[index:self.PIctrlVcLoop_sample+index])*self._PIctrlVcLoop_Rot
            self.PIctrlVcLoop_diff_record[0] = self._PIctrlVcLoop_VrefRot - mean_vc
        # update sample_list for next turn
        self.PIctrlVcLoop_sample_list = range(index+self.PIctrlVcLoop_every-self.ring.h,self.ring.h,self.PIctrlVcLoop_every)
        # update vc_previous for next turn
        self.PIctrlVcLoop_vc_previous = self.cavity_phasor_record[- self.PIctrlVcLoop_sample:]
        
        self.ig_phasor_record = self.PIctrlVcLoop_ig_phasor

    def init_DirectFB(self,gain,phase_shift,sample_num=0,every=0,delay=0):
        """
        Direct RF FB via Generator current (ig) by using PI controller, 
            Ref. https://link.aps.org/doi/10.1103/PhysRevAccelBeams.25.102002
        To keep the cavity voltage constant, cavity feedback (PIctrlVcLoop) should be called together.
        To avoid cavity-beam unmatching (large synchrotron oscilation of beam),
            Self.set_generator(I0) should be called previously.
        
        Parameters
        -------
        I0   : float
            Beam current
        gain : float
            Gain of the Direct RF feedback system
        sample_num : int
            Sample number to monitor Vc 
            The averaged Vc value in sample_num is monitored.
            Units are in bucket numbers.
        every : int
            interval to monitor and change Vc
            Units are in bucket numbers.
        loop_delay : int
            Loop delay of the assumed system. 
            Units are in bucket numbers.
        """
        if delay > 0:
            self.DirectFB_loopDelay = int(delay)
        elif delay == 0:
            self.DirectFB_loopDelay = self.PIctrlVcLoop_delay
        else:
            raise ValueError("Bad parameter set : loop_delay")
        if sample_num > 0:
            self.DirectFB_sample = int(sample_num)
        elif sample_num == 0:
            self.DirectFB_sample = self.PIctrlVcLoop_sample
        else:
            raise ValueError("Bad parameter set : sample_num")
        if every > 0:
            self.DirectFB_every = int(every)
        elif delay == 0:
            self.DirectFB_every = self.PIctrlVcLoop_every
        else:
            raise ValueError("Bad parameter set : every")
        record_size = int(np.ceil(self.DirectFB_loopDelay/self.DirectFB_every))
        if record_size < 1:
            raise ValueError("Bad parameter set : delay or sample_every")

        self.DirectFB_phaseShift = 0
        self.DirectFB_parameter_set([gain,phase_shift])
        if np.sum(np.abs(self.beam_phasor)) == 0:
            cavity_phasor = self.Vc*np.exp(1j*self.theta)
        else:
            cavity_phasor = np.mean(self.cavity_phasor_record)
        self.DirectFB_VcRecord = np.ones(record_size,dtype=complex)*cavity_phasor
        self.DirectFB_vc_previous = np.ones(self.DirectFB_sample,dtype=complex)*cavity_phasor 

        self.DirectFB_sample_list = range(0,self.ring.h,self.DirectFB_every)
        self._DirectFB = True

    def DirectFB_parameter_set(self,para):
        if isinstance(para,list): # if IQ is true, len(gain) should be 2
            self.DirectFB_gain = para[0]
            self.DirectFB_phaseShift = para[1]
        else:
            self.DirectFB_gain = para
        #self._DirectFB_phaseShift = self.psi - phase # using for calculation, definition in PRAB 25,102002 (2022)
        if np.sum(np.abs(self.beam_phasor)) == 0:
            vc = np.ones(self.ring.h)*self.Vc*np.exp(1j*self.theta)
        else:
            vc = self.cavity_phasor_record
        vg_drf = self.DirectFB_gain*vc*np.exp(1j*self._DirectFB_phaseShift)
        self.DirectFB_ig_phasor = self._Vg2Ig(vg_drf)

        self.PIctrlVcLoop_ig_phasor = self.ig_phasor_record  - self.DirectFB_ig_phasor
        if self._PIctrlVcLoop_FF:
            self.PIctrlVcLoop_FFconst = np.mean(self.PIctrlVcLoop_ig_phasor)


    def track_DirectFB(self):
        """
        Tracking method for the Direct RF feedback.

        Returns
        -------
        None.

        """
        vc_list = np.concatenate([self.DirectFB_vc_previous,self.cavity_phasor_record])
        #self.DirectFB_ig_phasor=np.ones(self.ring.h)*self.DirectFB_ig_phasor[-1]
        self.DirectFB_ig_phasor=np.roll(self.DirectFB_ig_phasor,1)
        for index in self.DirectFB_sample_list:
            # 2) updating Ig using last item of the list
            vg_drf = self.DirectFB_gain*self.DirectFB_VcRecord[-1]*np.exp(1j*self._DirectFB_phaseShift)
            self.DirectFB_ig_phasor[index:] = self._Vg2Ig(vg_drf) 
            # Shift the record
            self.DirectFB_VcRecord = np.roll(self.DirectFB_VcRecord,1)
            # 1) recording Vc
            mean_vc = np.mean(vc_list[index:self.DirectFB_sample+index])
            self.DirectFB_VcRecord[0] = mean_vc
        # update sample_list for next turn
        self.DirectFB_sample_list = range(index+self.DirectFB_every-self.ring.h,self.ring.h,self.DirectFB_every)
        # update vc_previous for next turn
        self.DirectFB_vc_previous = self.cavity_phasor_record[- self.DirectFB_sample:]

        #self.ig_phasor_record += self.DirectFB_ig_phasor # don't work if type is different.
        self.ig_phasor_record = self.PIctrlVcLoop_ig_phasor + self.DirectFB_ig_phasor

    def DirectFB_value(self):
        fac = np.abs(np.mean(self.DirectFB_ig_phasor)/np.mean(self.ig_phasor_record))
        alpha = 20*np.log10(fac)
        gamma = fac/(1-fac)
        return alpha,gamma
    
    def DirectFB_Vg(self,vc=-1):
        if vc == -1:
            vc = np.mean(self.cavity_phasor_record)
        vg_drf=self.DirectFB_gain*vc*np.exp(1j*self._DirectFB_phaseShift)
        vg_main=np.mean(self.generator_phasor_record)-vg_drf
        return  vg_main,vg_drf

    def DirectFB_fs(self,vg_main=-1,vg_drf=-1):
        vc = np.mean(self.cavity_phasor_record)
        if vg_drf ==-1:
            vg_drf=self.DirectFB_gain*vc*np.exp(1j*self._DirectFB_phaseShift)
        if vg_main == -1:
            vg_main=np.mean(self.generator_phasor_record)-vg_drf
        vg_sum = np.abs(vg_main)*np.sin(np.angle(vg_main))+np.abs(vg_drf)*np.sin(np.angle(vg_drf))
        omega_s = 0
        if (vg_sum) > 0.0:
            omega_s=np.sqrt(self.ring.ac*self.ring.omega1*(vg_sum)/self.ring.E0/self.ring.T0)
        return omega_s/2/np.pi

    def init_TunerLoop(self,gain=0.01,sample_turn=0,offset=0):
        """
        Cavity tuner loop keep the phase between cavity and generator current constant
        by changing the cavity tuning (psi or detune)
        Only a proportional contorller is assumed.

        Parameters :
        ------
        gain : float
            Proportional gain of Tuner Loop
            If not specified, 0.01 is used.
        sample_turn:
            Sample number to monitor phase differences. 
            Averages during sample_turn are used for feedback.
            Units are in turn numbers.
            The value longer than one synchrotron period (1/fs) is recommended.
            If not specified, 2-synchrotron period (2/fs) is used, although it is
            too fast compared to the actual situation.
        offset : float
            Tuning offset in rad
        """
        self._TunerLoop = True
        if sample_turn == 0:
            #fs = np.sqrt(self.ring.ac*self.ring.h*self.ring.U0/2/np.pi/self.ring.E0
            #    *np.sqrt(self.c/self.ring.U0*self.Vc/self.ring.U0-1))*self.ring.f1/self.ring.h
            fs = self.ring.synchrotron_tune(self.Vc)*self.ring.f1/self.ring.h
            sample_turn = 2/fs/self.ring.T0

        self.TunerLoop_Pgain = gain
        self.TunerLoop_offset = offset
        self.TunerLoop_record = int(sample_turn)
        self.TunerLoop_diff = 0
        self.TunerLoop_count = 0

    def track_TunerLoop(self):
        """
        Tracking method for the TunerLoop of CavityResonator.

        Returns
        -------
        None.

        """
        if self.TunerLoop_count == self.TunerLoop_record:
            diff = self.TunerLoop_diff/self.TunerLoop_record-self.TunerLoop_offset
            self.psi -= diff*self.TunerLoop_Pgain
            self.TunerLoop_count = 0
            self.TunerLoop_diff = 0
        else:
            self.TunerLoop_diff += self.cavity_phase - self.theta_g + self.psi
            self.TunerLoop_count += 1

    def is_AC_Robinson_stable(self,I0,tune,mode=[0],bool_return=False):
        """
        Check AC Robinson stability 
        Effect of Direct RF feedback is not included.

        This method caluclates the CBI growth rate from own impedance.

        Parameters
        ----------
        I0 : float
            Beam current in [A].
        tune : float
            fractional number of longitudinal tune
        mode : float or list of float
            Coupled Bunch Instability mode number 
        bool_return : bool
            if True, return bool
            
        Returns
        -------
        bool

        """
        return self.is_CBI_stable(I0,tune,mode,bool_return)

    def is_CBI_stable(self,I0,tune,mode=[0],max_freq=5,bool_return=False):
        """
        Check Coupled-Bunch-Instability stability 
        Effect of Direct RF feedback is not included.

        This method caluclates the CBI growth rate from own impedance.

        Parameters
        ----------
        I0 : float
            Beam current in [A].
        tune : float
            fractional number of longitudinal tune
        mode : float or list of float
            Coupled Bunch Instability mode number 
        bool_return : bool
            if True, return bool
            
        Returns
        -------
        bool

        """
        if tune > 1:
            tune = tune - int(tune)
        cof = self.ring.ac*I0/2.0/tune/self.ring.E0
        if isinstance(mode,list):
            CBImode = mode
        else:
            CBImode = [mode]

        gr = np.zeros(len(CBImode))
        gr_bool = np.zeros(len(CBImode),dtype=bool)
        count = 0
        for i in CBImode:
            """
            fp=self.ring.f1+self.ring.f0*(i+tune)
            fm=self.ring.f1-self.ring.f0*(i+tune)
            gr[count]=cof*(fp*np.real(self.Z(fp))-fm*np.real(self.Z(fm)))
            gr_bool[count] = ( gr[count] > 1/self.ring.tau[2] )
            """
            #fp = np.array([0,1,2,3,4])*self.ring.f1+i*self.ring.f0*(i+tune)
            fp = self.ring.f0*(np.arange(max_freq)*self.ring.h+(i+tune))
            #fm = np.array([1,2,3,4,5])*self.ring.f1-i*self.ring.f0*(i+tune)
            fm = self.ring.f0*(np.arange(1,max_freq+1)*self.ring.h-(i+tune))
            sumZ = np.sum(fp*np.real(self.Z(fp))) - np.sum(fm*np.real(self.Z(fm)))
            
            gr[count] = cof*sumZ
            gr_bool[count] = ( gr[count] > 1/self.ring.tau[2] )
            count +=1

        if bool_return:
            return gr_bool
        else:
            return gr 
        
    def plot_directFB_phasor(self):
        """
        Plot phasor diagram showing the vector addition of generator and beam 
        loading voltage.
            
        Returns
        -------
        Figure.

        """     
        if np.sum(np.abs(self.beam_phasor)) == 0:
            vc = self.Vc*np.exp(1j*self.theta)
        else:
            vc = np.mean(self.cavity_phasor_record)
        if self._DirectFB is False:
            self.DirectFB_gain = 0
            self._DirectFB_phaseShift = 0
        
        #vg_drf=self.DirectFB_gain*vc*np.exp(1j*self.DirectFB_phaseShift)
        #vg_main=np.mean(self.generator_phasor_record)-vg_drf
        vg_main,vg_drf = self.DirectFB_Vg(vc)
        
        vb=np.mean(self.beam_phasor_record)
        vc_amp=np.abs(vc)

        fig, ax = plt.subplots()
        arr1=ax.annotate('', xy=[vc.real/vc_amp, vc.imag/vc_amp], xytext=[0,0],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='red', edgecolor='red')
               )
        arr2=ax.annotate('', xy=[vg_main.real/vc_amp, vg_main.imag/vc_amp], xytext=[0,0],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='blue', edgecolor='blue')
               )
        arr3=ax.annotate('', xy=[(vg_main+vg_drf).real/vc_amp, (vg_main+vg_drf).imag/vc_amp], 
                         xytext=[vg_main.real/vc_amp, vg_main.imag/vc_amp],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='orange', edgecolor='orange')
               )
        arr4=ax.annotate('', xy=[(vg_main+vg_drf+vb).real/vc_amp, (vg_main+vg_drf+vb).imag/vc_amp], 
                         xytext=[(vg_main+vg_drf).real/vc_amp, (vg_main+vg_drf).imag/vc_amp],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='green', edgecolor='green')
               )
        
        ax.set_title("Gain = %.2f, PhaseShift = %.1f [deg]"%(self.DirectFB_gain,self.DirectFB_phaseShift/np.pi*180))
        p1 = ax.scatter(0, 0,c='red',s=0.5)
        p2 = ax.scatter(vg_main.real/vc_amp, vg_main.imag/vc_amp,c='blue',s=0.5)
        p3 =ax.scatter((vg_drf+vg_main).real/vc_amp, (vg_drf+vg_main).imag/vc_amp,c='orange',s=0.5)
        p4 =ax.scatter(vc.real/vc_amp, vc.imag/vc_amp,c='green',s=0.5)
        ax.legend(handles=[p1,p2,p3,p4],labels=['Vc','Vg_main','Vg_drf','Vb'],loc="best")

        #fig.legend(['Vc','Vg_main','Vg_drf','Vb'])
        
        return fig
    
    def plot_directFB_igphasor(self):
        """
        Plot phasor diagram showing the vector addition of generator and beam 
        loading voltage.
            
        Returns
        -------
        Figure.

        """     
        if np.sum(np.abs(self.beam_phasor)) == 0:
            ig_total = self.Vc*np.exp(1j*self.theta)
        else:
            ig_total = np.mean(self.cavity_phasor_record)
        if self._DirectFB is False:
            self.DirectFB_gain = 0
            self._DirectFB_phaseShift = 0
        
        #vg_drf=self.DirectFB_gain*vc*np.exp(1j*self.DirectFB_phaseShift)
        #vg_main=np.mean(self.generator_phasor_record)-vg_drf
        ig_main,ig_drf = self.DirectFB_Vg(ig_total)
        
        ig_beam=np.mean(self.beam_phasor_record)
        ig_amp=np.abs(ig_total)

        ig_main = np.mean(self.PIctrlVcLoop_ig_phasor)
        ig_drf = np.mean(self.DirectFB_ig_phasor)
        ig_drive = np.mean(self.PIctrlVcLoop_ig_phasor + self.DirectFB_ig_phasor)
        ig_total = self._Vg2Ig(np.mean(self.cavity_phasor_record))
        ig_beam = ig_total-ig_drive
        ig_amp= np.abs(ig_total)
        vc=np.mean(self.cavity_phasor_record)
        vc_amp = np.abs(vc)

        fig, ax = plt.subplots()
        arr1=ax.annotate('', xy=[ig_total.real/ig_amp, ig_total.imag/ig_amp], xytext=[0,0],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='red', edgecolor='red')
               )
        arr2=ax.annotate('', xy=[ig_main.real/ig_amp, ig_main.imag/ig_amp], xytext=[0,0],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='blue', edgecolor='blue')
               )
        arr3=ax.annotate('', xy=[(ig_main+ig_drf).real/ig_amp, (ig_main+ig_drf).imag/ig_amp], 
                         xytext=[ig_main.real/ig_amp, ig_main.imag/ig_amp],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='orange', edgecolor='orange')
               )
        arr4=ax.annotate('', xy=[(ig_main+ig_drf+ig_beam).real/ig_amp, (ig_main+ig_drf+ig_beam).imag/ig_amp], 
                         xytext=[(ig_main+ig_drf).real/ig_amp, (ig_main+ig_drf).imag/ig_amp],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='green', edgecolor='green')
               )
        arr0=ax.annotate('', xy=[vc.real/vc_amp, vc.imag/vc_amp], xytext=[0,0],
                arrowprops=dict(shrink=0, width=1, headwidth=8, 
                                headlength=10, connectionstyle='arc3',
                                facecolor='black', edgecolor='black')
               )
        
        ax.set_title("Gain = %.2f, PhaseShift = %.1f [deg]"%(self.DirectFB_gain,self.DirectFB_phaseShift/np.pi*180))
        p0 = ax.scatter(vc.real/vc_amp, vc.imag/vc_amp,c='black',s=0.5)
        p1 = ax.scatter(0, 0,c='red',s=0.5)
        p2 = ax.scatter(ig_main.real/ig_amp, ig_main.imag/ig_amp,c='blue',s=0.5)
        p3 =ax.scatter((ig_drf+ig_main).real/ig_amp, (ig_drf+ig_main).imag/ig_amp,c='orange',s=0.5)
        p4 =ax.scatter(ig_total.real/ig_amp, ig_total.imag/ig_amp,c='green',s=0.5)
        ax.legend(handles=[p0,p1,p2,p3,p4],labels=['Vc','Ig_total','Ig_main','Ig_drf','Ig_beam'],loc="best")

        #fig.legend(['Vc','Vg_main','Vg_drf','Vb'])
        
        return fig

#Routines to run a real-time projected-DMET calculation

import numpy as np
import system_mod
import utils
import applyham_pyscf
import sys
import os

import fci_mod
import pyscf.fci

############ CLASS TO RUN REAL-TIME DMET CALCULATION #########

class dynamics_driver():

    #####################################################################

    def __init__( self, h_site, V_site, hamtype, tot_system, delt, Nstep, Nprint=100, init_time=0.0, integ='rk1' ):

        #h_site     - 1 e- hamiltonian in site-basis for total system to run dynamics
        #V_site     - 2 e- hamiltonian in site-basis for total system to run dynamics
        #hamtype    - integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
        #tot_system - a previously defined DMET total system including all fragment information
        #delt       - time step
        #Nstep      - total number of time-steps
        #Nprint     - number of time-steps between printing
        #init_time  - the starting time for the calculation
        #integ      - the type of integrator used

        self.tot_system = tot_system
        self.delt       = delt
        self.Nstep      = Nstep
        self.Nprint     = Nprint
        self.init_time  = init_time
        self.integ      = integ

        print
        print '********************************************'
        print '     SET-UP REAL-TIME DMET CALCULATION       '
        print '********************************************'
        print

        #Input error checks

        #Convert rotation matrices and CI coefficients to complex arrays if they're not already
        for frag in self.tot_system.frag_list:
            if( not np.iscomplexobj( frag.rotmat ) ):
                frag.rotmat = frag.rotmat.astype(complex)
            if( not np.iscomplexobj( frag.CIcoeffs ) ):
                frag.CIcoeffs = frag.CIcoeffs.astype(complex)

        #Set-up Hamiltonian for dynamics calculation
        self.tot_system.h_site = h_site
        self.tot_system.V_site = V_site

        #Define output files
        self.file_output   = file( 'output.dat', 'wa' )
        self.file_corrdens = file( 'corr_density.dat', 'wa' )

    #####################################################################

    def kernel( self ):

        print
        print '********************************************'
        print '     BEGIN REAL-TIME DMET CALCULATION       '
        print '********************************************'
        print

        #DYNAMICS LOOP
        current_time = self.init_time
        for step in range(self.Nstep):

            #Print data before taking time-step, this always prints out data at initial time step
            if( np.mod( step, self.Nprint ) == 0 ):
                print 'Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation'
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate FCI coefficients and rotation matrix for all fragments
            self.integrate()

            #Increase current_time
            current_time += self.delt


        #Print data at final step regardless of Nprint
        print 'Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation'
        self.print_data( current_time )
        sys.stdout.flush()


        print
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "END REAL-TIME DMET CALCULATION"
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print

    #####################################################################

    def integrate( self ):
        #Subroutine to integrate equations of motion for CI coefficients
        #and rotation matrix for all fragments

        if( self.integ == 'rk1' ):
            #Use 1st order runge-kutta (ie euler's method) to integrate EOMs

            #Calculate terms needed for X-matrix
            self.tot_system.get_frag_corr1RDM()
            self.tot_system.get_glob1RDM()
            self.tot_system.get_nat_orbs()

            #Calculate embedding hamiltonian
            self.tot_system.get_frag_Hemb()

            #mrar
            for frag in self.tot_system.frag_list:
                #frag.Ecore = -fci_mod.get_FCI_E( frag.h_emb, frag.V_emb, 0.0, frag.CIcoeffs, 2*frag.Nimp, frag.Nimp, frag.Nimp )
                frag.Ecore = 0.0

            #Calculate X-matrix
            self.tot_system.get_Xmats( self.tot_system.Nele/2 )

            #Integrate rotation matrices
            for frag in self.tot_system.frag_list:
                frag.rotmat -= 1j * self.delt * np.dot( frag.rotmat, frag.Xmat )

            #Integrate CI coefficients
            for frag in self.tot_system.frag_list:
                frag.CIcoeffs -= 1j * self.delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore )

        #else if( self.integ == 'rk4' ):

        else:
            print 'ERROR: A proper integrator was not specified'
            print
            exit()
            

    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate DMET energy, which also includes calculation of 1 & 2 RDMs for each fragment
        self.tot_system.get_DMET_E()

        #Calculate total number of electrons
        self.tot_system.get_DMET_Nele()


        ######## PRINT OUT EVERYTHING #######

        #Print correlated density in the site basis
        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] = np.copy( np.diag( np.real( frag.corr1RDM[:frag.Nimp] ) ) )
            cnt += frag.Nimp
        corrdens = np.insert( corrdens, 0, current_time )
        np.savetxt( self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )
        self.file_corrdens.flush()

        #Print output data
        output    = np.zeros(3)
        output[0] = current_time
        output[1] = self.tot_system.DMET_E
        output[2] = self.tot_system.DMET_Nele
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Print current value of rotation matrix and CI coefficients for re-starting PING

    #####################################################################


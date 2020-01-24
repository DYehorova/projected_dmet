#Routines to run a real-time projected-DMET calculation

import numpy as np
import system_mod
import utils
import applyham_pyscf
import mf1rdm_timedep_mod
import sys
import os
import multiprocessing as multproc

import time

import fci_mod
import pyscf.fci
import integrators

############ CLASS TO RUN REAL-TIME DMET CALCULATION #########

class dynamics_driver():

    #####################################################################

    def __init__( self, h_site, V_site, hamtype, tot_system, delt, Nstep, Nprint=100, integ='rk1', nproc=1, init_time=0.0 ):

        #h_site     - 1 e- hamiltonian in site-basis for total system to run dynamics
        #V_site     - 2 e- hamiltonian in site-basis for total system to run dynamics
        #hamtype    - integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
        #tot_system - a previously defined DMET total system including all fragment information
        #delt       - time step
        #Nstep      - total number of time-steps
        #Nprint     - number of time-steps between printing
        #init_time  - the starting time for the calculation
        #integ      - the type of integrator used
        #nproc      - number of processors for calculation - careful, there is no check that this matches the pbs script

        self.tot_system = tot_system
        self.delt       = delt
        self.Nstep      = Nstep
        self.Nprint     = Nprint
        self.init_time  = init_time
        self.integ      = integ
        self.nproc      = nproc

        print()
        print('********************************************')
        print('     SET-UP REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()

        #Input error checks

        #Convert rotation matrices, CI coefficients, and MF 1RDM to complex arrays if they're not already
        for frag in self.tot_system.frag_list:
            if( not np.iscomplexobj( frag.rotmat ) ):
                frag.rotmat = frag.rotmat.astype(complex)
            if( not np.iscomplexobj( frag.CIcoeffs ) ):
                frag.CIcoeffs = frag.CIcoeffs.astype(complex)

        #Set-up Hamiltonian for dynamics calculation
        self.tot_system.h_site = h_site
        self.tot_system.V_site = V_site

        #Define output files
        #self.file_output   = file( 'output.dat', 'wa' )
        #self.file_corrdens = file( 'corr_density.dat', 'wa' )

        self.file_output   = open( 'output.dat', 'wb' )
        self.file_corrdens = open( 'corr_density.dat', 'wb' )

    #####################################################################

    def kernel( self ):

        print()
        print('********************************************')
        print('     BEGIN REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()

        #DYNAMICS LOOP
        current_time = self.init_time
        for step in range(self.Nstep):

            #Print data before taking time-step, this always prints out data at initial time step
            if( np.mod( step, self.Nprint ) == 0 ):
                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate FCI coefficients and rotation matrix for all fragments
            self.integrate(self.nproc)

            #Increase current_time
            current_time += self.delt


        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()


        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

    #####################################################################

    def integrate( self, nproc ):
        #Subroutine to integrate equations of motion

        if( self.integ == 'rk1_orb' ):
            #Use 1st order runge-kutta (ie euler's method) to integrate EOMs
            #Using EOM that explicitly integrate embedding orbitals and CI coefficients using X-matrix

            #Calculate appropriate changes in rotation matrices and CI coeffients
            #Note that l and k terms are for the rotation matrices and CI coefficients respectively
            l1_list, k1_list = self.one_rk_step_orb(nproc)

            #Update rotation matrices and CI coefficients by full time step
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   += l1_list[cnt]
                frag.CIcoeffs += k1_list[cnt]

        elif( self.integ == 'rk4_orb' ):
            #Use 4th order runge-kutta to integrate EOMs
            #Using EOM that explicitly integrate embedding orbitals and CI coefficients using X-matrix

            #Copy rotation matrices (ie orbitals) and CI coefficients at time t
            init_rotmat_list   = []
            init_CIcoeffs_list = []
            for frag in self.tot_system.frag_list:
                init_rotmat_list.append( np.copy(frag.rotmat) )
                init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )

            #Calculate appropriate changes in rotation matrices and CI coeffients
            #Note that l and k terms are for the rotation matrices and CI coefficients respectively
            l1_list, k1_list = self.one_rk_step_orb(nproc)

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*l1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*k1_list[cnt]

            l2_list, k2_list = self.one_rk_step_orb(nproc)

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*l2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*k2_list[cnt]

            l3_list, k3_list = self.one_rk_step_orb(nproc)

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0*l3_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*k3_list[cnt]

            l4_list, k4_list = self.one_rk_step_orb(nproc)

            #Update rotation matrices and CI coefficients by full time-step
            for cnt, frag in enumerate( self.tot_system.frag_list ):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0/6.0 * ( l1_list[cnt] + 2.0*l2_list[cnt] + 2.0*l3_list[cnt] + l4_list[cnt] )
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )

        elif( self.integ == 'rk4_mf' ):
            #Use 4th order runge-kutta to integrate EOMs
            #Using EOM that integrates CI coefficients and MF 1RDM 
            #embedding orbitals obtained by diagonalizing MF 1RDM at each step

            #Copy MF 1RDM and CI coefficients at time t
            init_mf1RDM = np.copy( self.tot_system.mf1RDM )
            init_CIcoeffs_list = []
            for frag in self.tot_system.frag_list:
                init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )

            #Calculate appropriate changes in MF 1RDM and CI coefficients
            #Note that l and k terms are for MF 1 RDM and CI coefficients respectively
            l1, k1_list = self.one_rk_step_mf(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 0.5*l1
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*k1_list[cnt]

            l2, k2_list = self.one_rk_step_mf(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 0.5*l2
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*k2_list[cnt]

            l3, k3_list = self.one_rk_step_mf(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 1.0*l3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*k3_list[cnt]

            l4, k4_list = self.one_rk_step_mf(nproc)

            #Update MF 1RDM and CI coefficients by full time-step
            self.tot_system.mf1RDM = init_mf1RDM + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
            for cnt, frag in enumerate( self.tot_system.frag_list ):
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )

            #Update rotation matrix at new time-step by diagonalizing MF 1RDM maintaing same phase from previous step
            #Technically maintaining phase amongst each partial step along RK algorithm
            prev_rotmat = []
            for frag in self.tot_system.frag_list:
                prev_rotmat.append( np.copy(frag.rotmat) )

            self.tot_system.get_frag_rotmat()

            for cnt, frag in enumerate(self.tot_system.frag_list):
                phase = np.diag( np.round( np.diag( np.real( np.dot( utils.adjoint( prev_rotmat[cnt] ), frag.rotmat ) ) ) ) )
                frag.rotmat = np.dot( frag.rotmat, phase )

        elif( self.integ == 'exact' ):
            #Exactly propagate CI coefficients
            #Only works if have 2 fragments, each spanning full space since then embedding hamiltonian is time-independent
            #and embedding orbitals are time-independent
            #Embedding hamiltonian must also be real

            #Calculate embedding hamiltonian (isnt technically needed since it doesnt change)
            self.tot_system.get_frag_Hemb()

            #Form FCI hamiltonian (technically only need to do this at time zero since doesnt change)
            for frag in self.tot_system.frag_list:
                dim1 = frag.CIcoeffs.shape[0]
                dim2 = frag.CIcoeffs.shape[1]
                Ndet = dim1*dim2
                H_fci = np.zeros( [Ndet,Ndet] )
                for i1 in range(dim1):
                    for i2 in range(dim2):
                        vec1 = np.zeros([dim1,dim2])
                        vec1[i1,i2] = 1.0
                        i = i2 + i1*dim2
                        for j1 in range(dim1):
                            for j2 in range(dim2):
                                j = j2 + j1*dim2
                                vec2 = np.zeros([dim1,dim2])
                                vec2[j1,j2] = 1.0

                                H_fci[i,j] = pyscf.fci.addons.overlap( vec1, np.real(applyham_pyscf.apply_ham_pyscf_fully_complex( vec2, frag.h_emb, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore )), 2*frag.Nimp, (frag.Nimp,frag.Nimp) )

                #Convert matrix of CI coeffs to a vector
                CIvec = frag.CIcoeffs.flatten('F')

                #Exactly integrate the CI coefficients using the FCI hamiltonian
                frag.CIcoeffs = integrators.exact_timeindep_coeff_matrix( CIvec, H_fci, self.delt ).reshape((dim1,dim2), order='F')

        else:
            print('ERROR: A proper integrator was not specified')
            print()
            exit()
            

    #####################################################################

    def one_rk_step_orb( self, nproc ):
        #Subroutine to calculate one change in a runge-kutta step of any order
        #Using EOM that explicitly integrate embedding orbitals and CI coefficients using X-matrix
        #Prior to calling this routine need to update rotation matrices and CI coefficients
        #of each fragment up to appropriate point

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
        self.tot_system.get_Xmats( int(self.tot_system.Nele/2), nproc )

        #Calculate change in rotation matrices
        new_rotmat_list = []
        for frag in self.tot_system.frag_list:
            new_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )

        #Calculate change in CI coefficients in parallel
        if( nproc == 1 ):
            new_CIcoeffs_list = []
            for frag in self.tot_system.frag_list:
                new_CIcoeffs_list.append( -1j * self.delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore ) )
        else:
            frag_pool = multproc.Pool(nproc)
            new_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
            frag_pool.close()

        return new_rotmat_list, new_CIcoeffs_list

    #####################################################################

    def one_rk_step_mf( self, nproc ):
        #Subroutine to calculate one change in a runge-kutta step of any order
        #Using EOM that integrates CI coefficients and MF 1RDM 
        #embedding orbitals obtained by diagonalizing MF 1RDM at each step
        #Prior to calling this routine need to update MF 1RDM and CI coefficients

        #Calculate embedding orbitals maintaining same phase from previous step
        prev_rotmat = []
        for frag in self.tot_system.frag_list:
            prev_rotmat.append( np.copy(frag.rotmat) )

        self.tot_system.get_frag_rotmat()

        for cnt, frag in enumerate(self.tot_system.frag_list):
            phase = np.diag( np.round( np.diag( np.real( np.dot( utils.adjoint( prev_rotmat[cnt] ), frag.rotmat ) ) ) ) )
            frag.rotmat = np.dot( frag.rotmat, phase )

        #Calculate the rest of the terms needed for time-derivative of MF-1RDM
        self.tot_system.get_frag_corr1RDM()
        self.tot_system.get_glob1RDM()
        self.tot_system.get_nat_orbs()

        #Calculate embedding hamiltonian
        self.tot_system.get_frag_Hemb()

        #Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        #Calculate change in mf1RDM
        change_mf1RDM = self.delt * mf1rdm_timedep_mod.get_ddt_mf1rdm_serial( self.tot_system, int(self.tot_system.Nele/2) )

        #Calculate change in CI coefficients in parallel
        if( nproc == 1 ):
            change_CIcoeffs_list = []
            for frag in self.tot_system.frag_list:
                change_CIcoeffs_list.append( -1j * self.delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore ) )
        else:
            frag_pool = multproc.Pool(nproc)
            change_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
            frag_pool.close()

        return change_mf1RDM, change_CIcoeffs_list

    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate DMET energy, which also includes calculation of 1 & 2 RDMs and embedding hamiltonian for each fragment
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

#####################################################################

def applyham_wrapper( frag, delt ):

    #Subroutine to call pyscf to apply FCI hamiltonian onto FCI vector in dynamics
    #Includes the -1j*timestep term
    #The wrapper is necessary to parallelize using Pool and must be separate from
    #the class because the class includes IO file types (annoying and ugly but it works)

    return -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore )

#####################################################################



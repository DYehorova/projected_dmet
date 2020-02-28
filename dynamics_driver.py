#Routines to run a real-time projected-DMET calculation

import numpy as np
import system_mod
import utils
import applyham_pyscf
import mf1rdm_timedep_mod
import sys
import os
import multiprocessing as multproc
import pickle
import scipy

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

        if( not np.iscomplexobj( self.tot_system.mf1RDM ) ):
            self.tot_system.mf1RDM = self.tot_system.mf1RDM.astype(complex)

        #Set-up Hamiltonian for dynamics calculation
        self.tot_system.h_site = h_site
        self.tot_system.V_site = V_site

        #Define output files
        self.file_output   = open( 'output.dat', 'w' )
        self.file_corrdens = open( 'corr_density.dat', 'w' )

        self.file_test = open( 'test_data.dat', 'w' )#msh

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
            current_time = self.init_time + (step+1)*self.delt


        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_corrdens.close()
        self.file_test.close()#msh

        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

    #####################################################################

    def integrate( self, nproc ):
        #Subroutine to integrate equations of motion

        if( self.integ == 'rk4' ):
            #Use 4th order runge-kutta to integrate EOMs

            #Copy MF 1RDM, CI coefficients, and embedding orbs at time t
            init_mf1RDM = np.copy( self.tot_system.mf1RDM )
            init_CIcoeffs_list = []
            init_rotmat_list   = []
            for frag in self.tot_system.frag_list:
                init_rotmat_list.append( np.copy(frag.rotmat) )
                init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )

            #Calculate appropriate changes in MF 1RDM, embedding orbitals, and CI coefficients
            #Note that l, k and m terms are for MF 1RDM, emb orbs, and CI coefficients respectively
            l1, k1_list, m1_list = self.one_rk_step(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 0.5*l1
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m1_list[cnt]

            l2, k2_list, m2_list = self.one_rk_step(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 0.5*l2
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]

            l3, k3_list, m3_list = self.one_rk_step(nproc)

            self.tot_system.mf1RDM = init_mf1RDM + 1.0*l3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]

            l4, k4_list, m4_list = self.one_rk_step(nproc)

            #Update MF 1RDM, emb orbs and CI coefficients by full time-step
            self.tot_system.mf1RDM = init_mf1RDM + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
            for cnt, frag in enumerate( self.tot_system.frag_list ):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( m1_list[cnt] + 2.0*m2_list[cnt] + 2.0*m3_list[cnt] + m4_list[cnt] )

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

    def one_rk_step( self, nproc ):
        #Subroutine to calculate one change in a runge-kutta step of any order
        #Using EOM that integrates CI coefficients and MF 1RDM 
        #embedding orbitals obtained by diagonalizing MF 1RDM at each step
        #Prior to calling this routine need to update MF 1RDM and CI coefficients

        #calculate the terms needed for time-derivative of mf-1rdm
        self.tot_system.get_frag_corr1RDM()
        self.tot_system.get_glob1RDM()
        self.tot_system.get_nat_orbs()

        #Calculate change in mf1RDM
        change_mf1RDM = mf1rdm_timedep_mod.get_ddt_mf1rdm_serial( self.tot_system, round(self.tot_system.Nele/2) )

        #Use change in mf1RDM to calculate X-matrix for each fragment
        self.tot_system.get_frag_Xmat( change_mf1RDM )

        #Multiply change in mf1RDM by time-step
        change_mf1RDM *= self.delt

        #Calculate change in embedding orbitals
        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )

        #calculate embedding hamiltonian
        self.tot_system.get_frag_Hemb()

        #Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        #Calculate change in CI coefficients in parallel
        if( nproc == 1 ):
            change_CIcoeffs_list = []

            for ifrag, frag in enumerate(self.tot_system.frag_list):
                change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )

        else:
            frag_pool = multproc.Pool(nproc)
            change_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
            frag_pool.close()

        return change_mf1RDM, change_rotmat_list, change_CIcoeffs_list

    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate DMET energy, which also includes calculation of 1 & 2 RDMs and embedding hamiltonian for each fragment
        self.tot_system.get_DMET_E()

        #Calculate total number of electrons
        self.tot_system.get_DMET_Nele()

        ####msh####
        frag = self.tot_system.frag_list[0]
        #Efci = fci_mod.get_FCI_E( frag.h_emb, frag.V_emb, 0.0, frag.CIcoeffs, 2*frag.Nimp, frag.Nimp, frag.Nimp )
        #print( Efci, file=self.file_test )
        testdata = np.zeros(3+self.tot_system.Nsites)
        testdata[0] = current_time
        testdata[1] = np.linalg.norm( frag.CIcoeffs )**2
        testdata[2] = np.real( np.sum( np.diag( frag.corr1RDM ) ) )
        testdata[3:] = np.copy( np.real( np.diag( np.dot( utils.adjoint( frag.rotmat ), frag.rotmat ) ) ) )
        np.savetxt( self.file_test, testdata.reshape(1, testdata.shape[0]), fmt_str )
        self.file_test.flush()
        ###########


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

        #Save total system to file for restart purposes using pickle
        file_system = open( 'restart_system.dat', 'wb' )
        pickle.dump( self.tot_system, file_system )
        file_system.close()

#####################################################################

def applyham_wrapper( frag, delt ):

    #Subroutine to call pyscf to apply FCI hamiltonian onto FCI vector in dynamics
    #Includes the -1j*timestep term and the addition of bath-bath terms of X-matrix to embedding Hamiltonian
    #The wrapper is necessary to parallelize using Pool and must be separate from
    #the class because the class includes IO file types (annoying and ugly but it works)

    Xmat_sml = np.zeros( [ 2*frag.Nimp, 2*frag.Nimp ], dtype = complex )
    Xmat_sml[ frag.Nimp:, frag.Nimp: ] = frag.Xmat[ frag.bathrange[:,None], frag.bathrange ]

    return -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb-Xmat_sml, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore )

#####################################################################



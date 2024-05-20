import numpy as np
import os
import json
import time
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.linalg import eig
from scipy.linalg import solve
from .utils import subfunctions as subf


class ComplBeam:
    """ ComplBeam model with the Equlibrium-based beam element. 
        The element allows for elements with varying twist and properties.

        Required input in mainInputFile
            - Input file names (properties, c2-pos, BC and loads)
            - "Analysistype" ("static" or "dynamic")
            - "int_type" ("gauss" or "trapz") 
            - "Nip" (number of integration points)
            - "mass_matrix" = "Timo" or "Compl" 

        Descriptions of the code is seen in the Master Thesis: 
        " Modelling of Wind Turbine Blades by Beam Elements from Complementary Energy" 
        
    """

    def __init__(self, mainInputFile):
        self.nodeLocations = None
        tic = time.perf_counter()
        print('-----------')

        ### INPUT ####
        # Define main directory
        self.maindir = os.path.dirname(mainInputFile)
        # Read main input file
        self.readMainInputFile(mainInputFile)
        # Spanwise direction unit vector (do not change this!)
        self.v1 = np.array([.0, .0, 1.0])
        # Read input files written in main input file
        try:
            self.readInputs(inputfolder=os.path.join(self.maindir))
        except:
            print('Could not read input files')
            return None
        # Number of nodes
        self.numNode = self.c2Input.shape[0]
        # Number of elements
        self.numElement = self.numNode - 1

        ### GEOMETRY AND PROPERTIES ###
        # Scurve distance through 1/2chord location in cross section at c2 points
        self.createScurve()
        # Interpolation of scurve and properties between nodes and properties points
        self.createInterpolators()
        # Node location as elastic center at c2 points
        self.calculateNodeLocations()
        # Average element properties
        self.calculalteIntPoints()
        # Cross-sectional matrices along the element coordinate systems
        self.calculateCrossSectionMatrices()

         ### ASSEMBLE MODEL ###
        # Element stiffness and mass matrix
        self.calculateElementMatrixComplimentary()
        # Interpolation function for displacements

        


        ## Edit: Julio Vázquez M.-L.
        # In the section where the mass matrix is calculated, the analysis type flag is set to "dynamic"
        # New begin
        self.analysistype = "dynamic"
        # New end
        ## Edit end

        if self.analysistype == "dynamic":
            if self.mass_matrix_type == 'Compl':
                self.calcDisplacementFieldCompl()
                self.massmatrixcomplementary()
                print('Compl Mass Matrix')
            elif self.mass_matrix_type == 'Timo':
                self.massmatrixoriginal()
                print('Timo Mass Matrix')
            else:
                print('Wrong mass matrix calculation type input given')
                self.massmatrixoriginal()
                print('Timo Mass Matrix is selected as default')

        # Global stiffness and mass matrix
        self.calculateGeneralMatrix()

        ## Edit: Julio Vázquez M.-L.
        # New begin
        self.analysistype = "static"
        # New end 

        ### STATIC LOADS ###
        # Initialize
        self.force = np.zeros((self.numNode*6))
        # Append static loads if defined in input-file
        if 'static_load' in self.inputFileNames:
            self.calculateLoads()
        if 'static_load_distributed' in self.inputFileNames or 'static_load_distributed_segment' in self.inputFileNames:
            self.calculateqtilde()
            self.calculateELementLoads()

        ### BOUNDARY CONDITIONS ###
        # Apply boundary conditions
        self.applyBoundary()
        print('ComplBeam Model Created')

         ### ANALYSIS ###
        # Static/ dynamic analysis if given in input file
        if self.analysistype == "dynamic":
            self.calcModeShapes()
            print('Dynamic analysis done')
        elif self.analysistype == "static":
            self.calcDeflection()
            print('Static analysis done')
            #self.calcDisplacementFieldCompl()
        else:
            print("No analysis chosen")

        ### OUTPUT ###
        toc = time.perf_counter() # Total runtime
        self.runTime = toc-tic

    def readMainInputFile(self, mainInputFile):
        """"This function reads the main input file"""

        try:
            with open(mainInputFile) as f:
                self.inputFileNames = json.load(f)
        except (IOError, FileNotFoundError) as error:
            print('********** Could not read the main input file ***********')
            print(error)
            raise
        except json.decoder.JSONDecodeError as error:
            print('********** Format error in main input file **********')
            print(error)
            raise

        if 'Analysistype' in self.inputFileNames:
            self.analysistype = self.inputFileNames['Analysistype']
        else:
            print('**********                   No Analysistype is defined                  **********')
            print('**********  "Analysistype" = "static" or "dynamic" (default = "dynamic") **********')
            self.analysistype = 'dynamic'
            
        if 'Nip' in self.inputFileNames:
            self.nip = self.inputFileNames['Nip']
        else:
            self.nip = 6
            print('********** No number of integration points defined **********')
            print('**********              "Nip" = 6                  **********')

        if 'mass_matrix' in self.inputFileNames:
            self.mass_matrix_type = self.inputFileNames['mass_matrix']
        else:
            self.mass_matrix_type = 'Timo'
            if self.analysistype == "dynamic":
                print('**********                  No \"mass_matrix\" type defined                    **********')
                print('**********          "mass_matrix" = "Timo" / "Compl" (default = "Timo")        **********')
                
            
        if 'int_type' in self.inputFileNames:
            self.intType = self.inputFileNames['int_type']
        else:
            print('**********                    No integration type defined                     **********')
            print('**********          "int_type" = "trapz" / "gauss" (default = "gauss")        **********')
            self.intType = 'gauss'

    def readInputs(self, inputfolder = ''):
        """ This function reads the input files given in the main input file"""

        # Read structural properties
        # Note: the first 2 rows with text are skipped and the 3rd row is properties names
        #       The order of properties columns can not be changed!        
        try:
            stPropertyInput = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['Properties']), skiprows=3)
            if stPropertyInput.shape[1] < 29:# Assumes isotropic if less than 29 properties variables
                self.ConvertIso2Aniso(stPropertyInput)
                self.stProperty_iso = stPropertyInput.copy()
            else: # Assumes anisotropic
                self.stProperty = stPropertyInput.copy()
                with open(os.path.join(inputfolder, self.inputFileNames['Properties'])) as f:
                    lines = f.readlines()
                    self.propnames = [i.strip() for i in lines[1].split(' ') if not i.startswith('[') if not i.strip() == ''][2::]

        except (IOError, FileNotFoundError) as error:
            print('********** Could not read the structural property file **********')
            print(error)
            raise

        # Read c2 definition
        # Note: The first 2 rows are skipped
        try:
            self.c2Input = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['c2_pos']), skiprows=2)
        except (IOError, FileNotFoundError) as error:
            print('********** Could not read the c2_pos file **********')
            print(error)
            raise

        # Read boundary Data
        # Note: The first 1 row for text is skipped
        try:
            self.boundaryData = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['Boundary']), skiprows=1).astype(int)
            if self.boundaryData.ndim == 1:
                self.boundaryData = self.boundaryData[np.newaxis, :]
        except (IOError, FileNotFoundError, KeyError) as error:
            print('********** Could not read the boundary file **********')
            print(error)

        # Read static nodal load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load' in self.inputFileNames:
            try:
                self.loadData = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['static_load']), skiprows=1)
                if self.loadData.ndim == 1:
                    self.loadData = self.loadData[np.newaxis, :]
            except (IOError, FileNotFoundError, KeyError) as error:
                print('********** Could not read the force file **********')
                print(error)

        # Read static distributed load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load_distributed' in self.inputFileNames:
            try:
                self.loadDataDistributed = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['static_load_distributed']), skiprows=1)
                if self.loadDataDistributed.ndim == 1:
                    self.loadDataDistributed = self.loadDataDistributed[np.newaxis, :]
            except (IOError, FileNotFoundError, KeyError) as error:
                print('Could not read the distributed force file')
                print(error)
        # Read static distributed load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load_distributed_segment' in self.inputFileNames:
            try:
                self.loadDataDistributedSegment = np.loadtxt(os.path.join(inputfolder, self.inputFileNames['static_load_distributed_segment']), skiprows=1)
                if self.loadDataDistributedSegment.ndim == 1:
                    self.loadDataDistributedSegment = self.loadDataDistributedSegment[np.newaxis, :]
            except (IOError, FileNotFoundError, KeyError) as error:
                print('Could not read the segment distributed force file')
                print(error)

    def ConvertIso2Aniso(self,stPropertyInput):
        """ This function converts an HAWC2 isotropic input file to the corresponding anisotropic input"""
        # Properties names
        self.propnames = ['m', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch', 'x_e', 'y_e', 'K_11', 'K_12', 'K_13', 'K_14', 'K_15', 'K_16', 'K_22', 'K_23', 'K_24', 'K_25', 'K_26', 'K_33', 'K_34', 'K_35', 'K_36', 'K_44', 'K_45', 'K_46', 'K_55', 'K_56', 'K_66']
        self.stProperty = np.zeros((stPropertyInput.shape[0], 30))
        # Standard parameters
        self.stProperty[:, 0] = stPropertyInput[:, 0]   # r
        self.stProperty[:, 1] = stPropertyInput[:, 1]   # m
        self.stProperty[:, 2] = stPropertyInput[:, 2]   # x_cg
        self.stProperty[:, 3] = stPropertyInput[:, 3]   # y_cg
        self.stProperty[:, 4] = stPropertyInput[:, 4]   # ri_x
        self.stProperty[:, 5] = stPropertyInput[:, 5]   # ri_y
        self.stProperty[:, 6] = stPropertyInput[:, 16]  # pitch
        self.stProperty[:, 7] = stPropertyInput[:, 17]  # x_e
        self.stProperty[:, 8] = stPropertyInput[:, 18]  # y_e

        # Assemble cross sectional stiffness matrix
        for i in range(np.shape(self.stProperty)[0]):
            # Rotation matrix from c2 to elastic coordinates
            t_mat = subf.get_tsb(self.v1, self.v1, np.deg2rad(self.stProperty[i, 6]))
            svec = t_mat.T @ np.array([stPropertyInput[i, 6], stPropertyInput[i, 7], 0])
            evec = t_mat.T @ np.array([stPropertyInput[i, 17], stPropertyInput[i, 18], 0])

            # Cross-sectional stiffness matrix
            self.stProperty[i, 9]  = stPropertyInput[i, 13] * stPropertyInput[i, 15] * stPropertyInput[i, 9]
            self.stProperty[i, 14] = - self.stProperty[i, 9] *  (svec[1]-evec[1])
            self.stProperty[i, 15] = stPropertyInput[i, 14] * stPropertyInput[i, 15] * stPropertyInput[i, 9]
            self.stProperty[i, 19] = self.stProperty[i, 15] * (svec[0]-evec[0])
            self.stProperty[i, 20] = stPropertyInput[i, 8] * stPropertyInput[i, 15]
            self.stProperty[i, 24] = stPropertyInput[i, 8] * stPropertyInput[i, 10]
            self.stProperty[i, 27] = stPropertyInput[i, 8] * stPropertyInput[i, 11]
            self.stProperty[i, 29] = stPropertyInput[i, 9] * stPropertyInput[i, 12] + self.stProperty[i, 19] *  (svec[0]-evec[0]) - self.stProperty[i, 14]*(svec[1]-evec[1])

    def createScurve(self):
        """ This function calculates the scurve distance through 1/2chord location in cross-section at c2 points.
            The distance is defined as the linear distance between each point."""
        # Initiate scurve break points
        s = np.zeros((self.numNode,))
        self.distanceBtwScurvePoints = np.linalg.norm(np.diff(self.c2Input[:, 1:4], axis=0), axis=1)
        s[1::] = np.cumsum(self.distanceBtwScurvePoints)
        # Total scurve length
        self.scurveLength = s[-1]
        # Non-dimensional scurve distance at c2 points/ nodes
        self.scurve = s/self.scurveLength

    def createInterpolators(self):
        """ This function creates interpolation functions along the scurve for properties and c2-line position.
            c2Pos: akima interpolation (4th order polynomial) of scurve position between c2 input.
            c2Twist: akima interpolation of twist in c2 file between c2 input.
            structuralPropertyInterpolator: Linear interpolation of properties between properties input points. 

            Note: the r-distance in the properties file is adjusted to match the scurve length.
            """
        # C2 akima Interpolators
        c2Pos = self.c2Input[:, 1:4]
        c2Twist = self.c2Input[:, 4]
        s = self.scurve

        if self.numNode<3: # uses midpoint if only two nodes
            a = np.array([s[0],(s[0]+s[1])/2,s[1]])
            b = np.zeros((3, 3))
            b[0, :] = c2Pos[0, :]
            b[-1, :] = c2Pos[-1, :]
            b[1, :] = (c2Pos[0, :] + c2Pos[1, :])/2
            self.scurvePosInterpolator  = Akima1DInterpolator(a, b)
            b = np.array([c2Twist[0],(c2Twist[0]+c2Twist[1])/2, c2Twist[1]])
            self.scurveTwistInterpolator  = Akima1DInterpolator(a, np.deg2rad(b))
        else:
            self.scurvePosInterpolator  = Akima1DInterpolator(s, c2Pos)
            self.scurveTwistInterpolator  = Akima1DInterpolator(s, np.deg2rad(c2Twist))

        # Structural property interpolators
        st_prop_r,st_prop = self.stProperty[:, 0], self.stProperty[:, 1:]
        st_prop_r = st_prop_r /st_prop_r[-1] # r-distance adjusted to non-dim scurve length
        self.struturalPropertyInterploator = {key: interp1d(st_prop_r, st_prop[:, i]) for i, key in enumerate(self.propnames)}

    def calculateNodeLocations(self):
        """"This function calculates the location of the nodes in the elastic center in the c2-points"""
        # Distance to elastic center from each c2 coordinate
        ec_s = np.zeros((self.numNode, 3))
        ec_s[:, 0], ec_s[:, 1] = self.struturalPropertyInterploator['x_e'](self.scurve), self.struturalPropertyInterploator['y_e'](self.scurve)

        # Tangent of scurve at c2-points
        self.scurveTangent = self.scurvePosInterpolator.derivative(1)(self.scurve)

        # Initiate node positions
        nodeLocations = np.zeros((self.numNode, 3))
        for i in range(self.numNode):
            nodeLocations[i, :] = self.c2Input[i, 1:4] + subf.get_tsb(self.v1, self.scurveTangent[i, :], np.deg2rad(self.c2Input[i, 4])) @ ec_s[i, :]
        self.nodeLocations = nodeLocations

    def calculalteIntPoints(self):
        # Non-dimensional points where the integrals are evaluated, including integral weights

        # Gauss integration
        if self.intType == 'gauss':
            xi, wi = subf.gaussPts(self.nip, -1, 1)
        # Trapz integration
        elif self.intType == 'trapz':
            xi = np.linspace(-1, 1,self.nip)
            wi = np.ones(np.shape(xi)[0])
            wi[1:-1] = 2
            wi = (xi[1]-xi[0])/2 * wi

        self.int_points = np.outer(np.ones(self.numElement), xi)
        self.int_weights = np.outer(np.ones(self.numElement), wi)

    def calculateCrossSectionMatrices(self):
        """ This function calculates the cross-sectional flexibility matrix 
            and mass matrix at the integration points. The matrices are 
            rotated to no twist (no angle between the x-axis and the XZ-plane). """

        # Initialize matrices
        D_mat = np.zeros((self.numElement,self.nip, 6, 6))
        C_mat = np.zeros((self.numElement,self.nip, 6, 6))
        M_sec = np.zeros((self.numElement,self.nip, 6, 6))
        # Loop over elements
        for i in range(self.numElement):

            # Location of integration points
            xi = self.int_points[i, :]

            # Loop over integration points
            for j in range(self.nip):
                # Non-dimensional location on beam
                scurvepos = (((xi[j]+1)/2)*(self.scurve[i+1]-self.scurve[i])+self.scurve[i])

                # Calculate total twist
                el_p1 =np.deg2rad( self.struturalPropertyInterploator['pitch'](scurvepos))
                el_p2 = self.scurveTwistInterpolator(scurvepos)
                el_p0 = el_p1 + el_p2

                # 6x6 Rotation matrix (from element to global basis, only twist)
                t_mat_rot = subf.get_tsb(self.v1, self.v1, el_p0)
                Tmat_rot = np.kron(np.eye(2, dtype=int), t_mat_rot)

                # Cross sectional flexibility matrix
                props = np.zeros(21)
                for jj, key in enumerate(self.propnames[8:]):
                    props[jj] = self.struturalPropertyInterploator[key](scurvepos)
                D_mat[i,j, :, :] = Tmat_rot @ subf.getSectionalStiffness(props) @ Tmat_rot.T

                # Cross sectional stiffness matrix with aero & structural twist
                C_mat[i,j, :, :] = subf.inv(D_mat[i,j, :, :]) 

                # mass parameters
                el_p_local = np.zeros(5)
                el_p_local[0] = self.struturalPropertyInterploator['m'](scurvepos)

                # Rotation matrix from c2 to elastic coordinates
                t_mat_c2toelem = subf.get_tsb(self.v1, self.v1, el_p1)

                # mass center x- and y-locations from elastic center
                cgvec = t_mat_c2toelem.T @ np.array([self.struturalPropertyInterploator['x_cg'](scurvepos), self.struturalPropertyInterploator['y_cg'](scurvepos), 0])
                evec = t_mat_c2toelem.T @ np.array([self.struturalPropertyInterploator['x_e'](scurvepos), self.struturalPropertyInterploator['y_e'](scurvepos), 0])
                el_p_local[1] = cgvec[0] - evec[0]
                el_p_local[2] = cgvec[1] - evec[1]

                # other properties
                for jj, key in enumerate(self.propnames[3:5]):
                    el_p_local[jj+3] = self.struturalPropertyInterploator[key](scurvepos)
                M_sec[i, j, :, :] = Tmat_rot @ subf.getSectionalMass(el_p_local) @ Tmat_rot.T

        # Store matrices in class 
        self.C_mat_twisted = C_mat
        self.D_mat_twisted = D_mat
        self.M_sec_twisted = M_sec

    def calculateElementMatrixComplimentary(self):
        """ This function calculates the elastic stiffness matrix the complementary beam element."""
        # Number of integration points
        nip = self.nip

        # Initilization of matrices
        Kele = np.zeros((self.numElement, 12, 12))
        Hinvele = np.zeros((self.numElement, 6, 6))
        Kd = np.zeros((self.numElement, 6, 6))
        R_init = np.zeros((self.numElement, 3, 3)) # initial element rotation matrix used in corot code
        L_init = np.zeros((self.numElement)) # initial element length used in corot code

        # Loop over elements 
        for i in range(self.numElement):
            # Gauss Points
            xi = self.int_points[i, :]
            wi = self.int_weights[i, :]
            # Vector between nodes
            diffNodes = self.nodeLocations[i+1, :] - self.nodeLocations[i, :]
            # Element length
            L = np.linalg.norm(diffNodes)
            L_init[i] = L
            # H matrix in equation 12 in couturier (2017)
            Hmat = np.zeros((6, 6))
            # Matrix to rotate a vector from local to global axes (q_global = T * q_elem)
            t_mat = subf.get_tsb(self.v1, diffNodes, 0)
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)
            R_init[i,:,:] = t_mat

            # Gauss/ Trapz integration
            for j in range(nip):
                # Cross sectional stiffness matrix
                Cmat = self.C_mat_twisted[i, j, :, :]

                # Element flexibility matrix
                Hmat += L/2 * wi[j] * (self.Tmat_complimentary(xi[j], L/2).T @ Cmat @ self.Tmat_complimentary(xi[j], L/2))

            # Inverse of element equlibrium-mode flexibility matrix
            Hinv = subf.inv(Hmat)

            # G matrix eq 16 in Couturier (2017)
            Gmat = np.concatenate([-1*self.Tmat_complimentary(-1, L/2), self.Tmat_complimentary(1, L/2)])

            # F_0_inv for Kd computation which is needed for Corot code
            # Kd = F_0_inv^-1 * Hinv * F_0_inv^-T where Gmat = S * F_0_inv (see jupyter page) S matrix is from Krenk's book
            F_0_inv = np.zeros((6,6))
            F_0_inv[0,1] = -L/2
            F_0_inv[1,0] = L/2
            F_0_inv[2,2] = -1
            F_0_inv[3,3] = -1
            F_0_inv[4,4] = -1
            F_0_inv[5,5] = -1
            
            Kd[i,:,:] = F_0_inv @ Hinv @ F_0_inv.T 

            # Element stiffness matrix from equation 21 in Coutureier (2017)
            # transformed to global coordinate system 
            Kele[i, :, :] = Tmat @ (Gmat @ Hinv @ Gmat.T) @ Tmat.T
            Hinvele[i, :, :] = Hinv
        # Store matrices
        self.Hinvele_twisted = Hinvele
        self.Kele = Kele
        self.Kd = Kd
        self.R_init = R_init
        self.L_init = L_init

    def massmatrixcomplementary(self):
        # Number of integration points
        nip = self.nip

        # Initilization of matrices
        Mele = np.zeros((self.numElement, 12, 12))

         # Loop over elements
        for i in range(self.numElement):
            # integtion weights (at general xi points)
            wi = self.int_weights[i, :]
            # Element length
            diffNodes = self.nodeLocations[i+1, :] - self.nodeLocations[i, :]
            L = np.linalg.norm(diffNodes)

            # Matrix to rotate a vector from local to global axes (q_global = T * q_elem)
            t_mat = subf.get_tsb(self.v1, diffNodes, 0)
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)

            # Get sectional mass matrix            
            M_int = np.zeros((12, 12))

            # Gauss integration
            for j in range(nip):
                # Cross sectional mass matrix 
                M_sec = self.M_sec_twisted[i, j, :, :]
                # Shape function matrix based on Compl deflection
                N = self.interpNmat_notwist[i, j, :, :]
                # Integration
                M_int += wi[j] * L/2 * (N.T @ M_sec @ N)

            # Element stiffness matrix from equation 21 in Coutureier (2017)
            # transformed to global coordinate system 
            Mele[i, :, :] = Tmat @ M_int @ Tmat.T
        self.Mele = Mele

    def massmatrixoriginal(self):
        """ This function calculates the the mass matrix of each element.
            Equation references to "Development of an anisotropic beam finite element for composite wind 
            turbine blades in multibody system" by Kim et al. (2013)."""     
        nip = self.nip

        # Polynomium order +1
        ni = 6 # TODO update to input variable

        # Initialization
        Mele = np.zeros((self.numElement, 12, 12))

        # A matrices from Eq-11 in the paper
        A1 = np.zeros((6*ni, 12))
        A2 = np.zeros((6*ni, 6*ni-12))
        A1[:12, :] = np.eye(12)
        A2[12:, :] = np.eye(6*ni-12)

        # Loop over elements
        for i in range(self.numElement):
            # Vector between nodes
            diffNodes = self.nodeLocations[i+1, :] - self.nodeLocations[i, :]
            # Element length
            L = np.linalg.norm(diffNodes)

            # Rotation matrix
            t_mat = subf.get_tsb(self.v1, diffNodes, 0)
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)

            # Gauss Points
            xi = self.int_points[i, :]
            wi = self.int_weights[i, :]

            # D matrix from eq-8 in the paper
            D_mat = np.zeros((6*ni, 6*ni))
            M_int = np.zeros_like(D_mat)

            # gauss integration 
            for j in range(nip):
                # Location on element in [m]
                z = (xi[j]+1)/2*L
                # Strain displacement matrix and polynomial matrix
                B, N = subf.getStrainDisplacementMatrix(z, ni)

                # Cross sectional flexibility and mass matrix without twist
                M_sec = self.M_sec_twisted[i, j, :, :]
                S_mat = self.D_mat_twisted[i, j, :, :]

                # Integration
                D_mat += wi[j]*L/2* (B.T @ S_mat @ B)
                M_int += wi[j]*L/2 * (N.T @ M_sec @ N)

            # N matrices from Eq-10 in the paper
            N1 = np.zeros((12, 12))
            N1[:6, :6] = N1[6:, :6] = np.eye(6)
            N1[6:, 6:] = np.eye(6)*L
            N2 = np.zeros((12, 6*(ni-2)))
            for j in range(ni-2):
                N2[6:, j*6:(j+1)*6] = np.eye(6)*L**(j+2)
            # Y matrices from eq-13 in the paper
            Y1 = A1 @ subf.inv(N1)
            Y2 = A2 - Y1 @ N2
            # Q, P and N_alpha from eq-15 and 16 in the paper
            P = Y2.T @ D_mat @ Y1
            Q = -1 * (Y2.T @ D_mat @ Y2)
            N_a = Y1 + (Y2 @ subf.inv(Q) @ P)
            # Kel
            Mele[i, :, :] = Tmat @ (N_a.T @ M_int @ N_a) @ Tmat.T
        # Save to class
        self.Mele = Mele  # Element mass matrices

    def calculateGeneralMatrix(self):
        """ This function assembles the global stiffness and mass matrix"""
        # Initialize
        K_mat = np.zeros((self.numNode*6, self.numNode*6))

        if self.analysistype == 'dynamic':
            M_mat = np.zeros_like(K_mat)

        # Loop over elements
        for i in range(self.numElement):
            d1 = i * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            d2 = (i+1) * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            dof = np.hstack((d1, d2))
            K_mat[np.ix_(dof, dof)] += self.Kele[i, :, :]
            if self.analysistype == 'dynamic':
                M_mat[np.ix_(dof, dof)] += self.Mele[i, :, :]
        self.K_mat_full = K_mat
        if self.analysistype == 'dynamic':
            self.M_mat_full = M_mat

    def calculateLoads(self):
        """ This function defines nodal loads given in the main coordinate system"""
        # Loop over load inputs
        for i in range(np.shape(self.loadData)[0]):
                i_node = int(self.loadData[i, 0])-1
                self.force[6*i_node:6*i_node+6] += self.loadData[i, 1:]

    def calculateqtilde(self):
        # Number of integration points
        nip = self.nip
        if ('static_load_distributed' in self.inputFileNames or'static_load_distributed_segment' in self.inputFileNames):
            self.qtilde_notwist = np.zeros((self.numElement,self.nip, 6))
            self.gvec = np.zeros((self.numElement,12))

        if 'static_load_distributed' in self.inputFileNames:
            # Loop over load inputs
            for i in range(np.shape(self.loadDataDistributed)[0]):
                # Element index
                i_elem = int(self.loadDataDistributed[i, 0])-1
                # Gauss Points
                xi = self.int_points[i_elem, :]
                # Length of element
                diffNodes = self.nodeLocations[i_elem+1, :] - self.nodeLocations[i_elem, :]
                L = np.linalg.norm(diffNodes)
                # Matrix to rotate from local to global axes (without twist)
                t_mat = subf.get_tsb(self.v1, diffNodes, 0)
                Tmat = np.kron(np.eye(4, dtype=int), t_mat)
                # Load values at the end of the element in element coordinate system
                P = Tmat.T @ self.loadDataDistributed[i, 1:]
                
                # g vector from equation 26 in coutourier (2017)
                self.gvec[i_elem,:] += np.concatenate([-1*self.getqtilde_linearload(P[0:6], P[6:], L/2, -1), self.getqtilde_linearload(P[0:6], P[6:], L/2, 1)], 0)

                # Gauss integration
                for j in range(nip):
                    # h vector
                    self.qtilde_notwist[i_elem,j,:] += self.getqtilde_linearload(P[0:6], P[6:], L/2, xi[j])
                
        if 'static_load_distributed_segment' in self.inputFileNames:
            # Loop over load inputs
            for i in range(np.shape(self.loadDataDistributedSegment)[0]):
                # Element index
                i_elem = int(self.loadDataDistributedSegment[i, 0])-1
                # Length of element
                diffNodes = self.nodeLocations[i_elem+1, :] - self.nodeLocations[i_elem, :]
                L = np.linalg.norm(diffNodes)
                 # Gauss Points
                xi = self.int_points[i_elem, :]

                # Matrix to rotate from local to global axes (without twist)
                t_mat = subf.get_tsb(self.v1, diffNodes, 0)
                Tmat = np.kron(np.eye(2, dtype=int), t_mat)

                # Load values at the end of the element in element coordinate system
                z1_nondim = self.loadDataDistributedSegment[i, 1]
                z2_nondim = self.loadDataDistributedSegment[i, 2]
                pvec1 = Tmat.T @ self.loadDataDistributedSegment[i, 3:9]
                pvec2 = Tmat.T @ self.loadDataDistributedSegment[i, 9:]

                # g vector from equation 26 in coutourier (2017)
                self.gvec[i_elem,:] += np.concatenate([-1*self.getqtilde_segment(pvec1, pvec2, z1_nondim*L, z2_nondim*L, 0), self.getqtilde_segment(pvec1, pvec2, z1_nondim*L, z2_nondim*L, L)]  , 0)

                # Gauss integration
                for j in range(nip):
                    self.qtilde_notwist[i_elem,j,:] += self.getqtilde_segment(pvec1, pvec2, z1_nondim*L, z2_nondim*L,(xi[j]+1)/2*L)
    
    def calculateELementLoads(self):
        """ This function calculates equivalent nodal loads for linear varying distributed loads
            along the element. The load directions are defined in the main coordinate system."""
        if hasattr(self, 'qtilde_notwist'): 
            if not hasattr(self, 'force'):
                self.force = np.zeros((self.numNode*6))
            if not hasattr(self, 'hvec_twisted'):
                self.hvec_twisted = np.zeros((self.numElement, 6))

            # Number of integration points
            nip = self.nip

            # Loop over load inputs
            for i_elem in range(self.numElement):
                
                # Length of element
                diffNodes = self.nodeLocations[i_elem+1, :] - self.nodeLocations[i_elem, :]
                L = np.linalg.norm(diffNodes)

                # Gauss Points
                xi = self.int_points[i_elem, :]
                wi = self.int_weights[i_elem, :]

                # G matrix eq 16 in Couturier (2017)
                Gmat = np.concatenate([-1*self.Tmat_complimentary(-1, L/2), self.Tmat_complimentary(1,  L/2)])
                
                gvec = self.gvec[i_elem,:]

                # h vector from equation 28 in coutourier (2017)
                hvec = np.zeros(6)

                # Gauss integration
                for j in range(nip):
                    # Cross sectional stiffness matrix
                    Cmat = self.C_mat_twisted[i_elem, j, :, :]
                    qtilde_notwist = self.qtilde_notwist[i_elem,j,:]

                    # h vector
                    hvec += L/2 * wi[j] * (self.Tmat_complimentary(xi[j], L/2).T @ Cmat @ qtilde_notwist)
                self.hvec_twisted[i_elem, :] = hvec

                # Inverse of element equlibrium-mode flexibility matrix
                Hinv = self.Hinvele_twisted[i_elem, :, :]

                # Nodal forces from equation 32 in coutourier et al
                t_mat = subf.get_tsb(self.v1, diffNodes, 0)
                Tmat = np.kron(np.eye(4, dtype=int), t_mat)
                rvec = Tmat @ (Gmat @ Hinv @ hvec - gvec)

                # Append to load vector
                self.force[i_elem*6:i_elem*6+12] += rvec

    def applyBoundary(self):
        """ This function applies boundary condition and store full matrices"""
        # Free and fixed dofs
        idx_delete = np.array([], dtype=int)
        for i_node, idx in zip(self.boundaryData[:, 0], self.boundaryData[:, 1::]):
            temp = np.array([6*(i_node-1)+j for j, k in enumerate(idx) if k == 0])
            idx_delete = np.append(idx_delete, temp)

        self.dofs = np.arange(self.numNode*6)
        self.dofs_fixed = idx_delete
        self.dofs_free = np.setdiff1d(self.dofs, self.dofs_fixed)

        # delete constrained dofs from K_mat_full
        self.K_mat = np.delete(self.K_mat_full, idx_delete, axis=1)
        self.K_mat = np.delete(self.K_mat, idx_delete, axis=0)

        # delete constrained dofs from M_mat_full
        if "M_mat_full" in self.__dict__:
            self.M_mat = np.delete(self.M_mat_full, idx_delete, axis=1)
            self.M_mat = np.delete(self.M_mat, idx_delete, axis=0)

        # Delete contrained dofs from load vector
        if "force" in self.__dict__:
            self.force_full = self.force
            self.force = np.delete(self.force, idx_delete )

    def calcModeShapes(self):
        """ This function solves the eigen-value problem and calculates the mode shapes"""
        assert all(hasattr(self, attr) for attr in ["K_mat", "M_mat"]),'First call "applyBoundary" before calcModeShapes'
        w, eig_v = eig(self.K_mat, self.M_mat)
        sorted_idx = np.argsort(w.real)
        sorted_w = w.real[sorted_idx]
        sorted_w[sorted_w < 1e-5] = 0

        self.eigen_vec = eig_v[:, sorted_idx]
        self.freqs = np.sqrt(sorted_w)/(2*np.pi)
        for i_mode in range(self.freqs.shape[0]):
            eig_val = self.eigen_vec[:, i_mode]
            absval = abs(eig_val)
            idx = np.argmax(absval)
            self.eigen_vec[:, i_mode] = eig_val/absval.max() * np.sign(eig_val[idx])

        self.eigen_vec_full = np.zeros([self.numNode*6, np.shape(self.eigen_vec)[1]])
        self.eigen_vec_full[self.dofs_free, :] = self.eigen_vec

    def calcDeflection(self):
        """ This function calcualtes the static deflection"""
        self.defl = solve(self.K_mat, self.force)
        self.defl_full = np.zeros(self.numNode*6)
        self.defl_full[self.dofs_free] = self.defl

    def calcDisplacementFieldCompl(self):
        # Initialization
        if not hasattr(self, 'qtilde_notwist'):
            qtilde = np.zeros((self.numElement,self.nip, 6))
            print('zero qtilde')
        else:
            qtilde = self.qtilde_notwist
        if not hasattr(self, 'hvec_twisted'):
            hvec = np.zeros((self.numElement, 6))
            print('zero hvec')
        else:
            hvec = self.hvec_twisted
        
        # Number of integration points
        nip = self.nip

        # Allocating main matrices
        Hinvmat = self.Hinvele_twisted
        interpNmat = np.zeros((self.numElement, nip, 6, 12))
        interpNcor = np.zeros((self.numElement, nip, 6))
        interpPos = np.zeros((self.numElement, nip))
        
        # Connectivity matrices 
        Tm = np.zeros((6, 6))
        Tm[4, 0] = -1
        Tm[3, 1] = 1
        Imat = np.eye(6)
        IImat = np.concatenate([Imat, np.zeros((6, 6))], axis=1)

        # Loop over elements
        for i in range(self.numElement):
            # Integration Points
            xi = self.int_points[i, :]

            # Length of entire element
            diffNodes = self.nodeLocations[i+1, :] - self.nodeLocations[i, :]
            L = np.linalg.norm(diffNodes)
            a = L/2

            # G matrix eq 16 in Couturier (2017)
            Gmat = np.concatenate([-1*self.Tmat_complimentary(-1, a), self.Tmat_complimentary(1, a)]) 

            # Initialization
            Amat = np.zeros((6, 6))
            Adistr = np.zeros(6)
            Bdistr = np.zeros(6)
            Bmat = np.zeros((6, 6))

            # Loop over calculation segments
            for j in range(1, nip):
                # Element segment lengts
                z1 = (1 + xi[j-1])/ 2 * (self.scurve[i+1] - self.scurve[i]) * self.scurveLength
                z2 = (1 + xi[j])/ 2 * (self.scurve[i+1] - self.scurve[i]) * self.scurveLength

                # Cross sectional stiffness matrix from global scurve position 
                Cmat1 = self.C_mat_twisted[i, j-1, :, :]
                Cmat2 = self.C_mat_twisted[i, j, :, :]

                # Calculate T0 matrix w.r.t element mid
                T01 = (Imat + (z1-a)*Tm)
                T02 = (Imat + (z2-a)*Tm)

                # Increase A and B integrals
                Amat += (z2-z1)/2 * ((Imat+z1 * Tm).T @ Cmat1 @ T01 + (Imat+z2 * Tm).T @ Cmat2 @ T02)
                Bmat += (z2-z1)/2 * (Tm.T @ Cmat1 @ T01 + Tm.T @ Cmat2 @ T02)

                N_alpha = Amat  @ Hinvmat[i, :, :] @ Gmat.T + Imat @ IImat
                N_beta = Bmat  @ Hinvmat[i, :, :] @ Gmat.T + Tm.T @ IImat
                Nmat = N_alpha - z2 * N_beta

                # Correction of shape functions in case of distributed load
                Adistr += (z2-z1)/2 * ((Imat+z1 * Tm).T @ Cmat1 @ qtilde[i, j-1, :] + (Imat+z2 * Tm).T @ Cmat2 @ qtilde[i, j, :])
                Bdistr += (z2-z1)/2 * (Tm.T @ Cmat1 @ qtilde[i, j-1, :] + Tm.T @ Cmat2 @ qtilde[i, j, :])

                N_alpha_distr =Adistr - Amat  @ Hinvmat[i, :, :] @ hvec[i,:]
                N_beta_distr  =Bdistr - Bmat  @ Hinvmat[i, :, :] @ hvec[i,:]

                # Correction vector
                Ncor = N_alpha_distr - z2 * N_beta_distr

                # Save matrices
                interpNmat[i,j, :, :] = Nmat
                interpPos[i,j] = z2 +self.scurve[i] * self.scurveLength
                interpNcor[i,j, :] = Ncor

            interpNmat[i, 0, :, :] = np.concatenate((np.eye(6), np.zeros((6, 6))), axis=1)
            interpNmat[i,-1, :, :] = np.concatenate((np.zeros((6, 6)), np.eye(6)), axis=1)
            interpPos[i, 0] = self.scurve[i]*self.scurveLength
            interpPos[i,-1] = self.scurve[i+1]*self.scurveLength
            self.interpNmat_notwist = interpNmat 
            self.interpPos = interpPos
            self.interpNcor = interpNcor

    @staticmethod
    def Tmat_complimentary(xi, a):
        """ This function returns the section fore distribution matrix at xi wher a is the
        half length of thebeam and x1 in [-1, 1]"""
        T = np.eye(6)
        T[4, 0] = -a*xi
        T[3, 1] = a*xi
        return T

    @staticmethod
    def getqtilde_linearload(pvec1, pvec2, a, xi):
        # Calculates qtilde for linear load, pvec, half beam length a and location xi
        # Location on spline
        z = (xi+1)/2*2*a
        # Length of element
        L = a*2
        # Identity matrix
        Imat = np.eye(6)
        # T matrix
        T = np.zeros((6, 6))
        T[4, 0] = -1
        T[3, 1] = 1
        # load vector
        delta = (pvec2-pvec1)/L
        qq =-z*( Imat @ (1/2 * delta * z + pvec1 ) + T @ (1/ 6 * delta * z**2 + 1 / 2 * pvec1 * z))    
        return qq

    @staticmethod
    def getqtilde_segment(pvec1, pvec2, z1, z2, z):
        
        # Initilization
        qtilde = np.zeros(6)

        Tmat = np.zeros((6, 6))
        Tmat[4, 0] = -1
        Tmat[3, 1] = 1
        Imat = np.eye(6)

        # Explicit calculation of qtilde
        if z > z1: #internal forces are 0 between tip and load interval
            if z > z2:
                z0 = z2
            else:
                z0 = z

            DD = (pvec2-pvec1)/(z2-z1)
            C1 = 1/3 * Tmat @ DD * (z0**3-z1**3)
            C2 = 1/2 * (Tmat @ DD *(z0+z1)- Tmat@pvec1 + Imat@DD)*(z0**2-z1**2)
            C3 = z0*(-z1*Tmat@DD+Tmat@pvec1)*(z0-z1)
            C4 = Imat@(-DD*z1+pvec1)*(z0-z1)
            qtilde = (C1-C2-C3-C4)

            # Internal forces between segment interval and support 
            if z > z2:
                T = np.eye(6)
                T[4, 0] = -(z-z2)
                T[3, 1] = (z-z2)
                qtilde = T@qtilde
        return qtilde

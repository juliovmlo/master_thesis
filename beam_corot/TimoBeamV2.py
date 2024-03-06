import numpy as np
import os
import json
import time
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.linalg import eig
from scipy.linalg import solve
from .utils import subfunctions as subf


class TimoBeamV2:
    """ TimoBeam model with the nonhomogenneous equlibrium-based beam element. 

        Required input in mainInputFile
            - Input file names (properties, c2-pos, BC and loads)
            - "Analysistype" ("static" or "dynamic")
            - "int_type" ("gauss" or "trapz") 
            - "Nip" (number of integration points)
            - "Npmax" (maximum power of polynomials)
            - "int_type" ("gauss" or "trapz") 

        Descriptions of the code is seen in the Master Thesis: 
        " Modelling of Wind Turbine Blades by Beam Elements from Complementary Energy" 
        
        S163778 June 2022
    """

    def __init__(self, mainInputFile):
        tic = time.perf_counter()
        print('-----------')

        ###### INPUT ######
        # Define main directory
        self.maindir = os.path.dirname(mainInputFile) 
        # Read main input file
        self.readMainInputFile(mainInputFile)
        # Spanwise direction unit vector (do not change this!)
        self.v1 = np.array([.0,.0,1.0])
        # Number of segments for element property calculations
        self.numSegment = 100
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
        # Curved distance through 1/2chord location in cross section at c2 points
        self.createScurve()
        # Interpolation of centerline and properties between nodes and properties points
        self.createInterpolators()
        # Node location as elastic center at c2 points
        self.calculateNodeLocations()
        # Integration Points and weights
        self.calculalteIntPoints()
        # Caclulate cross sectional matrices at integration points
        self.calculateCrossSectionMatrices()

        ### ASSEMBLE MODEL ###
        # Element stiffness and mass matrix
        self.calculateElementMatrix()
        # Global stiffness and mass matrix
        self.calculateGeneralMatrix()

        ### STATIC LOADS ###
        # Initialize
        self.force = np.zeros((self.numNode*6))
        # Append static loads if defined in inputfile
        if 'static_load' in self.inputFileNames:
            self.calculateNodalLoads() 
        if 'static_load_distributed' in self.inputFileNames:
            self.calculateDistributedLoads()
        if 'static_load_distributed_segment' in self.inputFileNames:
            self.calculateSegmentLoads()

        ### BOUNDARY CONDITIONS ###
        # Apply boundary conditions
        self.applyBoundary()
        print('TimoBeamV2 Model Created')

        ##### Analysis #####
        # Static/ dynamic analysis if given in input file
        if self.analysistype == "dynamic":
            self.calcModeShapes()
            print('Dynamic analysis done')
        elif self.analysistype == "static":
            self.calcDeflection()
            print('Static analysis done')
        else:
            print("No analysis chosen")

        ### OUTPUT ###
        toc = time.perf_counter()  # Total runtime
        self.runTime = toc-tic


    def readMainInputFile(self,mainInputFile):
        '''This function reads the main input file'''
        try:
            with open(mainInputFile) as f:
                self.inputFileNames = json.load(f)
                self.analysistype = self.inputFileNames['Analysistype']
        except (IOError,FileNotFoundError) as error:
            print('********** Could not read the main input file ***********')
            print(error)
            raise
        except(json.decoder.JSONDecodeError) as error:
            print('********** Format error in main input file **********')
            print(error)
            raise
        if 'Npmax' in self.inputFileNames:
            npmax = self.inputFileNames['Npmax']
            self.ni = npmax+1
        else:
            self.ni = 6
            print('********** No maximum polynomial power defined **********')
            print('**********              "Npmax" = 5           **********')
        if 'Nip' in self.inputFileNames:
            self.nip = self.inputFileNames['Nip']
        else:
            self.nip = 6
            print('********** No number of integration points defined **********')
            print('**********              "Nip" = 6                **********')
            raise

        if 'mass_matrix' in self.inputFileNames:
            self.mass_matrix_type = self.inputFileNames['mass_matrix']
            print('Timo-2 uses only "Timo" type mass_matrix !!')
        else:
            self.mass_matrix_type = 'Timo'
            if self.analysistype == "dynamic":
                print('Timo-2 uses only "Timo" type mass_matrix !!')

        if 'int_type' in self.inputFileNames:
            self.intType = self.inputFileNames['int_type']
        else:
            print('**********                    No integration type defined                     **********')
            print('**********          "int_type" = "trapz" / "gauss" (default = "gauss")        **********')
            self.intType = 'gauss'

            
    def readInputs(self, inputfolder=''):
        """ This function reads the input files given in the main input file"""
        # Read structural properties
        # Note: the first 2 rows with text are skipped and the 3rd row is properties names
        #       The order of properties columns can not be changed!
        try:
            stPropertyInput = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['Properties']), skiprows=3)
            if stPropertyInput.shape[1]<29: # Assumes isotropic if less than 29 properties variables
                self.convertIso2Aniso(stPropertyInput)
                self.stProperty_iso = stPropertyInput.copy()
            else: # Assumes anisotropic
                self.stProperty = stPropertyInput.copy()
                with open(os.path.join(inputfolder,self.inputFileNames['Properties'])) as f:
                    lines=f.readlines()
                    self.propnames = [i.strip() for i in lines[1].split(' ') if not i.startswith('[') if not i.strip()==''][2::]

        except (IOError,FileNotFoundError) as error:
            print('********** Could not read the structural property file **********')
            print(error)
            raise

        # Read c2 definition
        # Note: The first 2 rows are skipped
        try:
            self.c2Input = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['c2_pos']), skiprows=2)
        except (IOError,FileNotFoundError) as error:
            print('********** Could not read the c2_pos file **********')
            print(error)
            raise

        # Read boundary Data
        # Note: The first 1 row for text is skipped
        try:
            self.boundaryData = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['Boundary']), skiprows=1).astype(int)
            if self.boundaryData.ndim==1:
                self.boundaryData = self.boundaryData[np.newaxis,:]
        except (IOError,FileNotFoundError,KeyError) as error:
            print('********** Could not read the boundary file **********')
            print(error)

        # Read static nodal load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load' in self.inputFileNames:
            try:
                self.loadData = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['static_load']), skiprows=1)
                if self.loadData .ndim==1:
                    self.loadData  = self.loadData[np.newaxis,:]
            except (IOError,FileNotFoundError,KeyError) as error:
                print('********** Could not read the force file **********')
                print(error)

        # Read static distributed load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load_distributed' in self.inputFileNames:
            try:
                self.loadDataDistributed = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['static_load_distributed']), skiprows=1)
                if self.loadDataDistributed .ndim==1:
                    self.loadDataDistributed  = self.loadDataDistributed[np.newaxis,:]
            except (IOError,FileNotFoundError,KeyError) as error:
                print('Could not read the distributed force file')
                print(error)
        
        # Read segment distributed load Data if it exist
        # Note: The first 1 row with text is skipped
        if 'static_load_distributed_segment' in self.inputFileNames:
            try:
                self.loadDataDistributedSegment = np.loadtxt(os.path.join(inputfolder,self.inputFileNames['static_load_distributed_segment']), skiprows=1)
                if self.loadDataDistributedSegment .ndim==1:
                    self.loadDataDistributedSegment  = self.loadDataDistributedSegment[np.newaxis,:]
            except (IOError,FileNotFoundError,KeyError) as error:
                print('Could not read the segment distributed force file')
                print(error)

    def convertIso2Aniso(self,stPropertyInput):
        ''' This function converts an HAWC2 isotropic input file to the corresponding anisotropic input'''
        # Properties names
        self.propnames = ['m', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch', 'x_e', 'y_e', 'K_11', 'K_12', 'K_13', 'K_14', 'K_15', 'K_16', 'K_22', 'K_23', 'K_24', 'K_25', 'K_26', 'K_33', 'K_34', 'K_35', 'K_36', 'K_44', 'K_45', 'K_46', 'K_55', 'K_56', 'K_66']
        self.stProperty = np.zeros((stPropertyInput.shape[0],30))
        # Standard parameters
        self.stProperty[:,0] = stPropertyInput[:,0] # r
        self.stProperty[:,1] = stPropertyInput[:,1] # m
        self.stProperty[:,2] = stPropertyInput[:,2] # x_cg
        self.stProperty[:,3] = stPropertyInput[:,3] # y_cg
        self.stProperty[:,4] = stPropertyInput[:,4] # ri_x
        self.stProperty[:,5] = stPropertyInput[:,5] # ri_y
        self.stProperty[:,6] = stPropertyInput[:,16] # pitch
        self.stProperty[:,7] = stPropertyInput[:,17] # x_e
        self.stProperty[:,8] = stPropertyInput[:,18] # y_e
        
        # Assemble cross sectional stiffness matrix
        for i in range(np.shape(self.stProperty)[0]):
            # Rotation matrix from c2 to elastic coordinates
            t_mat = subf.get_tsb(self.v1, self.v1, np.deg2rad(self.stProperty[i,6]))
            svec = t_mat.T @ np.array([stPropertyInput[i,6],stPropertyInput[i,7],0])
            evec = t_mat.T @ np.array([stPropertyInput[i,17],stPropertyInput[i,18],0])
    
            # Cross sectional stiffness matrix
            self.stProperty[i,9]  = stPropertyInput[i,13] * stPropertyInput[i,15] * stPropertyInput[i,9]
            self.stProperty[i,14] = - self.stProperty[i,9] *  (svec[1]-evec[1])
            self.stProperty[i,15] = stPropertyInput[i,14] * stPropertyInput[i,15] * stPropertyInput[i,9]
            self.stProperty[i,19] = self.stProperty[i,15] * (svec[0]-evec[0])
            self.stProperty[i,20] = stPropertyInput[i,8] * stPropertyInput[i,15]
            self.stProperty[i,24] = stPropertyInput[i,8] * stPropertyInput[i,10]
            self.stProperty[i,27] = stPropertyInput[i,8] * stPropertyInput[i,11]
            self.stProperty[i,29] = stPropertyInput[i,9] * stPropertyInput[i,12] + self.stProperty[i,19] *  (svec[0]-evec[0]) - self.stProperty[i,14]*(svec[1]-evec[1])

    def createScurve(self):
        ''' This function calculates the curved distance through 1/2chord location at the c2 points.
            The distance is defined as the linear distance between each point, normalized with the full beam length.'''
        # Initiate scurve break points
        s = np.zeros((self.numNode,))
        self.distanceBtwScurvePoints = np.linalg.norm(np.diff(self.c2Input[:,1:4], axis=0), axis=1)
        
        s[1::] = np.cumsum(self.distanceBtwScurvePoints)
        # Total scurve length
        self.scurveLength = s[-1]
        # Nondimensional scurve distance at c2 points/ nodes
        self.scurve = s/self.scurveLength

    def createInterpolators(self):
        ''' This function creates interpolation functions along the scurve for properties and c2-line position.
            c2Pos: akima interpolation (4th order polynomial) of scurve position between c2 input.
            c2Twist: akima interpolation of twist in c2 file between c2 input.
            structuralPropertyInterpolator: Linear interpolation of properties between properties input points. 

            Note: the r-distance in the properties file is adjusted to match the scurve length.
            '''
        # C2 Akima Interpolators
        c2Pos = self.c2Input[:,1:4]
        c2Twist = self.c2Input[:,4]

        # corresponding scurve break points
        s = self.scurve

        if self.numNode<3: # uses midpoint if only two nodes
            a = np.array([s[0],(s[0]+s[1])/2,s[1]])
            b = np.zeros((3,3))
            b[0,:] = c2Pos[0,:]
            b[-1,:] = c2Pos[-1,:]
            b[1,:] = (c2Pos[0,:] + c2Pos[1,:])/2
            self.scurvePosInterpolator  = Akima1DInterpolator(a, b)
            b = np.array([c2Twist[0],(c2Twist[0]+c2Twist[1])/2,c2Twist[1]])
            self.scurveTwistInterpolator  = Akima1DInterpolator(a, np.deg2rad(b))
        else:    
            self.scurvePosInterpolator  = Akima1DInterpolator(s, c2Pos)
            self.scurveTwistInterpolator  = Akima1DInterpolator(s, np.deg2rad(c2Twist))

        # Structural property interpolator
        st_prop_r,st_prop = self.stProperty[:,0], self.stProperty[:,1:]
        st_prop_r = st_prop_r /st_prop_r[-1] # r-distance adjusted to non-dim scurve length
        self.struturalPropertyInterploator = {key: interp1d(st_prop_r, st_prop[:, i]) for i, key in enumerate(self.propnames)}
                
    def calculateNodeLocations(self):
        '''This function calculates the location of the nodes in the elastic center in the c2-points'''
        # Distance to elastic center from each c2 coordinate
        ec_s = np.zeros((self.numNode, 3))
        ec_s[:,0], ec_s[:,1] = self.struturalPropertyInterploator['x_e'](self.scurve), self.struturalPropertyInterploator['y_e'](self.scurve)

        # Tangent of scurve at c2-points
        self.scurveTangent = self.scurvePosInterpolator.derivative(1)(self.scurve)

        # Initiate node positions
        nodeLocations = np.zeros((self.numNode,3))
        for i in range(self.numNode):
            nodeLocations[i,:] = self.c2Input[i,1:4] + subf.get_tsb(self.v1, self.scurveTangent[i,:], np.deg2rad(self.c2Input[i,4])) @ ec_s[i,:]
        self.nodeLocations = nodeLocations   

    def calculalteIntPoints(self):
        # Non dimensional points where the integrals are evaluated, including integral weights

        # Gauss integration
        if self.intType == 'gauss':
            xi, wi  = subf.gaussPts(self.nip, -1, 1)
        # Trapz integration
        elif self.intType == 'trapz':
            xi = np.linspace(-1,1,self.nip)
            wi = np.ones(np.shape(xi)[0])
            wi[1:-1] = 2
            wi = (xi[1]-xi[0])/2 *wi
        else:
            print('ERROR! Chose integraiton type as: trapz or gauss ')

        self.int_points = np.outer(np.ones(self.numElement),xi)
        self.int_weights = np.outer(np.ones(self.numElement),wi)

    def calculateCrossSectionMatrices(self):
        """ This function calculates the cross-sectional flexibility matrix 
            and mass matrix at the integration points. The matrices are 
            rotated to no twist (no angle between the x-axis and the XZ-plane). """

        # Initialize matrices
        S_mat = np.zeros((self.numElement,self.nip,6,6))
        M_sec = np.zeros((self.numElement,self.nip,6,6))

        # Loop over elements
        for i in range(self.numElement):
            # Element length in m
            diffNodes = self.nodeLocations[i+1,:] - self.nodeLocations[i, :]
            L = np.linalg.norm(diffNodes) 

            # Integration Points
            xi = self.int_points[i,:]
            
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
                S_mat[i,j,:,:] = Tmat_rot @ subf.getSectionalStiffness(props) @ Tmat_rot.T 

                # mass parameters
                el_p_local = np.zeros(5)
                el_p_local[0] = self.struturalPropertyInterploator['m'](scurvepos)

                # Rotation matrix from c2 to elastic coordinates
                t_mat_c2toelem = subf.get_tsb(self.v1, self.v1, el_p1)

                # mass center x- and y-locations from elastic center
                cgvec = t_mat_c2toelem.T @ np.array([self.struturalPropertyInterploator['x_cg'](scurvepos),self.struturalPropertyInterploator['y_cg'](scurvepos),0])
                evec = t_mat_c2toelem.T @ np.array([self.struturalPropertyInterploator['x_e'](scurvepos),self.struturalPropertyInterploator['y_e'](scurvepos),0])  
                el_p_local[1] =  cgvec[0] - evec[0]
                el_p_local[2] = cgvec[1] - evec[1]

                # other properties
                for jj, key in enumerate(self.propnames[3:5]):
                    el_p_local[jj+3] = self.struturalPropertyInterploator[key](scurvepos)
                M_sec[i,j,:,:] = Tmat_rot @subf.getSectionalMass(el_p_local)@ Tmat_rot.T 

        # Store matrices in class
        self.S_mat_notwist = S_mat
        self.M_sec_notwist = M_sec
            
    def calculateElementMatrix(self):
        ''' This function calculates the elastic stiffness matrix and the mass matrix of each element.
            Equation references to "Development of an anisotropic beam finite element for composite wind 
            turbine blades in multibody system" by Kim et al. (2013).'''
        # Number of integration points
        nip = self.nip

        # Polynomium order +1
        ni = self.ni

        # Initialization
        Kele = np.zeros((self.numElement,12,12))
        Mele = np.zeros_like(Kele)
        Nele_a = np.zeros((self.numElement,ni*6,12))

        # A matrices from Eq-11 in the paper
        A1 = np.zeros((6*ni,12))
        A2 = np.zeros((6*ni,6*ni-12))
        A1[:12,:] = np.eye(12)
        A2[12:,:] = np.eye(6*ni-12)

        # Loop over elements
        for i in range(self.numElement):
            # Element length
            diffNodes = self.nodeLocations[i+1,:] - self.nodeLocations[i, :]
            L = np.linalg.norm(diffNodes) 

            # 12x12 Rotation matrix (from element to global basis, without twist)
            t_mat = subf.get_tsb(self.v1, diffNodes, 0) 
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)

            # Integration Points and weights
            xi = self.int_points[i,:]
            wi = self.int_weights[i,:]

            # D matrix from eq-8 in the paper
            D_mat = np.zeros((6*ni,6*ni))
            M_int = np.zeros_like(D_mat)

            # Gauss/ Trapz integration 
            for j in range(nip):
                # Location on beam element in [m]
                z = (xi[j]+1)/2*L

                # Polynomial and strain-displacemenet matrix 
                B, N = subf.getStrainDisplacementMatrix(z, ni)

                # Cross sectional flexibility and mass matrix, including rotation from twist
                S_mat = self.S_mat_notwist[i,j,:,:]
                M_sec = self.M_sec_notwist[i,j,:,:]

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

            # Element stiffness and mass matrix in global coordinate system
            Kele[i,:,:] = Tmat @ (N_a.T @ D_mat @ N_a) @ Tmat.T
            Mele[i,:,:] = Tmat @ (N_a.T @ M_int @ N_a) @ Tmat.T

            # Shape function coefficient vector
            Nele_a[i,:,:] = N_a

        # Store matrices in class
        self.N_a_notwist = Nele_a # N_alpha (in element coordinate system)
        self.Kele = Kele  # Element stiffness matrices
        self.Mele = Mele  # Element mass matrices
        
    def calculateGeneralMatrix(self):
        ''' This function assembles the global stiffness and mass matrix'''
        # Initialize
        K_mat = np.zeros((self.numNode*6, self.numNode*6))
        if self.analysistype == 'dynamic':
            M_mat = np.zeros_like(K_mat)

        # Loop over elements
        for i in range(self.numElement):
            d1 = i * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            d2 = (i+1) * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            dof = np.hstack((d1, d2))

            K_mat[np.ix_(dof, dof)] += self.Kele[i,:,:]
            if self.analysistype == 'dynamic':
                M_mat[np.ix_(dof, dof)] += self.Mele[i,:,:]
        self.K_mat_full = K_mat 
        if self.analysistype == 'dynamic':
            self.M_mat_full = M_mat  

    def calculateNodalLoads(self):
        ''' This function defines nodal loads given in the main coordinate system'''
        # Loop over load inputs
        for i in range(np.shape(self.loadData)[0]):
                i_node = int(self.loadData[i,0])-1
                self.force[6*i_node:6*i_node+6] += self.loadData[i,1:]

    def calculateDistributedLoads(self):
        ''' This function calculates equivalent nodal loads for linear varying distributed loads
            along the element. The load directions are defined in the main coordinate system.'''

        # Number of integration points
        nip = self.nip
        ni = self.ni

        # Initialize load vector if no nodal loads are defined
        #if not hasattr(self, 'force'):
        #    self.force = np.zeros((self.numNode*6))


        # Loop over load inputs
        for i in range(np.shape(self.loadDataDistributed)[0]):
            # Element index
            i_elem = int(self.loadDataDistributed[i,0])-1
            # vector between element nodes
            diffNodes = self.nodeLocations[i_elem+1,:] - self.nodeLocations[i_elem, :]
            # Length of element
            L = np.linalg.norm(diffNodes)
            # Gauss Points
            xi = self.int_points[i_elem,:]
            wi = self.int_weights[i_elem,:]
            
            # Matrix to rotate from local to global axes
            t_mat = subf.get_tsb(self.v1, diffNodes, 0)
            Tmat = np.kron(np.eye(2, dtype=int), t_mat)
            # Load values at the end of the element in element coordinate system
            pvec1 = Tmat.T @ self.loadDataDistributed[i,1:7]
            pvec2 = Tmat.T @ self.loadDataDistributed[i,7:]   
        
            # Gauss integration 
            N_load_int = np.zeros(12)
            for j in range(nip):
                # Distributed load at gauss point
                P = pvec1+(pvec2-pvec1)*(xi[j]+1)/2 
                # shape function at gauss point
                B, N = subf.getStrainDisplacementMatrix((xi[j]+1)/2*L, ni)
                shapefunction_mat = N @ self.N_a_notwist[i_elem,:,:]
                # Append to equivalent load vector
                N_load_int +=  wi[j] *L/2* shapefunction_mat.T @ P

            # Rotate to global coordinates and append to load vector
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)
            self.force[i_elem*6:i_elem*6+12] +=  Tmat @ N_load_int 

    def calculateSegmentLoads(self):
        ''' This function calculates equivalent nodal loads for linear varying distributed loads
            along the element. The load directions are defined in the main coordinate system.'''

        # Number of integration points
        nip = self.nip
        ni = self.ni

        # Initialize load vector if no nodal loads are defined
        #if not hasattr(self, 'force'):
        #    self.force = np.zeros((self.numNode*6))


        # Loop over load inputs
        for i in range(np.shape(self.loadDataDistributedSegment)[0]):
            # Element index
            i_elem = int(self.loadDataDistributedSegment[i,0])-1
            # vector between element nodes
            diffNodes = self.nodeLocations[i_elem+1,:] - self.nodeLocations[i_elem, :]
            # Length of element
            L = np.linalg.norm(diffNodes)
            # Matrix to rotate from local to global axes
            t_mat = subf.get_tsb(self.v1, diffNodes, 0)
            # Load values at the end of the element in element coordinate system
            Tmat = np.kron(np.eye(2, dtype=int), t_mat)

            # Non_dim element location for start/ end point
            z1_nondim = self.loadDataDistributedSegment[i,1]
            z2_nondim = self.loadDataDistributedSegment[i,2]
            pvec1 = Tmat.T @ self.loadDataDistributedSegment[i,3:9]
            pvec2 = Tmat.T @ self.loadDataDistributedSegment[i,9:]  

            # Integration length of defined load (non-zero load vector)
            Lint = L * (z2_nondim-z1_nondim)

            # Gauss Points
            xi = self.int_points[i_elem,:]
            wi = self.int_weights[i_elem,:]
            
            # Gauss integration 
            N_load_int = np.zeros(12)
            for j in range(nip):
                # Position on element
                znondim = (xi[j]+1)/2*(z2_nondim-z1_nondim)+z1_nondim
                # Distributed load at gauss point
                P = pvec1+(pvec2-pvec1)*((xi[j]+1)/2)
                # shape function at gauss point
                B, N = subf.getStrainDisplacementMatrix(znondim*L, ni)
                shapefunction_mat = N @ self.N_a_notwist[i_elem,:,:]
                # Append to equivalent load vector
                N_load_int +=  wi[j] * Lint/2 * shapefunction_mat.T @ P

            # Rotate to global coordinates and append to load vector
            Tmat = np.kron(np.eye(4, dtype=int), t_mat)
            self.force[i_elem*6:i_elem*6+12] +=  Tmat @ N_load_int 


    def applyBoundary(self):
        ''' This function applies the boundary condition'''
        # Free and fixed dofs
        idx_delete = np.array([],dtype=int)
        for i_node, idx in zip(self.boundaryData[:,0], self.boundaryData[:,1::]):
            temp = np.array([6*(i_node-1)+j for j, k in enumerate(idx) if k==0])
            idx_delete = np.append(idx_delete,temp)  
        self.dofs = np.arange(self.numNode*6)
        self.dofs_fixed = idx_delete
        self.dofs_free = np.setdiff1d(self.dofs,self.dofs_fixed)
            
        # delete constrained dofs from K_mat_full
        self.K_mat = np.delete(self.K_mat_full, idx_delete, axis=1 )
        self.K_mat = np.delete(self.K_mat, idx_delete, axis=0 )

        # delete constrained dofs from M_mat_full
        if "M_mat_full"  in self.__dict__:
            self.M_mat = np.delete(self.M_mat_full, idx_delete, axis=1 )
            self.M_mat = np.delete(self.M_mat, idx_delete, axis=0 )

        # Delete contrained dofs from load vector
        if "force"  in self.__dict__:
            self.force_full = self.force
            self.force = np.delete(self.force, idx_delete )

    def calcModeShapes(self):
        ''' This function solves the eigenvale problem and calculates the mode shapes'''
        assert all(hasattr(self, attr) for attr in ["K_mat", "M_mat"]),'First call "applyBoundary" before calcModeShapes'
        w, eig_v = eig(self.K_mat, self.M_mat)
        sorted_idx = np.argsort(w.real)
        sorted_w = w.real[sorted_idx]
        sorted_w[sorted_w < 1e-5] = 0

        self.eigen_vec = eig_v[:,sorted_idx]
        self.freqs = np.sqrt(sorted_w)/(2*np.pi)
        for i_mode in range(self.freqs.shape[0]):
            eig_val = self.eigen_vec[:,i_mode]
            absval = abs(eig_val)
            idx = np.argmax(absval)
            self.eigen_vec[:,i_mode] = eig_val/absval.max() * np.sign(eig_val[idx])

        self.eigen_vec_full = np.zeros([self.numNode*6,np.shape(self.eigen_vec)[1]])
        self.eigen_vec_full[self.dofs_free,:] = self.eigen_vec    

    def calcDeflection(self):
        ''' This function calcualtes the static deflection'''
        self.defl = solve(self.K_mat,self.force)
        self.defl_full = np.zeros(self.numNode*6)
        self.defl_full[self.dofs_free] = self.defl

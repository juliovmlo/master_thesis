# -*- coding: utf-8 -*-
"""
Created on Jun 3 2022

@author: s163778
"""
#
import numpy as np
from scipy.linalg import lu
from . import rotation as rot  #import rotation functions
#
def get_tsb(v1,v2,angle):
        ''' Returns the rotation matrix which rotates a vector q_elem defined in the element coorinate system (x',y',v2), 
            to a vector q_global in a standard coordinate sytem ([1,0,0]',[0,1,0]',[0,0,1]'), thus
            q_global = tsb q_elem
        '''
        if v1[0] != 0 or v1[1]!= 0 or v1[2] != 1 :
            print('****** ERROR in get_tsb - v1 has to be a [0,0,1] vector ******')
            
        # normalize to unit vectors
        v2 = v2/np.linalg.norm(v2)
        v1 = v1/np.linalg.norm(v1)
        # Compute tsb without rotation around v2 (twist)
        cond1 = (abs(v2[0]) > 1e-6 and abs(v2[1]) > 1e-6)  
        cond2 = (abs(v2[0]) < 1e-6 and abs(v2[1]) < 1e-6 and v2[2]<0)

        if cond1 or cond2: # If v2 is not in the xz or yz plane or v2 = [0,0,-1]
            tsb_no_twist = np.zeros((3,3))
            tsb_no_twist[:,-1] = v2.copy()
            tsb_no_twist[0,1] = 0.0
            tsb_no_twist[1,1] = tsb_no_twist[2,2]/np.linalg.norm(tsb_no_twist[1:,-1])
            tsb_no_twist[2,1] = -tsb_no_twist[1,2]/np.linalg.norm(tsb_no_twist[1:,-1])
            tsb_no_twist[:,0] = np.cross(tsb_no_twist[:,1],tsb_no_twist[:,2])
        else: # if v2 is in the xz or yz plane
            tsb_no_twist = rot.rot_mat_2_vec(v1,v2)

        # Flip y-axis if negative (following HAWC2 formulation)
        if tsb_no_twist[1,1] < 0.0: 
            tsb_no_twist[1:,1] = -tsb_no_twist[1:,1]
            tsb_no_twist[:,0]  = np.cross(tsb_no_twist[:,1],tsb_no_twist[:,2])

        # Compute rotation around v2 (twist)
        tsb = tsb_no_twist.copy()
        tsb[:,:2] = np.matmul(rot.vec_mat(v2,angle),tsb_no_twist[:,:2])
        return tsb
def gaussPts(ngp, a, b):
        '''gaussPts(ngp,a,b) returns ngp sample points and weights for Gauss-Legendre quadrature. 
        Following python documentation page the values are tested up to degree 100.
        The sample points are translated from the standard [-1,1] range to a [a,b] range.
         '''
        xi, wi = np.polynomial.legendre.leggauss(ngp)
        xi = [(b-a)/2.*x+(a+b)/2. for x in xi]
        wi = [w*(b-a)/2. for w in wi]
        return xi, wi
#
def getSectionalStiffness(K_vec):
    ''' This function calculates the cross sectional stiffness matrix based on input from the properties file'''
    S = np.zeros((6,6))
    for i in range(6):
        for j in range(i,6):
            S[i,j] = K_vec[i*6+j-int(i*(i+1)/2)]
            S[j,i] = S[i,j]
    return S 
#
def getSectionalMass(M_vec):
    ''' This function calculates the cross sectional mass matrix based on input from the properties file'''
    mass = M_vec[0]
    x_cg, y_cg, rx, ry= tuple(M_vec[1:5])
    Ixx = mass * rx**2
    Iyy = mass * ry**2
    Ixy = 0 # assumed 0! 
    Ms11 = np.eye(3) * mass
    Ms12 = np.array([[0., 0., -mass*y_cg],
                     [0., 0., mass*x_cg],
                     [mass*y_cg, -mass*x_cg, 0.]])
    Ms22 = np.array([[Ixx, -Ixy, 0.],
                     [-Ixy, Iyy, 0.],
                     [0., 0., Ixx+Iyy]])
    Ms = np.block([[Ms11, Ms12],
                   [-Ms12, Ms22]])
    return Ms
#
def getStrainDisplacementMatrix(z, n):
        ''' This function returns the strains displacement matrix form Kim et al (2013).'''
        N = np.zeros((6, n*6))
        dN = np.zeros((6, n*6))
        N[:, :6] = np.eye(6)
        for i in np.arange(1, n):
            N[:, i*6:(i+1)*6] = np.eye(6)*z**i
            dN[:, i*6:(i+1)*6] = np.identity(6)*i*z**(i-1)
        B0 = np.zeros((6, 6))
        B1 = np.eye(6)
        B0[0, 4] = -1.0
        B0[1, 3] = 1.0
        B = B0 @ N + B1 @ dN
        return B, N

def inv(A):

    p,l,u = lu(A, permute_l = False)
    l = np.dot(p,l) 
    l_inv = np.linalg.inv(l)
    u_inv = np.linalg.inv(u)
    A_inv = np.dot(u_inv,l_inv)
    return A_inv
    #
if __name__ == "__main__":
    print('Shared subfunctions for beam models')

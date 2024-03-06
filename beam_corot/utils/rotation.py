# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:30:57 2017

@author: ozgo
"""
#
import numpy as np
#import math
#
def vec_mat(axis,angle):
    """
    Axis angle to rotation matrix
    input = axis, angle
    output = 3x3 rotation matrix
    """
    if float(angle) == 0.0 or np.linalg.norm(axis) < 10e-9:
        matrix = np.array([[1., 0., 0.], \
                  [0., 1., 0.],
                  [0., 0., 1.]])
    else:
        # equaiton 3.32 from S.Krenk       
        axis = np.asarray(axis)
        axis  = axis / np.linalg.norm(axis)
        matrix = np.cos(angle)*np.eye(3) + np.sin(angle)*skew_sym(axis) + (1-np.cos(angle)) * np.outer(axis,axis.T)
    return matrix
#
def mat_vec(R):
    """
    Rotation matrix to axis angle
    input = 3x3 rotation matrix 
    output = angle, vector
    """
    if (abs(R[0,0]+R[1,1]+R[2,2]-3.0)) > 1e-9:
        angle = np.arccos((R[0,0]+R[1,1]+R[2,2]-1.0)/2.0)
        x = (R[2,1]-R[1,2])/((R[2,1]-R[1,2])**2+(R[0,2]-R[2,0])**2+
                             (R[1,0]-R[0,1])**2)**0.5
        y = (R[0,2]-R[2,0])/((R[2,1]-R[1,2])**2+(R[0,2]-R[2,0])**2+
                             (R[1,0]-R[0,1])**2)**0.5
        z = (R[1,0]-R[0,1])/((R[2,1]-R[1,2])**2+(R[0,2]-R[2,0])**2+
                             (R[1,0]-R[0,1])**2)**0.5
    else:
        angle = 0
        x = 0
        y = 0
        z = 0
    vector = np.array([x, y, z])
    return angle,vector    
#
def vec_q(axis,angle):
    """
    Axis angle to quaternion
    input = axis, angle
    output = q [q0,q1,q2,q3]
    """
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    q0 = np.cos(angle/2)
    q1 = axis[0] * np.sin(angle/2)
    q2 = axis[1] * np.sin(angle/2)
    q3 = axis[2] * np.sin(angle/2)
    q = np.array([q0, q1, q2, q3])
    return q
#
def q_vec(q):
    """
    Quaternion to axis, angle
    input = q [q0,q1,q2,q3]
    output = axis, angle
    """
    if np.linalg.norm(q[1:]) < 1e-9:
        axis = np.array([0,0,0])
    else:
        axis = q[1:]/np.linalg.norm(q[1:])
    angle = 2 * np.arctan2((np.linalg.norm(q[1:])),q[0])
    return axis, angle
#
def q_mat(q):
    """
    Quaternion to rotation matrix
    input = q [q0,q1,q2,q3]
    output = 3x3 R
    """
    # equation 3.52 (S. Krenk)
    R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[1]*q[2]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
    return R
#
def v_tilde(u):
    '''
    It gives the skew symmetric matrix of a vector for matrix multipication
    input = u
    output = 3x3 matrix
    '''
    u_tilde = np.array([[0,-u[2],u[1]],
                        [u[2],0,-u[0]],
                        [-u[1],u[0],0]])
    return u_tilde
#
def vec_conseq(v2,v1):
    '''
    Returns the rotation vector which is result of two consecutive rotations v1 and v2 (rotation vectors)
    R_total = R2*R1
    input = v1,v2 (R3 = R2 x R1)
    output = v3 
    '''
    angle1 = np.linalg.norm(v1)
    angle2 = np.linalg.norm(v2)
    if angle1 < 1e-9:
        angle1 = 1.0
    if angle2 < 1e-9:
        angle2 = 1.0
    axis1 = v1 / angle1
    axis2 = v2 / angle2
    R1 = vec_mat(axis1,angle1)
    R2 = vec_mat(axis2,angle2)
    R3 = np.matmul(R2,R1)
    angle3,axis3 = mat_vec(R3)
    v3 = axis3 * angle3
    return v3
#
def rot_mat_2_vec(v1,v2):
    '''
    Returns the rotation matrix which rotates unit vector v1 to unit vector v2
    v2 = Rv1
    Parameters
    ----------
    v1 : unit vector  
    v2 : unit vector

    Returns
    -------
    Rotation matrix R

    '''
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    #
    c = np.dot(v1,v2)
    v = np.matmul(skew_sym(v1), v2)
    #
    R = np.eye(3) + skew_sym(v) + np.matmul(skew_sym(v),skew_sym(v)) * 1/(1+c)
    return R
#


def skew_sym(n):
    '''
    It gives the skew symmetric matrix of a vector for matrix multipication
    input = n
    output = 3x3 matrix
    '''
    n_tilde = np.array([[0,-n[2],n[1]],
                        [n[2],0,-n[0]],
                        [-n[1],n[0],0]])
    return n_tilde

def rotation_tensor(phi, n):
    # see page 51 Eq. (3.9)
    n_tilde = skew_sym(n)
    R = np.cos(phi) * np.eye(3) + np.sin(phi) * n_tilde + (1-np.cos(phi)) * np.dot(n,n.T)
    return R

def get_rotation_angle(R):
    # see page 51 Eq (3.11)
    return np.arccos(0.5*(np.trace(R) - 1))

def get_rotation_axis(R):
    # see page 52
    phi = get_rotation_angle(R)
    n_tilde = (R-R.T) / (2*np.sin(phi))
    return np.array([n_tilde[2,1], n_tilde[0,2], n_tilde[1,0]])

def rotation_tensor_btw_node_element_triples(e1,e0):
    # Eq (3.15) page 54
    s = e1 + e0
    # e_bar = s / np.linalg.norm(s)
    R1 = np.eye(3) + 2*np.dot(e1,e0.T) - 2*(np.dot(s,s.T))/(np.linalg.norm(s)**2)
    # Eq (3.16) page 55
    # R23 = np.eye(3) - 2 * np.dot(e_bar,e_bar.T)
    # return R1, R23
    return R1

def get_T_inv(phi,n):
    # Eq. (3.47) page 63
    T_inv = np.eye(3)
    if phi > 0:
        T_inv = np.outer(n,n)
        T_inv += (0.5*phi/np.tan(0.5*phi))*(np.eye(3) - np.outer(n,n))
        T_inv -= 0.5*phi*skew_sym(n)
    return T_inv

def get_quaternion(phi,n):
    # see eq (3.48) page 64
    r = np.zeros((4,))
    r[0] = np.cos(0.5*phi)
    r[1:] = np.sin(0.5*phi)*n
    return r

def rot_mat_q(r):
    '''
    
    Parameters
    ----------
    r : quaternions (r0,r1,r2,r3)

    Returns
    -------
    R : Rotation matrix

    '''
    # See eq (3.50) page 64
    r0 = r[0]
    r_vec = r[1:]
    R = (r0**2 - np.dot(r_vec.T,r_vec))*np.eye(3)
    R += 2 * r0 *skew_sym(r_vec)
    R += 2 * np.dot(r_vec,r_vec.T)
    return R

def quaternion_from_rotation_tensor(R):
    # See page 65
    r = np.zeros((4))
    S = 4*np.diag(np.array([1+R[0,0]+R[1,1]+R[2,2],
                       1+R[0,0]-R[1,1]-R[2,2],
                       1-R[0,0]+R[1,1]-R[2,2],
                       1-R[0,0]-R[1,1]+R[2,2]]))
    S[1,0] = S[0,1] = 4*(R[2,1] - R[1,2])
    S[2,0] = S[0,2] = 4*(R[0,2] - R[2,0])
    S[3,0] = S[0,3] = 4*(R[1,0] - R[0,1])
    S[2,1] = S[1,2] = 4*(R[1,0] + R[0,1])
    S[3,1] = S[1,3] = 4*(R[0,2] + R[2,0])
    S[3,2] = S[2,3] = 4*(R[1,2] + R[2,1])

    for j in range(4):
        r[j] = 0.5 * np.sqrt(S[j,j])
    return r

def add_two_quaternion(p,q):
    '''
    Parameters
    ----------
    p : follows rotation by q
    q : first rotation

    Returns
    -------
    r : total rotation quaternion

    '''
    # See eq (3.63) page 66
    r = np.zeros((4))
    p0,p_vec = p[0], p[1:]
    if abs(q[0]) > 1e-9 :
        q0,q_vector = q[0], q[1:]
    else:
        q0 = 1.0
        q_vector = np.array([0,0,0])
    r[0] = p0*q0 - np.dot(p_vec.T,q_vector)
    r[1:] = p0*q_vector + q0*p_vec + np.cross(p_vec,q_vector)
    return r

def get_rotationAxisAngle_from_quaternion(r):
    phi = 2*np.arccos(r[0])
    n = r[1:] / np.sin(0.5*phi)
    return phi, n

def get_mean_difference_rotation_of_two_rotation(pA,pB):
    s = np.zeros((4,)) # difference
    r = np.zeros_like(s) # mean
    pA0, pA_vec = pA[0], pA[1:]
    pB0, pB_vec = pB[0], pB[1:]
    # see eq (3.71) page 69
    s[0] = np.sqrt(0.25*(pA0+pB0)**2 + 0.25*np.linalg.norm(pA_vec+pB_vec)**2)    
    # see eq (3.70) page 69
    r[0] = 0.5*(pA0+pB0)/s[0]
    r[1:] = 0.5*(pA_vec+pB_vec) / s[0]
    # see eq (3.75) page 69
    s[1:] = pA0*pB_vec - pB0*pA_vec + np.cross(pA_vec,pB_vec)
    return r, s


if __name__ == "__main__":
    print('Conversions between rotation matrix, vectors and quaternions. Please see attributes in rotation module:')
#    print(dir(rot))

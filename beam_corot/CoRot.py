#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:28:54 2021

@author: ozgo
"""

import numpy as np
import os
import sys
import json
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.linalg import eig
#main_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
#utils_path = os.path.join(main_path,'utils')
#sys.path.insert(0, utils_path)
#import rotation as rot
from .utils import rotation as rot
from .utils import subfunctions as subf

class CoRot:
    def __init__(self, beam, numForceInc=10,max_iter=99,eps=1e-6):
        # Questions:
        # - What is numForceInc?
        #       It has something to do with the number of steps. It is like a incremental force.
        self.beam = beam
        self.max_iter = max_iter
        self.eps = eps * np.linalg.norm(self.beam.force)/numForceInc
        # self.eps = eps * np.min(self.beam.L_init) #self.beam.splineLength / numForceInc # for displacements
        self.numForceInc = numForceInc
        #
        self.force_inc = self.beam.force / self.numForceInc
        self.total_disp = np.zeros((self.beam.numNode*6))
        self.total_disp_q = np.zeros((self.beam.numNode*7))
        self.total_disp_q[3::7] = 1.0
        self.final_pos = np.zeros((self.beam.numNode*6))
        self.final_pos_q = np.zeros((self.beam.numNode*7))
        #
        self.diffq = np.zeros((self.beam.numElement,4))
        self.meanq = np.zeros((self.beam.numElement,4))
        self.nodePos = np.zeros((self.beam.numElement,2,3))
        self.elementDef = np.zeros((self.beam.numElement,6))
        self.n_vec = np.zeros((self.beam.numElement,3))
        self.elementDefForce = np.zeros((self.beam.numElement,6))
        self.nodeForces = np.zeros((self.beam.numElement,12))
        self.Ke = np.zeros((self.beam.numElement,12,12))
        self.Ke_kd = np.zeros((self.beam.numElement,12,12))
        self.Ke_kr = np.zeros((self.beam.numElement,12,12))
        self.Ke_geo = np.zeros((self.beam.numElement,12,12))
        self.Ke_gl = np.zeros((self.beam.numElement,12,12))
        #
        self.firstStep()
        self.iterAfterFirst()
        print("Final residual = %9.6f %%" %(np.linalg.norm(self.residual)/np.linalg.norm(self.force)*100))
        self.getFinalPositions()  # later
        
    def firstStep(self):
        self.K_mat = self.beam.K_mat.copy()
        self.R = self.beam.R_init.copy() # initial R
        self.L = self.beam.L_init.copy() # initial Length
        # 
        self.force = self.force_inc.copy()
        u = np.linalg.inv(self.K_mat) @ self.force
        u_all = self.getalldisp(u).copy()
        self.getTotalDisp(u_all) # self.total_disp , self.total_disp_q
        self.getElementNodeDef() # get self.nodeDef
        self.getNodePos() # get self.nodePos
        self.getElementq() # get self.diffq and self.meanq
        self.rotateElementAxis() # get self.R, self.n
        self.getElementDef() # get self.elementDef
        self.getElementNodalForces() # get self.elementDefForce and self.nodeForces
        self.getBeamNodeForces() # get self.beamForces
        self.residual = self.force - self.beamForces
        #
        self.getNewKe()
        self.rotateKe()
        self.calculateGeneralMatrix()
        self.applyBoundary()
        
        
    def iterAfterFirst(self):
        for i in range(self.numForceInc):
            self.force = self.force_inc * (i+1)
            i_iter = 0
            iter_res = self.force - self.beamForces
            # while np.linalg.norm(iter_res) > (self.eps) and i_iter < self.max_iter :
            while np.linalg.norm(iter_res) > (self.eps*(i+1)) and i_iter < self.max_iter :
                # u = self.beam.splineLength
            # while np.linalg.norm(u) > self.eps and i_iter < self.max_iter :
                f_new = self.force - self.beamForces
                self.getNewKe()
                self.rotateKe()
                self.calculateGeneralMatrix()
                self.applyBoundary()
                u = np.linalg.inv(self.K_mat) @ f_new
                u_all = self.getalldisp(u).copy()
                self.getTotalDisp(u_all) # self.total_disp , self.total_disp_q
                self.getElementNodeDef() # get self.nodeDef
                self.getNodePos() # get self.nodePos
                self.getElementq() # get self.diffq and self.meanq
                self.rotateElementAxis() # get self.R, self.n
                self.getElementDef() # get self.elementDef
                self.getElementNodalForces() # get self.elementDefForce and self.nodeForces
                self.getBeamNodeForces() # get self.beamForces
                self.residual = self.force - self.beamForces
                iter_res = self.residual.copy()
                # print('%15.8e' %np.linalg.norm(iter_res))
                # if np.linalg.norm(iter_res) > np.linalg.norm(f_new) and i_iter>0:
                #     print('residual becomes larger in %i th step'%i_iter)
                i_iter = i_iter + 1
            print("Step %i is converged after %i iteration" %(i+1,i_iter))
            # print("Delta u = %15.8e" %np.linalg.norm(u))
            # print("Residual = %15.8e" %np.linalg.norm(self.residual))
            # print("Residual = %9.6f %%" %(np.linalg.norm(self.residual)/np.linalg.norm(self.force)*100))
    
        
        
    def getElementq(self):
        for i in range(self.beam.numElement):
            self.diffq[i,0] = 0.5 * np.sqrt(
                (self.nodeDef[i,0,3] + self.nodeDef[i,1,3])**2 +
                np.linalg.norm(self.nodeDef[i,0,4:7] + self.nodeDef[i,1,4:7])**2)
            # if abs(self.diffq[i,0]) < 1e-9:
            #     self.diffq[i,0] = 1.0
            self.meanq[i,0] = 0.5 * (self.nodeDef[i,0,3] + 
                                   self.nodeDef[i,1,3]) / self.diffq[i,0]
            
            self.meanq[i,1:4] = 0.5 * (self.nodeDef[i,0,4:7] + 
                                       self.nodeDef[i,1,4:7]) / self.diffq[i,0]
            
            self.diffq[i,1:4] = 0.5 * (self.nodeDef[i,0,3] * self.nodeDef[i,1,4:7] - 
                                     self.nodeDef[i,1,3] * self.nodeDef[i,0,4:7] + 
                                     rot.skew_sym(self.nodeDef[i,0,4:7]) @ self.nodeDef[i,1,4:7]) /(
                                         self.diffq[i,0])
            
                                     
    def getNodePos(self):
        for i in range(self.beam.numElement):
            self.nodePos[i,0,:] = self.beam.nodeLocations[i,:] + self.nodeDef[i,0,:3]
            self.nodePos[i,1,:] = self.beam.nodeLocations[i+1,:] + self.nodeDef[i,1,:3]
            # element Length
            el_vec = self.nodePos[i,1,:] - self.nodePos[i,0,:]
            self.L[i] = np.linalg.norm(el_vec)
                
                                             
    def rotateElementAxis(self):
        for i in range(self.beam.numElement):
            R = rot.q_mat(self.meanq[i,0:4])
            n_new = R @ self.beam.R_init[i,:,:]
            #
            el_vec = self.nodePos[i,1,:] - self.nodePos[i,0,:]
            n = n_new[:,-1] + el_vec / np.linalg.norm(el_vec) # element unit vector
            self.n_vec[i,:] = n / np.linalg.norm(n)
            n_temp = n_new.copy()
            n_temp[:,-1] = -n_temp[:,-1] 
            self.R[i,:,:] = (np.eye(3) - 2*np.outer(self.n_vec[i,:],self.n_vec[i,:].T)) @ n_temp
            
        
    def  getElementDef(self):
        for i in range(self.beam.numElement):
            self.elementDef[i,3:] = 4 * self.R[i,:,:].T @ self.diffq[i,1:]
            phi_a = 4 * self.R[i,:,:].T @ (rot.skew_sym(self.R[i,:,-1]) @ self.n_vec[i,:])
            self.elementDef[i,:2] = phi_a[:2].copy()
            self.elementDef[i,2] = self.L[i] - self.beam.L_init[i]
                
                
    def getElementNodalForces(self):
        for i in range(self.beam.numElement):
            self.elementDefForce[i,:] = self.beam.Kd[i,:,:] @ self.elementDef[i,:]
            
            Re = np.kron(np.eye(4, dtype=int), self.R[i,:,:].copy())
            L = self.L[i].copy()

            s_mat = np.array([[0,2/L,0,0,0,0],[-2/L,0,0,0,0,0],[0,0,-1,0,0,0],
              [1,0,0,-1,0,0],[0,1,0,0,-1,0],[0,0,0,0,0,-1],
              [0,-2/L,0,0,0,0],[2/L,0,0,0,0,0],[0,0,1,0,0,0],
              [1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,0,0,0,1]])
            
            self.nodeForces[i,:] = Re @ s_mat @ self.elementDefForce[i,:]
            
    
    def getBeamNodeForces(self):
        f_all = np.zeros((self.beam.numElement+1)*6)
        f_all[:6] = self.nodeForces[0,:6].copy()
        f_all[-6:] = self.nodeForces[-1,-6:].copy()
        for i in range(self.beam.numElement-1):
             f_all[i*6+6:i*6+12] = self.nodeForces[i,6:12] + self.nodeForces[i+1,:6]
        self.beamAllForces = f_all.copy()

        idx_delete = np.array([],dtype=int)
        for i_node, idx in zip(self.beam.boundaryData[:,0], self.beam.boundaryData[:,1::]):
            temp = np.array([6*(i_node-1)+j for j, k in enumerate(idx) if k==0])
            idx_delete = np.append(idx_delete,temp)
        f = np.delete(f_all, idx_delete)
        self.beamForces = f.copy()    
                
        
    def getNewKe(self):
        for i in range(self.beam.numElement):
            L = self.L[i].copy()
            
            # Constitutive stiffness
            s_mat = np.array([[0,2/L,0,0,0,0],[-2/L,0,0,0,0,0],[0,0,-1,0,0,0],
              [1,0,0,-1,0,0],[0,1,0,0,-1,0],[0,0,0,0,0,-1],
              [0,-2/L,0,0,0,0],[2/L,0,0,0,0,0],[0,0,1,0,0,0],
              [1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,0,0,0,1]])
            
            self.Ke_kd[i,:,:] = s_mat @ self.beam.Kd[i,:,:] @ s_mat.T
           
            #
            node_f = s_mat @ self.elementDefForce[i,:]
            Qx = -2*self.elementDefForce[i,1]/L
            Qy = 2*self.elementDefForce[i,0]/L
            N = self.elementDefForce[i,2].copy()
            T = self.elementDefForce[i,5].copy()
            MxA = node_f[3].copy()
            MyA = node_f[4].copy()
            MxB = node_f[9].copy()
            MyB = node_f[10].copy()
            
            ## Kr Symmetric
            Kr = np.zeros((12,12))

            Kr[:3,:3] = 1/L * np.array([[N, 0, -Qx],
                                  [0, N, -Qy],
                                  [-Qx, -Qy, 0]])
            
            Kr[6:9,6:9] = Kr[:3,:3].copy()
            Kr[:3,6:9] = -Kr[:3,:3].copy()
            Kr[6:9,:3] = -Kr[:3,:3].copy()
            
            Kr[:3,3:6] = 1/L * np.array([[T, 0, MxA],
                                  [0, T, MyA],
                                  [0, 0, 0]])
            
            Kr[3:6,:3] = Kr[:3,3:6].copy().T
            Kr[6:9,3:6] = -Kr[:3,3:6].copy()
            Kr[3:6,6:9] = -Kr[:3,3:6].copy().T
            
            Kr[:3,9:12] = 1/L * np.array([[-T, 0, MxB],
                                  [0, -T, MyB],
                                  [0, 0, 0]])
            
            Kr[9:12,:3] = Kr[:3,9:12].copy().T
            Kr[6:9,9:12] = -Kr[:3,9:12].copy()
            Kr[9:12,6:9] = -Kr[:3,9:12].copy().T
            
            Kr[3:6,9:12] = 1/2 * np.array([[0, T, 0],
                                  [-T, 0, 0],
                                  [0, 0, 0]])
            Kr[9:12,3:6] = Kr[3:6,9:12].copy().T
            
            Kr[3:6,3:6] = 1/2 * np.array([[0, 0, -MyA],
                                  [0, 0, MxA],
                                  [-MyA, MxA, 0]])
            
            Kr[9:12,9:12] = 1/2 * np.array([[0, 0, -MyB],
                                  [0, 0, MxB],
                                  [-MyB, MxB, 0]])
                      
            self.Ke_kr[i,:,:] = Kr.copy()

            ## Geometric stiffness matrix
            Kg = np.zeros((12,12))
            
            Kg[:3,:3] = 1/L * np.array([[0.2*N, 0, 0],
                                  [0, 0.2*N, 0],
                                  [0, 0, 0]])
            
            Kg[6:9,6:9] = Kg[:3,:3].copy()
            Kg[:3,6:9] = -Kg[:3,:3].copy()
            Kg[6:9,:3] = -Kg[:3,:3].copy()
            
            
            Kg[:3,3:6] = np.array([[0, 0.1*N, 0],
                                  [-0.1*N, 0, 0],
                                  [0, 0, 0]])
            
            Kg[3:6,:3] = Kg[:3,3:6].copy().T
            Kg[6:9,3:6] = -Kg[:3,3:6].copy()
            Kg[3:6,6:9] = -Kg[:3,3:6].copy().T
            
            Kg[:3,9:12] = np.array([[0, 0.1*N, 0],
                                  [-0.1*N, 0, 0],
                                  [0, 0, 0]])
            
            Kg[9:12,:3] = Kg[:3,9:12].copy().T
            Kg[6:9,9:12] = -Kg[:3,9:12].copy()
            Kg[9:12,6:9] = -Kg[:3,9:12].copy().T
            
            Kg[3:6,9:12] = 1/6 * np.array([[-0.2*L*N, 0, L*Qx],
                                  [0, -0.2*L*N, L*Qy],
                                  [L*Qx, L*Qy, 0]])
            Kg[9:12,3:6] = Kg[3:6,9:12].copy().T
            
            Kg[3:6,3:6] = 1/6 * np.array([[0.8*L*N, 0, MyA+MyB],
                                  [0, 0.8*L*N, -MxA-MxB],
                                  [MyA+MyB, -MxA-MxB, 0]])
            
            Kg[9:12,9:12] = 1/6 * np.array([[0.8*L*N, 0, MyA+MyB],
                                  [0, 0.8*L*N, -MxA-MxB],
                                  [MyA+MyB, -MxA-MxB, 0]])
            
            self.Ke_geo[i,:,:] = Kg.copy()
            
            self.Ke[i,:,:] =  self.Ke_kd[i,:,:] +  self.Ke_kr[i,:,:] + self.Ke_geo[i,:,:]
            
    def rotateKe(self):
        for i in range(self.beam.numElement):
            for j in range(4):
                for k in range(4):
                    self.Ke_gl[i,j*3:(j+1)*3,k*3:(k+1)*3] = self.R[i,:,:] @ self.Ke[i,j*3:(j+1)*3,k*3:(k+1)*3] @ self.R[i,:,:].T
                    
    def calculateGeneralMatrix(self):
        K_mat = np.zeros((self.beam.numNode*6, self.beam.numNode*6))
        for i in range(self.beam.numElement):
            d1 = i * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            d2 = (i+1) * 6 * np.ones((6,), dtype=int) + np.arange(6, dtype='int')
            dof = np.hstack((d1, d2))
            K_mat[np.ix_(dof, dof)] += self.Ke_gl[i,:,:]
        self.K_mat_full = K_mat.copy()    


    def applyBoundary(self):
        idx_delete = np.array([],dtype=int)
        for i_node, idx in zip(self.beam.boundaryData[:,0], self.beam.boundaryData[:,1::]):
            temp = np.array([6*(i_node-1)+j for j, k in enumerate(idx) if k==0])
            idx_delete = np.append(idx_delete,temp)
        # delete constraint DOFs from K_mat_full & M_mat_full
        self.K_mat = np.delete(self.K_mat_full, idx_delete, axis=1 )
        self.K_mat = np.delete(self.K_mat, idx_delete, axis=0)        
            
    
    def getTotalDisp(self,u_all):
        for i in range(self.beam.numNode):
            self.total_disp[i*6:i*6+3] = self.total_disp[i*6:i*6+3] + u_all[i*6:i*6+3]
            self.total_disp[i*6+3:i*6+6] = rot.vec_conseq(u_all[i*6+3:i*6+6],self.total_disp[i*6+3:i*6+6])
            self.total_disp_q[i*7:i*7+3] = self.total_disp_q[i*7:i*7+3] + u_all[i*6:i*6+3]
            angle = np.linalg.norm(u_all[i*6+3:i*6+6])
            if angle == 0.0:
                axis = self.beam.v1.copy()
            else:
                axis = u_all[i*6+3:i*6+6]/angle
            p = rot.vec_q(axis,angle)
            self.total_disp_q[i*7+3:i*7+7] = rot.add_two_quaternion(p,self.total_disp_q[i*7+3:i*7+7])
    
    
    def getFinalPositions(self):
        for i in range(self.beam.numNode):
            self.final_pos[i*6:i*6+3] = self.beam.nodeLocations[i,:] + self.total_disp[i*6:i*6+3]
            self.final_pos_q[i*7:i*7+3] = self.beam.nodeLocations[i,:] + self.total_disp_q[i*7:i*7+3]
            R = subf.get_tsb(self.beam.v1, self.beam.scurveTangent[i,:], np.deg2rad(self.beam.c2Input[i,4]))
            q = rot.quaternion_from_rotation_tensor(R)
            axis, angle = rot.q_vec(q)
            n = axis*angle
            q_final = rot.add_two_quaternion(self.total_disp_q[i*7+3:i*7+7],q)
            self.final_pos_q[i*7+3:i*7+7] = q_final
            self.final_pos[i*6+3:i*6+6] = rot.vec_conseq(self.total_disp[i*6+3:i*6+6],n)
    
    
    def getElementNodeDef(self):
        u = self.total_disp_q.copy()
        u_el = np.zeros((self.beam.numElement,2,7))
        for i in range(0,self.beam.numElement):
            u_el[i,0,:] = u[i*7:i*7+7]
            u_el[i,1,:] = u[i*7+7:i*7+14]   
        self.nodeDef = u_el
        
    
    
    def getalldisp(self,u):
        idx_delete = np.array([],dtype=int)
        for i_node, idx in zip(self.beam.boundaryData[:,0], self.beam.boundaryData[:,1::]):
            temp = np.array([6*(i_node-1)+j for j, k in enumerate(idx) if k==0])
            idx_delete = np.append(idx_delete,temp)
        u_all = u.copy()
        for i,idx in enumerate(np.sort(idx_delete)):
            u_all = np.insert(u_all, idx , 0.0)
        return u_all

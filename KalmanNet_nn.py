"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from pyquaternion import Quaternion
import numpy as np
from utils.math import warp_to_pi

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self,config):
        super().__init__()
        self.dt, self.has_velo = config['basic']['LiDAR_interval'], config['basic']['has_velo']

        self.class_num = config['basic']['CLASS_NUM']

        self.seq_len_input, self.batch_size = 1,1

        self.h = {}

        self.mul_model = False
    
    def NNBuild(self, MD, SD):

        self.device = torch.device('cuda')

        # self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.MD = MD

        self.SD = SD    

        self.in_mult_KNet = 5
        # self.hidden_KNet = 10
        self.out_mult_KNet = 40

        # Number of neurons in the 1st hidden layer
        #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        self.InitKGainNet()

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self):

        # self.seq_len_input = 1 # KNet calculates time-step by time-step
        # self.batch_size = args.n_batch # Batch size

        self.prior_Q = torch.eye(self.SD).to(self.device)
        self.prior_Sigma = torch.zeros((self.SD, self.SD)).to(self.device)
        self.prior_S = torch.eye(self.MD).to(self.device)
        self.prior_R = torch.eye(self.MD).to(self.device)

        # GRU to track Q
        self.d_input_Q = self.SD * self.in_mult_KNet
        self.d_hidden_Q = self.SD ** 2
        self.GRU_Q = nn.ModuleDict()
        self.GRU_Q= nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.SD * self.in_mult_KNet
        self.d_hidden_Sigma = self.SD ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.MD ** 2 + self.MD * self.in_mult_KNet + self.MD ** 2
        self.d_hidden_S = self.MD ** 2
        self.GRU_S= nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # GRU to track R
        self.d_input_R = self.MD * self.in_mult_KNet
        self.d_hidden_R = self.MD ** 2
        self.GRU_R = nn.GRU(self.d_input_R, self.d_hidden_R).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.MD ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.MD * self.SD
        self.d_hidden_FC2 = self.d_input_FC2 * self.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.SD ** 2
        self.FC3 =nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.SD
        self.d_output_FC5 = self.SD * self.in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.SD
        self.d_output_FC6 = self.SD * self.in_mult_KNet
        self.FC6= nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = self.MD
        self.d_output_FC7 = self.MD * self.in_mult_KNet
        self.FC7= nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU()).to(self.device)

            
        self.d_input_FC8 = self.MD
        self.d_output_FC8 = self.MD * self.in_mult_KNet
        self.FC8 = nn.Sequential(
            nn.Linear(self.d_input_FC8, self.d_output_FC8),
            nn.ReLU()).to(self.device)
            

    def getInitState(self, det_infos: dict) -> torch.tensor:
        """from detection init tracklet
        Acceleration and yaw(turn) rate are both set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0
        """
        init_state = torch.zeros(self.SD,1)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set x, y, z, w, l, h, (v, if velo is valid)
        init_state[:6] = torch.tensor(det[:6]).unsqueeze(-1)
        if self.has_velo: init_state[6] = torch.hypot(torch.tensor(det[6]), torch.tensor(det[7]))
        
        # set yaw
        init_state[-2] = det_box.yaw  

        # self.y_previous = self.StateToMeasure(init_state)
        
        init_state.requires_grad_(True)

        return init_state  
        
    # def getInitState(self, det_infos: dict) -> torch.tensor:
    #     """from detection init tracklet
    #     Acceleration and yaw(turn) rate are both set to 0. when velociy
    #     on X/Y-Axis are available, the combined velocity is also set to 0
    #     """
    #     init_state = torch.zeros(self.SD, 1, requires_grad=True)  # 启用梯度追踪
    #     det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
    #     # set x, y, z, w, l, h, (v, if velo is valid)
    #     init_state[:6] = torch.tensor(det[:6], requires_grad=True).unsqueeze(-1)
    #     if self.has_velo: init_state[6] = torch.hypot(torch.tensor(det[6], requires_grad=True), torch.tensor(det[7], requires_grad=True))
        
    #     # set yaw
    #     init_state[-2] = torch.tensor(det_box.yaw, requires_grad=True)

    #     # self.y_previous = self.StateToMeasure(init_state)
        
    #     return init_state


    # @staticmethod
    def stateTransition(self, state: torch.tensor) -> torch.tensor:

        assert state.shape == (10,1)

        dt = self.dt

        # Assuming state is a PyTorch tensor
        # x, y, z, w, l, h, v, a, theta, omega = state.squeeze().tolist()
        x = state[0]
        y = state[1]
        z = state[2]
        w = state[3]
        l = state[4]
        h = state[5]
        v = state[6]
        a = state[7]
        theta = state[8]
        omega = state[9]
        yaw_sin, yaw_cos = torch.sin(theta), torch.cos(theta)
        next_v, next_ry = v + a * dt, theta + omega * dt

        # corner case (tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt**2 / 2
            predict_state = torch.cat([
                x + displacement * yaw_cos,
                y + displacement * yaw_sin,
                z, w, l, h,
                next_v, a,
                next_ry, omega
            ]).unsqueeze(-1)
        else: #积分
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = torch.sin(next_ry), torch.cos(next_ry)
            predict_state = torch.cat([x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
                             y + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
                             z, w, l, h,
                             next_v, a, 
                             next_ry, omega]).unsqueeze(-1)
            
        return predict_state

    def StateToMeasure(self, state: torch.tensor) -> torch.tensor:
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        # x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        x = state[0]
        y = state[1]
        z = state[2]
        w = state[3]
        l = state[4]
        h = state[5]
        v = state[6]
        # a = state[7]
        theta = state[8]
        # omega = state[9]
        if self.has_velo:
            state_info = torch.cat([x, y, z,
                          w, l, h,
                          v * torch.cos(theta),
                          v * torch.sin(theta),
                          theta]).unsqueeze(-1)
        else:
            state_info = torch.cat([x, y, z,
                          w, l, h,
                          theta]).unsqueeze(-1)
        
        return state_info

    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        # print (state.shape)
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state

    def getOutputInfo(self, state: torch.tensor) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        # state_np = self.warpStateYawToPi(state.detach().numpy())

        state_np = self.warpStateYawToPi(state)
        rotation = Quaternion(axis=(0, 0, 1), radians=state_np[-2, 0]).q
        list_state = state_np.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)
    
    ###############
    ### Forward ###
    ###############
    def forward(self, obs_diff,obs_innov_diff,fw_evol_diff,fw_update_diff,track_id,class_id):
        
        # print (obs_diff.shape)
        # exit()
        return self.step_KGain_est(obs_diff,obs_innov_diff,fw_evol_diff,fw_update_diff,track_id,class_id)

    def init_hidden_KNet(self,track_id:int):
        h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        h_R = self.prior_R.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

        
        self.h[track_id] = [h_S,h_Sigma,h_Q,h_R]

    #######################
    ### Kalman Net Step ###
    #######################
    # def KNet_step(self, obs_diff,obs_innov_diff,fw_evol_diff,fw_update_diff):
    #     # Compute Kalman Gain
    #     self.step_KGain_est(obs_diff,obs_innov_diff,fw_evol_diff,fw_update_diff)

        # self.y_previous =  self.m1y

    def h_reset(self):
        self.h = {}

    def step_KGain_est(self, obs_diff,obs_innov_diff,fw_evol_diff,fw_update_diff,track_id,class_id):   
        # both in size [batch_size, n]

        # obs_diff = y - self.y_previous
        # obs_innov_diff = y - m1y 

        # fw_evol_diff = m1x_posterior - m1x_posterior_previous
        # fw_update_diff = m1x_posterior - m1x_prior_previous

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff,track_id,class_id)
    
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.SD, self.MD))

        return self.KGain

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff,track_id,class_id):

        # def expand_dim(x):
        #     expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
        #     expanded[0, :, :] = x
        #     return expanded

        # obs_diff = expand_dim(obs_diff)
        # obs_innov_diff = expand_dim(obs_innov_diff)
        # fw_evol_diff = expand_dim(fw_evol_diff)
        # fw_update_diff = expand_dim(fw_update_diff)

        obs_diff = torch.transpose(obs_diff,0,1).unsqueeze(0)
        obs_innov_diff = torch.transpose(obs_innov_diff,0,1).unsqueeze(0)
        fw_evol_diff = torch.transpose(fw_evol_diff,0,1).unsqueeze(0)
        fw_update_diff = torch.transpose(fw_update_diff,0,1).unsqueeze(0)

        h_S = self.h[track_id][0]
        h_Sigma = self.h[track_id][1]
        h_Q = self.h[track_id][2]
        h_R = self.h[track_id][3]

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_update_diff    #fw_evol_diff
        if not self.mul_model:
            out_FC5 = self.FC5(in_FC5)
        else:
            out_FC5 = self.FC5[f'{class_id}'](in_FC5)

        # Q-GRU
        in_Q = out_FC5
        if not self.mul_model:
            out_Q, h_Q = self.GRU_Q(in_Q, h_Q)
        else:
            out_Q, h_Q = self.GRU_Q[f'{class_id}'](in_Q, h_Q)

        # FC 6
        in_FC6 = fw_evol_diff   #fw_update_diff
        if not self.mul_model:
            out_FC6 = self.FC6(in_FC6)
        else:
            out_FC6 = self.FC6[f'{class_id}'](in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        if not self.mul_model:
            out_Sigma, h_Sigma = self.GRU_Sigma(in_Sigma, h_Sigma)
        else:
            out_Sigma, h_Sigma = self.GRU_Sigma[f'{class_id}'](in_Sigma, h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        if not self.mul_model:
            out_FC1 = self.FC1(in_FC1)
        else:
            out_FC1 = self.FC1[f'{class_id}'](in_FC1)

        in_FC8 = obs_innov_diff
        if not self.mul_model:
            out_FC8 = self.FC8(in_FC8)
        else:
            out_FC8 = self.FC8[f'{class_id}'](in_FC8)

        in_R = out_FC8
        if not self.mul_model:
            out_R, h_R = self.GRU_R(in_R, h_R)
        else:
            out_R, h_R = self.GRU_R[f'{class_id}'](in_R, h_R)

        # FC 7
        # in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        in_FC7 = obs_diff
        if not self.mul_model:
            out_FC7 = self.FC7(in_FC7)
        else:
            out_FC7 = self.FC7[f'{class_id}'](in_FC7)

        # in_S_post = torch.cat((out_R, out_FC7), 2)

        # S-GRU
        in_S = torch.cat((out_R, out_FC7,out_FC1), 2)
        if not self.mul_model:
            out_S, h_S = self.GRU_S(in_S, h_S)
        else:
            out_S, h_S = self.GRU_S[f'{class_id}'](in_S, h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        if not self.mul_model:
            out_FC2 = self.FC2(in_FC2)
        else:
            out_FC2 = self.FC2[f'{class_id}'](in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        if not self.mul_model:
            out_FC3 = self.FC3(in_FC3)
        else:
            out_FC3 = self.FC3[f'{class_id}'](in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        if not self.mul_model:
            out_FC4 = self.FC4(in_FC4)
        else:
            out_FC4 = self.FC4[f'{class_id}'](in_FC4)

        # updating hidden state of the Sigma-GRU
        h_Sigma = out_FC4

        self.h[track_id][0] = h_S
        self.h[track_id][1] = h_Sigma
        self.h[track_id][2] = h_Q
        self.h[track_id][3] = h_R

        return out_FC2

'''
    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        
        # Set State Evolution Function
        self.f = f
        self.MD = m

        # Set Observation Function
        self.h = h
        self.SD = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        """
        input M1_0 (torch.tensor): 1st moment of x at time 0 [batch_size, m, 1]
        """
        self.T = T

        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y,2) - torch.squeeze(self.y_previous,2) 
        obs_innov_diff = torch.squeeze(y,2) - torch.squeeze(self.m1y,2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_posterior_previous,2)
        fw_update_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_prior_previous,2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.MD, self.SD))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y # [batch_size, n, 1]

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        #m1x_posterior_previous 

        # return
        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_update_diff    #fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_evol_diff   #fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2
    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

'''

# def stateTransition(state: torch.tensor) -> torch.tensor:

#     assert state.shape == (10,1)

#     dt = 0.5

#     # Assuming state is a PyTorch tensor
#     # x, y, z, w, l, h, v, a, theta, omega = state.squeeze().tolist()
#     x = state[0]
#     y = state[1]
#     z = state[2]
#     w = state[3]
#     l = state[4]
#     h = state[5]
#     v = state[6]
#     a = state[7]
#     theta = state[8]
#     omega = state[9]
#     yaw_sin, yaw_cos = torch.sin(theta), torch.cos(theta)
#     next_v, next_ry = v + a * dt, theta + omega * dt

#     print (x.shape)

#     # corner case (tiny yaw rate), prevent divide-by-zero overflow
#     if abs(omega) < 0.001:
#         displacement = v * dt + a * dt**2 / 2
#         predict_state = torch.cat([
#             x + displacement * yaw_cos,
#             y + displacement * yaw_sin,
#             z, w, l, h,
#             next_v, a,
#             next_ry, omega
#         ]).unsqueeze(-1)
#     else: #积分
#         ry_rate_inv_square = 1.0 / (omega * omega)
#         next_yaw_sin, next_yaw_cos = torch.sin(next_ry), torch.cos(next_ry)
#         predict_state = torch.cat([x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
#                             y + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
#                             z, w, l, h,
#                             next_v, a, 
#                             next_ry, omega]).unsqueeze(-1)
        
#     return predict_state

# if __name__ == "__main__":

#     tensor1 = torch.randn((10, 1), requires_grad=True)

#     tensor2 = stateTransition(tensor1)

#     tensor2.sum().backward()

#     print (tensor1.grad)
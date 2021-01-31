#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import copy

import numpy as np
np.random.seed(0)
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

import cupy.cudnn

# QNet
# ニューラルネットワークのクラス
class QNet(chainer.Chain):
    
    # __init__( n_in, n_units, n_out)
    #       n_in: 入力層サイズ
    #    n_units: 中間層サイズ
    #      n_out: 出力層サイズ
    """
    def __init__(self, n_in, n_units, n_out):
        super(QNet, self).__init__(
            l1=L.Linear(n_in, n_units),
            l20=L.Linear(n_units, n_units),
            l21=L.Linear(n_units, n_units),
            l22=L.Linear(n_units, n_units),
            l23=L.Linear(n_units, n_units),
            l24=L.Linear(n_units, n_units),
            l25=L.Linear(n_units, n_units),
            l26=L.Linear(n_units, n_units),
            l27=L.Linear(n_units, n_units),
            l28=L.Linear(n_units, n_units),   # 追加
            l29=L.Linear(n_units, n_units),  # 追加
            l3=L.Linear(n_units, n_out),
        )
    """
    # conv2dで構成
    def __init__(self, n_in, n_units, n_out):
        super(QNet, self).__init__(
            l1=L.Convolution2D(in_channels=2,out_channels=n_units, ksize = 3, pad = 1),   # 6*6
            l200=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1), 
            l201=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l202=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l203=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l204=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l205=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l206=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l207=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),
            l208=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),# 追加
            l209=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1), # 追加
            l210=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),  # 追加 
            l211=L.Convolution2D(in_channels=n_units,out_channels=n_units, ksize = 3, pad = 1),  #    追加 
            l212=L.Convolution2D(in_channels=n_units,out_channels=1, ksize = 1, nobias=True),  #    未使用
            l31=L.Bias(shape=(8*8)),# 未使用
            l32=L.Linear(n_units*64,256,nobias=True),  # 追加
            l33=L.Linear(256,n_out,nobias=True),  # 追加

            b01=L.BatchNormalization(size=n_units),
            b02=L.BatchNormalization(size=n_units),
            b03=L.BatchNormalization(size=n_units),
            b04=L.BatchNormalization(size=n_units),
            b05=L.BatchNormalization(size=n_units),
            b06=L.BatchNormalization(size=n_units),
            b07=L.BatchNormalization(size=n_units),
            b08=L.BatchNormalization(size=n_units),
            b09=L.BatchNormalization(size=n_units),
            b10=L.BatchNormalization(size=n_units),
            b11=L.BatchNormalization(size=n_units),
            b12=L.BatchNormalization(size=n_units),
        )
   

    #value(x)
    #       x: 入力層の値
    #ニューラルネットワークによる計算
    #Return: 出力層の結果
    def value(self, x , drop=0.4):
        h = F.relu(self.b01(self.l1(x)))
        h = F.relu(self.b02(self.l200(h)))
        h = F.relu(self.b03(self.l201(h)))
        h = F.relu(self.b04(self.l202(h)))
        h = F.relu(self.b05(self.l203(h)))
        h = F.relu(self.b06(self.l204(h)))
        h = F.relu(self.b07(self.l205(h)))
        h = F.relu(self.b08(self.l206(h)))
        h = F.relu(self.b09(self.l207(h)))
        h = F.relu(self.b10(self.l208(h))) #追加
        h = F.relu(self.b11(self.l209(h))) #追加 
        h = F.relu(self.b12(self.l210(h))) #追加 
        h = F.relu(self.l211(h)) #追加 
#        print('h210',h.size,h)
        #h= F.relu(self.l31(F.reshape(h,(len(h.data), (8*8)))))
        #print('l31',h.size,h)
        #print(h)
       # print(len(h.data))
        h= F.relu(self.l32(F.reshape(h,(len(h.data), -1)))) # 内部の２次元配列を１次元に変換
  #      print('l32',h.size,h)
        h = F.dropout(h,drop) 
        return self.l33(h)

    #__call__(s_data, a_data, y_data)
    #    s_data: 状態
    #    a_data: アクション
    #    y_data: 教師データ(次の行動の最大Q値)
    #学習用コールバック。
    #1. s_data を Forward 伝播する(Q,Q_Data)
    #2. t_data に Q_Dataをコピー
    #3．t_data の a_data[i] の値を y_data[i]の Q 値で置き換え教師データ作成(t)
    #4. Q,t の二乗誤差を算出
    #
    #Return: 二乗誤差計算結果
    def __call__(self, s_data, a_data, y_data):
        self.loss = None

        #print('s_data',s_data)
        #print('a_data',a_data)
        #print('y_data',y_data)

        s = chainer.Variable(self.xp.asarray(s_data))
        Q = self.value(s)
        
        Q_data = copy.deepcopy(Q.data)
        
        if type(Q_data).__module__ != np.__name__:
            Q_data = self.xp.asnumpy(Q_data)
        
        t_data = copy.deepcopy(Q_data)
        for i in range(len(y_data)):
            t_data[i, a_data[i]] = y_data[i]
        
        t = chainer.Variable(self.xp.asarray(t_data))
        self.loss = F.mean_squared_error(Q, t)
       # print('Loss:', self.loss.data)

       
        return self.loss

# 利用する
class MyAgent():
    pass

# エージェントクラス
class KmoriReversiAgent(Agent):
    
    #__init__(gpu, size,index)
    #    gpu: GPU 番号(0以上、CPU 使用の場合 -1)
    #    size: 正方形ボードの 1 辺の長さ(6 以上の偶数)
    #    index: モデルファイルの初期値（-1は無し）
    # エージェントの初期化、学習の内容を定義する
    def __init__(self, gpu, size, index):
        # サイズは 6 以上の偶数で。
        if size<6 and size%2 != 0 : print("size must be even number and 6 or above!") ; exit()
        # 盤の情報(オセロは8)
        self.n_rows = int(size)
        self.n_cols = self.n_rows
        
        # 学習のInputサイズ
        self.dim = self.n_rows * self.n_cols # ボードサイズ=出力層のサイズ
        self.bdim = self.dim * 2 # 学習用データのサイズ
        
        self.gpu = gpu
        
        # 学習を開始させるステップ数
        self.learn_start = 1 * 10**3
        
        # 保持するデータ数(changed)
        self.capacity = 1 * 10**4
        
        # eps = ランダムに○を決定する確率
        self.eps_start = 0.01
        self.eps_end = 0.01
        self.eps = self.eps_start
        self.eps_step = 0
        self.eps_max = 5* 10**3
        # 学習時にさかのぼるAction数
        self.n_frames = 1    # org 9
        
        # 一度の学習で使用するデータサイズ
        self.batch_size = 200
        
        self.replay_mem = []
        self.last_state = None
        self.last_action = None
        self.last_state2 = None
        self.last_action2 = None
        self.last_reward = None
        self.reward = None
        #self.state = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32)
        self.state = np.zeros((1, 2, self.n_rows, self.n_cols)).astype(np.float32)
        self.step_counter = 0
        
        #self.update_freq = 1 * 10**3
        self.update_freq = 1
        
        self.r_win = 1.0
        self.r_draw = 0.01
        self.r_lose = -1.0
        
        self.frozen = False
        
        self.win_or_draw = 0
        self.stop_learning = 500

        self.file_idx=index
        self.model_name='my_model_8x8_conv3'
        self.opt_name='my_opt_8x8_conv3'

        self.debug_flag=False

        ws_size = 2*1024*1024*1024
        cupy.cudnn.set_max_workspace_size(ws_size)



    #agent_init(task_spec_str)
    #    task_spec_str: RL_Glue から渡されるタスク情報
    # ゲーム情報の初期化
    def agent_init(self, task_spec_str):
        task_spec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_str)
        self.eps_step=0

        if not task_spec.valid:
            raise ValueError(
                'Task spec could not be parsed: {}'.format(task_spec_str))
        
        self.gamma = task_spec.getDiscountFactor() #　割引率
        #　DQN　作成
        #　Arg1: 入力層サイズ
        #　Arg2:　隠れ層ノード数
        #　Arg3：　出力層サイズ
        #self.Q = QNet(self.bdim*self.n_frames, self.bdim*self.n_frames, self.dim)
        self.Q = QNet(self.bdim*self.n_frames, self.dim*3 , self.dim)
        if self.file_idx>=0:
            serializers.load_hdf5(self.model_name+"_{0:05}.hdf5".format(self.file_idx), self.Q)
            self.step_counter= self.file_idx*1000 
            self.learn_start += self.step_counter


        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.Q.to_gpu()
        self.xp = np if self.gpu < 0 else cuda.cupy
        
        self.targetQ = copy.deepcopy(self.Q)
        
        self.optimizer = optimizers.RMSpropGraves(lr=0.01, alpha=0.95,momentum=0.0,eps=0.01)
       # self.optimizer = optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=0.001, eps=1e-08, eta=1.0)
        #self.optimizer = optimizers.SGD(lr=0.01)
        self.optimizer.setup(self.Q)

        if self.file_idx>=0:
            serializers.load_hdf5(self.opt_name+"_{0:05}.hdf5".format(self.file_idx), self.optimizer)


        self.file_idx=self.file_idx+1 

    #agent_start(observation)
    #    observation: ゲーム状態(ボード状態など)
    #environment.py の env_startの次に呼び出される。
    #1手目 Action を決定し、実行する。
    #実行した Action をエージェントへの情報として RL_Glue に渡す。
    def agent_start(self, observation):
        if self.debug_flag: print('agent start')

        # stepを1増やす
        self.step_counter += 1
        
        #開始時にstateをクリアしないとだめじゃない？
        #self.state = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32)
        self.state = np.zeros((1, 2, self.n_rows, self.n_cols)).astype(np.float32)

        #　kmori： 独自のobservationを使用して、状態をアップデート。
        # 一部サンプルに合わせ、残りは別の方法で作成した。
        self.update_state(observation)
        self.update_targetQ()

        if self.debug_flag: print('自分が打つ手を決定する。')
  
        # 自分が打つ手を決定する。
        int_action = self.select_int_action()
        action = Action()
        action.intArray = [int_action]
        if self.debug_flag: print('eps を更新する。')

        # eps を更新する。epsはランダムに○を打つ確率
        self.update_eps()
        
        # state = 盤の状態 と action = ○を打つ場所 を退避する
        self.last_state2 = copy.deepcopy(self.last_state) # ２手前の状態
        self.last_action2 = copy.deepcopy(self.last_action) # ２手前の行動
        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)
        
        return action
    
    #agent_step(reward, observation)
    #    reward: 報酬
    #    observation: ゲーム状態(ボード状態など)
    #エージェントの二手目以降、ゲームが終わるまで呼ばれる。
    #(Reversi の場合、報酬は常にゼロとなる)
    def agent_step(self, reward, observation):
        if self.debug_flag: print('agent step')

        # ステップを1増加
        self.step_counter += 1
        
        self.update_state(observation)
        self.update_targetQ()
        
        # 自分が打つ手を決定する。
        int_action = self.select_int_action() # 戻り値が -1 ならパス。
        action = Action()
        action.intArray = [int_action]
        self.reward = reward
        
        # epsを更新
        self.update_eps()
        
        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=False)
        
        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()
        
        self.last_state2 = copy.deepcopy(self.last_state) # ２手前の状態
        self.last_action2 = copy.deepcopy(self.last_action) # ２手前の行動
        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)
        
        # ○の位置をエージェントへ渡す
        return action
    
    #agent_end(reward)
    #    reward: 報酬
    # ゲームが終了した時点で呼ばれる
    def agent_end(self, reward):
        
        if self.debug_flag: print('agent end')

        self.eps_step +=1
       # 環境から受け取った報酬
        self.reward = reward
        
        if not self.frozen:
            if self.reward >= self.r_draw:
                self.win_or_draw += 1
            else:
                self.win_or_draw = 0
            
            if self.win_or_draw == self.stop_learning:
                self.frozen = True
                f = open('result.txt', 'a')
                f.writelines('Agent frozen\n')
                f.close()
        
        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=True)
        
        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()
    
    def agent_cleanup(self):
        # (今後実装)
        # RL_Cleanup により呼ばれるはず。
        # ここでモデルをセーブすればきっといい。
        print("model writting ... "+ str(self.file_idx))
        print("step:{0} /  eps:{1}".format(self.step_counter,self.eps))
        if self.gpu >= 0:
            self.Q.to_cpu()
        serializers.save_hdf5(self.model_name+"_{0:05}.hdf5".format(self.file_idx), self.Q , compression=4)
        serializers.save_hdf5(self.opt_name+"_{0:05}.hdf5".format(self.file_idx), self.optimizer , compression=4)
        self.file_idx=self.file_idx+1
        if self.gpu >= 0:
            self.Q.to_gpu()


    def agent_message(self, message):
        pass
    
    #update_state(observation=None)
    #    observation: ゲーム状態(ボード状態など)
    #ゲーム状態を state に格納する。
    def update_state(self, observation=None):
        # 学習用の状態保存。
        if observation is None:
            frame = np.zeros(1, 1, self.bdim*2).astype(np.float32)
        else:
            # observation の内容から、学習用データを作成。
            #print('ov',observation.intArray)
            """
            observation_binArray=[]
            pageSize=self.n_rows*self.n_cols
            # コマの位置
            for i in range(0,pageSize):store_transition
                observation_binArray.append(int(observation.intArray[i]))
                observation_binArray.append(int(observation.intArray[pageSize+i]))
            """
            # コマを置ける場所
            # observationから取り出すがstateには入れない
            self.nextStone=[]
            pageSize=self.n_rows*self.n_cols
            for i in range(0,pageSize):
                self.nextStone.append(int(observation.intArray[2*pageSize+i]))
            #    self.nextStone.append(int(observation.intArray[3*pageSize+i]))
            #print('next',self.nextStone)
            """
            for i in range(0,pageSize):
                observation_binArray.append(int(observation.intArray[2*pageSize+i]))
                observation_binArray.append(int(observation.intArray[3*pageSize+i]))
            """
            ##conv2D用
            observation_binArray=[]
            pageSize=self.n_rows*self.n_cols
            # コマの位置
            for i in range(0,pageSize*2):
                observation_binArray.append(int(observation.intArray[i]))
           # frame = (np.asarray(observation_binArray).astype(np.float32)
           #                                          .reshape(1, 1, -1))
            frame = (np.asarray(observation_binArray).astype(np.float32).reshape(1, 2, self.n_rows, self.n_cols))
        #self.state = np.vstack((self.state[:,1:], frame))
        self.state = frame

        #print(self.state)
    
    #update_eps()
    #ゲームの手数合計に基づき、ε-Greedy 法の ε を更新。
    def update_eps(self):
        #self.eps =self.eps_end # 固定にしてみる
        if self.frozen: self.eps=0
        
        #if self.step_counter>self.capacity: # ファイルから継続する場合はここからスタート
        self.eps=self.eps_start
        if self.step_counter > self.learn_start:
            self.eps = self.eps_start - self.eps_step*(self.eps_start - self.eps_end)/self.eps_max
        if self.eps<self.eps_end: self.eps=self.eps_end
        #print(self.eps)
       

    #update_targetQ()
    #update_freq 毎に、現時点の Q 値を、targetQ(Q 値推測用 Network) にコピー。
    def update_targetQ(self):
        if self.step_counter % self.update_freq == 0:
            self.targetQ = copy.deepcopy(self.Q)
    
    #select_int_action()
    #現在のボード状態から、DQN を用いて打つ手を決める。
    #コマを置く場所を決める。
    def select_int_action(self):
        #bits = self.state[0, -1]  #　ここでは stateの最後の要素つまり現時点の情報を得ている。
        #print ('state')
        #print(self.state)
        #print(bits)
        
        # ここでは、空きマスを取得している。
        # このオセロでは、コマを置ける場所は Observation に含まれるのでそれを使用する。
        free=[]
        #freeInBoard=bits[(self.n_rows*self.n_cols):]
        freeInBoard=self.nextStone #stateではなく、observationから取り出して使う
        for i in range(0, len(freeInBoard)) :
            if int(freeInBoard[i]) != 0 :
                free.append(i)
        #print(freeInBoard,free)
                
        # 置ける場所がなければここでパス
        if len(free)==0:
            # pass...
            return -1
        
        #　Q値を求める
        #print(self.state)
        s = chainer.Variable(self.xp.asarray(self.state))
        #print(s)
        Q = self.Q.value(s,drop=0.0)
        
        # Follow the epsilon greedy strategy
        if np.random.rand() < self.eps:
            int_action = free[np.random.randint(len(free))]
            #print("random challenge:",self.step_counter,int_action)
        else:
            #　先頭のQ値
            Qdata = Q.data[0]
            if type(Qdata).__module__ != np.__name__:
                Qdata = self.xp.asnumpy(Qdata)
            
            #　アクションを決定します。
            #     石を置けるマスの中から、Q値の最も高いものを行動として選択しています。
            for i in np.argsort(-Qdata):
                if i in free:
                    int_action = i
                    break
          #  print("select challenge",int_action)
        
        """
        # softmax select
        #　先頭のQ値
        Qdata = Q.data[0]
        if type(Qdata).__module__ != np.__name__:
            Qdata = self.xp.asnumpy(Qdata)

        Qdata_valid=[]
        Qdata_idx=[]
        for i in free:
            Qdata_valid.append(Qdata[i])
            Qdata_idx.append(i)
        print(("Q:",Qdata_idx,Qdata_valid))
        int_action=self.softmax_selection(10,Qdata_valid,Qdata_idx)                  
        """
        return int_action
    
    def softmax_selection(self,tau, values,idxs):
        """
            softmax 行動選択
        """
        emax = max([(v/tau) for v in values])
        sum_exp_values = sum([np.exp(v/tau-emax) for v in values])   # softmax選択の分母の計算
        p = [np.exp(v/tau-emax)/sum_exp_values for v in values]      # 確率分布の生成
        print("P=",p)

        action = np.random.choice(np.arange(len(values)), p=p)  # 確率分布pに従ってランダムで選択
       # print("action",action)
        return idxs[action]

    def store_transition(self, terminal=False):
        if len(self.replay_mem) < self.capacity:
            self.replay_mem.append(
                (self.last_state[0], self.last_action, self.reward,
                 self.state[0], terminal))
        else:
            #　self.replay_mem[1:]　で先頭つまり最古の要素を除く配列に、新しいものを追加。
            # これにより FIFO　でリストが回転する。
            self.replay_mem = (self.replay_mem[1:] +
                [(self.last_state[0], self.last_action, self.reward, 
                  self.state[0], terminal)])
    
    def replay_experience(self):
        #　replay_memory　から　バッチサイズ分の要素をランダムに取得する。
        indices = np.random.randint(0, len(self.replay_mem), self.batch_size)
        samples = np.asarray(self.replay_mem)[indices]
        #print(samples)
    
        s, a, r, s2, t = [], [], [], [], []
        
        for sample in samples:
            s.append(sample[0]) #last_state 打つまえの状態(frame)
            a.append(sample[1]) #last_action　打つ場所(int)
            r.append(sample[2]) #reward　報酬(int)
            s2.append(sample[3]) #state　結果(state)
            t.append(sample[4]) #terminal　終了？(int)
        
        s = np.asarray(s).astype(np.float32)
        a = np.asarray(a).astype(np.int32)
        r = np.asarray(r).astype(np.float32)
        s2 = np.asarray(s2).astype(np.float32)
        t = np.asarray(t).astype(np.float32)
        
        #Q 値推測用ネットワーク targetQ を取得し、s2の Q 値を求める
        #print(s2)
        s2 = chainer.Variable(self.xp.asarray(s2))
        Q = self.targetQ.value(s2)
        Q_data = Q.data
        
        #DDQN
        
        if type(Q_data).__module__ == np.__name__:
            max_Q_data = np.max(Q_data, axis=1)
        else:
            max_Q_data = np.max(self.xp.asnumpy(Q_data).astype(np.float32), axis=1)
        #print("DQN",max_Q_data)
        

        # DQN メインネットワークから最大値の位置だけ取得して、ターゲットネットワークから最大値として算出
        """
        mainQ = self.Q.value(s2)
        mainQ_data = mainQ.data
        if type(mainQ_data).__module__ == np.__name__:
            maxarg_Q_data = np.argmax(self.xp.asnumpy(mainQ_data).astype(np.float32), axis=1)
            maxarg_Q_data = np.expand_dims(maxarg_Q_data,axis=1) # 最大値のインデックスを元の配列と同型にする
            max_Q_data = np.take_along_axis(self.xp.asnumpy(Q_data).astype(np.float32),maxarg_Q_data,axis=1).reshape(-1) # メインネットワークから最大値を取り出す
            
        else:
            maxarg_Q_data = np.argmax(self.xp.asnumpy(mainQ_data).astype(np.float32), axis=1)
            maxarg_Q_data = np.expand_dims(maxarg_Q_data,axis=1)
            max_Q_data = np.take_along_axis(self.xp.asnumpy(Q_data).astype(np.float32),maxarg_Q_data,axis=1).reshape(-1)
        #print("DDQN",max_Q_data)
        """
        #targetQで推測した Q 値を使用して 教師データ t 作成
        t = np.sign(r) + (1 - t)*self.gamma*max_Q_data
        
        self.optimizer.update(self.Q, s, a, t)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--size', '-s', default=6, type=int,
                        help='Reversi board size')
    parser.add_argument('--index', '-i', default=-1, type=int,
                        help='model file start index')
    args = parser.parse_args()
    
    AgentLoader.loadAgent(KmoriReversiAgent(args.gpu,args.size,args.index))

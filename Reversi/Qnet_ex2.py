import numpy as np
np.random.seed(0)
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers

class MyQNet(chainer.Chain):
    
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
        super(MyQNet, self).__init__(
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

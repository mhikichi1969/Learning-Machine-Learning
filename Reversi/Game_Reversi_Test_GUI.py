#
# -*- coding: utf-8 -*- 
#
# Game_Reversi.py を人間がテストするためのスクリプト
#
import sys
import Game_Reversi as game
# Tkinterモジュールのインポート
import tkinter

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers

#from Qnet_ex1 import MyQNet as QNet
#from Qnet_ex2 import MyQNet as QNet

class MyReversi(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self.canvas = tkinter.Canvas(root, bg="white", height=300, width=300)

        self.offsx=30
        self.offsy=30
        self.size = 30
        #elf.canvas.create_polygon(250, 10, 220, 100, 150, 100,fill="green")
        #self.canvas.create_line(10, 200, 150, 150, fill='red')
        #self.canvas.create_oval(100, 100, 150, 150)
        self.btn_pass = tkinter.Button(root, text='PASS',command=self.pass_click) 
        self.btn_end = tkinter.Button(root, text='終了',command=self.end_click) 
        self.btn_restart = tkinter.Button(root, text='リスタート',command=self.restart_click) 
        self.btn_pass.place(x=40, y=320) 
        self.btn_end.place(x=140, y=320) 
        self.btn_restart.place(x=240, y=320) 

        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.lclick)


        self.g=game.Game_Reversi(8,8)
        self.initBord()
        self.update(self.g.g_board)

        #self.model_name = 'my_model_8x8_conv3_00060.hdf5'
        #self.Q = QNet(128, 128 , 64)
        #serializers.load_hdf5(self.model_name, self.Q)

        ## 相手から
        self.g.turn=-1
        board, reward, done = self.g.step(None,1.0,True)
        self.update(board)

    def initBord(self):
        tag_idx=0
        for j in range(0,8):
            for i in range(0,8):
                self.canvas.create_rectangle(self.offsx+i*self.size, self.offsy+j*self.size, 
                                    self.offsx+i*self.size+self.size, self.offsy+j*self.size+self.size, 
                                    fill = 'green', tag=str(tag_idx))
                tag_idx=tag_idx+1



    def update(self,b):
        stone=''
        for j in range(0,8):
            for i in range(0,8):
                if b[j,i]==1: stone='black'
                if b[j,i]==-1: stone='white'
                if b[j,i]!=0:
                    cx1=self.offsx+i*self.size+2
                    cy1=self.offsy+j*self.size+2
                    cx2=cx1+self.size-4
                    cy2=cy1+self.size-4
                    self.canvas.create_oval(cx1,cy1,cx2,cy2, 
                        fill = stone)


    def lclick(self,event):
        x,y = self.canvas.canvasx(event.x),self.canvas.canvasy(event.y)

        l = [self.canvas.itemcget(obj, 'tags') for obj in self.canvas.find_overlapping(x,y,x,y)]
        print(l) # ['en_1 current']など
        num=-1
        if len(l)>0:
            num_str = l[0].split()
            num = int(num_str[0])
            print(num)
            row=num//8
            col=num%8
            board, reward, done = self.g.step((row,col),tau=0.0001)
            self.update(board)

    def pass_click(self):
            board, reward, done = self.g.step((-1,-1),tau=0.0001)
            self.update(board)
    def end_click(self):
        pass

    def restart_click(self):
        self.initBord()
        self.g=game.Game_Reversi(6,6)
        self.update(self.g.g_board)
        
    
# ウィンドウ（フレーム）の作成
root = tkinter.Tk()

# ウィンドウの名前を設定
root.title("MyReversi")

# ウィンドウの大きさを設定
root.geometry("400x400")
# イベントループ（TK上のイベントを捕捉し、適切な処理を呼び出すイベントディスパッチャ）
app = MyReversi(master=root)
app.mainloop()


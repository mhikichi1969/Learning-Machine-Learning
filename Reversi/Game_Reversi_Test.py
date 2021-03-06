#
# -*- coding: utf-8 -*- 
#
# Game_Reversi.py を人間がテストするためのスクリプト
#
import sys
import Game_Reversi as game

g=game.Game_Reversi(6,6)
turn=g.turn
bFirst=True

while True:
    if bFirst : print(g.g_board) ; bFirst=False
    try:
        print("Input coordination in N,N format(like'4,3') for "+str(turn))
        print("To Pass, enter '-1,-1'")
        if sys.version_info.major == 2:
            pos=tuple(map(int,input()))
        else: #should be 3.x
            pos=tuple(map(int,input().split(',')))
    except ValueError:
        print("Invalid...")
        continue
    
    board, reward, done = g.step(pos)
    
    if done:
        print("!!!!!!!!!!!!!Exitng!!!!!!!!!!!!!")
        break

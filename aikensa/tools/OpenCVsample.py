import sys
import cv2
import numpy as np
#TIS.pyを別フォルダに格納吸う場合は下記のように記載
#sys.path.append("../python-common")

import TIS

# このサンプルはカメラからのイメージ取得とOpenCVで画像処理する一連の処理する方法を記載しています。
# 必要なパッケージ:
# pyhton-opencv
# pyhton-gst-1.0
# tiscamera

Tis = TIS.TIS()

#　DMK 33UX264 Serial: 16710581 、 解像度640x480＠30 fpsで表示する
#　selectDeviceを使ってデバイスをOpenするためコメントアウトしています
#　Tis.openDevice("16710581", 640, 480, "30/1", TIS.SinkFormats.BGRA,True)

###GigEの場合 カメラプロパティ設定
#Tis.List_Properties()で出力されたプロパティ値を引数に渡してください。
# errorが出る場合はコメントアウトしてください。

#selectDeviceを利用することでデバイスの選択、フォーマット、フレームレートをコマンドラインで指定できる
if not Tis.selectDevice():
    quit(0)


#tcam-ctrl -p <serial>で出力されたプロパティ値を引数に渡してください。
# errorが出る場合はコメントアウトしてください。
Tis.Set_Property("GainAuto","Off")# ゲインを固定化するために自動ゲインをオフにする
Tis.Set_Property("Gain", 9.6)
Tis.Set_Property("ExposureAuto", "Off")# 露光時間を固定化するために自動露光時間をオフにする
Tis.Set_Property("ExposureTime", 2977)
Tis.Set_Property("BalanceWhiteAuto", "Off")# ホワイトバランスを固定化するために自動ホワイトバランス調整をオフにする
Tis.Set_Property("BalanceWhiteRed", 1.0)
Tis.Set_Property("BalanceWhiteBlue", 1.0)
Tis.Set_Property("BalanceWhiteGreen", 1.0)
Tis.Set_Property("TriggerMode","Off")

#tcam-ctrl -p <serial>で出力されたプロパティ値を引数に渡してください。
# errorが出る場合はコメントアウトしてください。

print("Gain Auto : %s " % Tis.Get_Property("GainAuto"))
print("Gain : %d" % Tis.Get_Property("Gain"))
print("Exposure Auto : %s " % Tis.Get_Property("ExposureAuto"))
print("Exposure Time (us) : %d" % Tis.Get_Property("ExposureTime"))
print("Whitebalance Auto : %s " % Tis.Get_Property("BalanceWhiteAuto"))
print("Whitebalance Red : " ,Tis.Get_Property("BalanceWhiteRed"))
print("Whitebalance Blue :  " ,Tis.Get_Property("BalanceWhiteBlue"))
print("Whitebalance Green : " ,Tis.Get_Property("BalanceWhiteGreen"))


#Gstreamerでの映像伝送
Tis.Start_pipeline()  

print('Press Esc to stop')
lastkey = 0

cv2.namedWindow('Window') 

# OpenCVのerode関数で使う引数
kernel = np.ones((5, 5), np.uint8)  

#カウンター初期化
imagecounter=0

while lastkey != 27:
    #Timeout設定時間（１秒）以内にPCに画像を取得できた場合次の処理へ進む
    if Tis.Snap_image(1) is True: 
        # 画像取得
        image = Tis.Get_image() 
        # OpenCV で画像処理
        image = cv2.erode(image, kernel, iterations=5) 
        #画像処理後の画像を表示
        cv2.imshow('Window', image) 

        #Jpegファイルの名称にインデックス番号を付与するためにフレームの数をカウント
        imagecounter += 1
        filename = "./image{:04}.jpg".format(imagecounter)
        #Jpeg画像を保存（コメントアウトしています）
        #cv2.imwrite(filename, image)
    lastkey = cv2.waitKey(10)

# Gstreamerのパイプラインを停止
Tis.Stop_pipeline()
cv2.destroyAllWindows()
print('Program ends')

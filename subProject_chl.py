#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
칼만필터 이해해보기

  img = cv2.imread('/home/chl/Pictures/1.png')

  ball_point = [(100,100),(150,100),(200,100),
                (250,100),(300,100),(350,100),
                (400,100),(450,100),(500,100)]

  for pt in ball_point:
    cv2.circle(img,pt,15,(0,0,255),-1)

    predicted = kf.predict(pt[0],pt[1])

    cv2.circle(img,(predicted[0],predicted[1]), 15, (255,0,0), 4)
'''

class Kalman:
  kf = cv2.KalmanFilter(4,2)
  kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
  kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
  kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * (1e-4) #* (1e-5) # process noise covariance matrix (Q)
  '''
  파라미터에 따라 곡선 예측 또는 직선 예측을 잘 하는 값이 있는데, 반비례관계인가..?
  '''
  kf.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) # measurement noise covariance matrix (R). [1,0],[0,1] is the default value anyway…
  kf.statePost = np.array([[66],[514]],np.float32) # corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
  # 66, 514

  def predict(self, coordX, coordY):
    measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
    self.kf.correct(measured) # Updates the predicted state from the measurement.
    predicted = self.kf.predict()
    x,y = int(predicted[0]), int(predicted[1])
    return x, y


def onChange(arg):pass
cv2.namedWindow("trackbar")
cv2.createTrackbar('value','trackbar', 56, 255, onChange) # best : 56
cv2.createTrackbar('fixed','trackbar', 70, 255, onChange) # best : 70



def main():

  cap = cv2.VideoCapture("/home/chl/Videos/subProject.avi")

  if (not cap.isOpened()):
    print("Open Falied")
    return -1

  col = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  row = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  center = col // 2

  ## 마스크 제작
  mask = np.zeros((480,640), dtype=np.uint8)
  pts = np.array([[0,300],[640,300],[640,440],[0,440]])
  cv2.fillPoly(mask, [pts], [255,255,255])

  ## 칼만필터 선언
  kf = Kalman()

  ## lpos, rpos 저장 공간
  line_tmp = [['index'],['lpos'],['rpos']]
  cnt = 0
  index = 0

  while True:
    ret, frame = cap.read()

    if not ret: break

    ## hls에서 중요한 파라미터인 명도와 threshold value 값을 조정하기 위한 트랙바 설정
    fixed = cv2.getTrackbarPos('fixed', 'trackbar')
    value = cv2.getTrackbarPos('value', 'trackbar')

    ## 횟수 세기
    cnt += 1


    ############################# 영상처리 #############################
    ##################################################################

    ## 밝기 평균 고정
    m = cv2.mean(frame, mask)
    scalar = (-int(m[0])+fixed, -int(m[1])+fixed, -int(m[2])+fixed, 0)
    dst = cv2.add(frame, scalar, mask)

    ## 블러
    blur = cv2.GaussianBlur(dst, (0,0), 3.5)
    blur = cv2.GaussianBlur(blur, (0,0), 3.5)
    blur = cv2.GaussianBlur(blur, (0,0), 3.5)
    '''
    가우시안 블러를 한번만 사용하니 잡음이 어느 정도 검출되어서 3번을 주기로 함
    '''

    ## hls 변환
    _, hls, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    tmp = cv2.bitwise_and(hls, mask)

    ## 이진화
    #_, lane = cv2.threshold(tmp, 0, 255, cv2.THRESH_OTSU |cv2.THRESH_BINARY)

    _, lane = cv2.threshold(tmp, value, 255, cv2.THRESH_BINARY)
    '''
    오츠를 사용하려 했으나, value값을 변경해도 큰 변화가 없어서 바이너리만 사용하기로 판단
    '''


    ############################# 차선검출 #############################
    ##################################################################

    ## lpos, rpos 찾기. 안쪽에서 바깥쪽
    left, right = -1, -1
    lpos, rpos = 0, col

    for l in range(center-65, -1, -1):
      if lane[400, l] == 0:
        lpos = l
        left = 1
        break

    for r in range(center+65, col):
      if lane[400, r] == 0:
        rpos = r
        right = 1
        break
    '''
    블러로 인하여 바이너리 경계가 넓어져도 라이다와 겹치지 않으면서도,
    라이다와 차선이 겹치는 경우 최대한으로 차선을 검출할 수 있도록 하는 경계를 center±65로 잡음
    '''

    if lpos > 5:
      lpos -= 5
    if rpos < 635:
      rpos += 5

    if left:
      cv2.rectangle(frame, (lpos - 10, 390),(lpos + 10, 410), (0, 255, 0), 3)

    if right:
      cv2.rectangle(frame, (rpos - 10, 390),(rpos + 10, 410), (0, 255, 0), 3)


    ############################# 차선저장 #############################
    ##################################################################

    if cnt == 30:
      cnt = 0
      line_tmp[0].append(index)
      line_tmp[1].append(lpos)
      line_tmp[2].append(rpos)
      print(f'index: {index}, lpos:{lpos}, rpos:{rpos}')
      print("-------------------------")
      index+=1

    ############################# 자매품 #############################
    ##################################################################

    # 칼만필터: 차선이 없는 경우, 이전 프레임에서 예측한 차선을 현재의 차선으로 선택
    # 라이다가 차선을 가리는 구간은 잘 추정하는데, 곡선으로 인하여 차선이 없어지는 경우는 값 변동이 심하다
    if left == -1:
      lpos = predicted[0]

    if right == -1:
      rpos = predicted[1]

    predicted = kf.predict(lpos,rpos)

    cv2.rectangle(frame, (lpos - 5, 395),(lpos + 5, 405), (0, 0, 255), 3)
    cv2.rectangle(frame, (rpos - 5, 395),(rpos + 5, 405), (0, 0, 255), 3)
    '''
    1. 칼만 필터를 좀 더 정교하게 다룰 수 있었으면 좋았을텐데 아쉽다.
       이번 프로젝트에서는 어떤 느낌인지만 파악하고, 이론적인 공부를 통하여 파라미터 값을 조정해봐야겠다.
    2. 칼만 필터를 차선 뿐만 아니라 조향각에 대해서도 적용할 수 있을 것 같다.
    '''


    ############################# 영상재생 #############################
    ##################################################################

    cv2.imshow('lane',lane)
    cv2.imshow("img",frame)
    if (cv2.waitKey(0) == 27):
      break

  line = pd.DataFrame(line_tmp)
  line = line.transpose()
  line.to_csv('/home/chl/Documents/line.csv', header=False, index=False)

  ############################# 그래프 작성 #############################
  ####################################################################
  dp1 = pd.read_csv("/home/chl/Documents/a.csv")
  dp2 = pd.read_csv("/home/chl/Documents/line.csv")

  plt.figure(figsize=(100,12))
  plt.plot(dp1['index'], dp1['new lposl'], label="lposl_line")
  plt.plot(dp1['index'], dp1['new lposr'], label="lposr_line")
  plt.scatter(dp2['index'], dp2['lpos'], label="lpos")
  plt.legend()
  plt.xticks(np.arange(0,109,1))

  plt.xlabel("index")
  plt.ylabel("lpos")
  plt.title("Tracking Lpos")

  plt.show()

  plt.figure(figsize=(100,12))
  plt.plot(dp1['index'], dp1['new rposl'], label="rposl_line")
  plt.plot(dp1['index'], dp1['new rposr'], label="rposr_line")
  plt.scatter(dp2['index'], dp2['rpos'], label="rpos")
  plt.legend()
  plt.xticks(np.arange(0,109,1))

  plt.xlabel("index")
  plt.ylabel("rpos")
  plt.title("Tracking Rpos")

  plt.show()

if __name__ == '__main__':
  main()

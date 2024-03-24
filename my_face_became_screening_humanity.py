import cv2 as cv
import numpy as np

# 카메라 초기화
camera = cv.VideoCapture(0)  # 기본 카메라를 사용하려면 0으로 변경하거나 카메라 인덱스를 지정합니다.

# 카메라가 열렸는지 확인
if not camera.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

# 카메라 속성 가져오기
fps = camera.get(cv.CAP_PROP_FPS)
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

# 얼굴 감지를 위한 분류기 초기화
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # 프레임 읽기
    ret, frame = camera.read()

    if ret:
        # 그레이 스케일로 변환
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 얼굴 감지
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        

        for (x, y, w, h) in faces:
            # 얼굴에 사각형 그리기 (테두리)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 얼굴 영역 선택
            roi_color = frame[y:y+h, x:x+w]

            # 얼굴 부분을 검은색으로 채우기
            roi_black = np.zeros_like(roi_color)
            
            # 에지 감지
            gray_roi = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray_roi, 100, 200)

            # 흰색 테두리를 얼굴 부분에 적용
            roi_black[edges != 0] = [255, 255, 255]

            # 검은색 얼굴 부분을 원본 프레임에 복사
            frame[y:y+h, x:x+w] = roi_black

        # 영화같은 느낌을 주기위해 양 옆을 검게
        frame[:,:50] = 0  # 왼쪽 옆
        frame[:,-50:] = 0  # 오른쪽 옆

        # 프레임 출력
        cv.imshow('Movied_Face', frame)

        # 종료 키 확인
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

# 리소스 해제
camera.release()
cv.destroyAllWindows()

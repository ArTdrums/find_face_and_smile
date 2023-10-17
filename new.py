import cv2
from loguru import logger
import time

logger.add('debug_3.json', format='{time} {level} {message}',
           level="DEBUG", rotation='200 KB', compression='zip')
logger.debug('debug!')
logger.info('info')
logger.error('error')

start = time.perf_counter()


@logger.catch()
def face_funk(path_img, path_net):
    img = cv2.imread(path_img)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(path_net)
    coordinates_faces = face_cascade.detectMultiScale(img, scaleFactor=3, minNeighbors=3)  # находим лица

    # scaleFactor=2 на сколько большие изоражения можем искать, minNeighbors=3 -сколько соседей может быть

    for (x, y, width, height) in coordinates_faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), thickness=3)
        # cv2.imwrite("images/people_face_1.jpg", img)

        cv2.imshow('Results', img)
        cv2.waitKey(1000)


face_funk('images/people_img_1.jpg', 'cascades/haarcascade_frontalface_default.xml')


def smile_funk(path_img, path_net):
    img = cv2.imread(path_img)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(path_net)
    coordinates_faces = face_cascade.detectMultiScale(img, scaleFactor=4, minNeighbors=6)  # находим лица

    # scaleFactor=2 на сколько большие изоражения можем искать, minNeighbors=3 -сколько соседей может быть

    for (x, y, width, height) in coordinates_faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 255), thickness=3)
        # cv2.imwrite("images/people_face_1.jpg", img)

        cv2.imshow('Results', img)
        if cv2.waitKey(1000) == ord("q"):
            break


smile_funk('images/people_img_1.jpg', 'cascades_2/haarcascade_smile.xml')

print(time.perf_counter() - start)

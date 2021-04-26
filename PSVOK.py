import cv2
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui
import os
import des
from PyQt5.QtWidgets import QGridLayout, QWidget, QMessageBox
from PIL import Image

Img_Name_List = []


class ExampleApp(QtWidgets.QMainWindow, des.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.center()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.browse_folder)

    def center(self, ):
        frame_window = self.frameGeometry()
        center_coord = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame_window.moveCenter(center_coord)
        self.move(frame_window.topLeft())

    def browse_folder(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение")
        if directory[0]:
            try:
                Image.open(directory[0])
                self.lineEdit.setText(directory[0])
                mymain(directory[0])
                k = 0
                j = 'Result.jpg'
                pix = QtGui.QPixmap(j)
                self.label.setPixmap(pix)
                j = 'Result2.jpg'
                pix2 = QtGui.QPixmap(j)
                self.label_2.setPixmap(pix2)
                layout = QGridLayout()
                os.remove('Result.jpg')
                os.remove('Result2.jpg')
                for i in Img_Name_List:
                    lbl = QtWidgets.QLabel(self)
                    pix = QtGui.QPixmap(i)
                    lbl.setPixmap(pix)
                    layout.addWidget(lbl, 0, k)
                    k += 1
                    os.remove(i)
                Img_Name_List.clear()
                w = QWidget()
                w.setLayout(layout)
                self.scrollArea.setWidget(w)
            except IOError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Ошибка")
                msg.setInformativeText('Выберите изображение')
                msg.setWindowTitle("Ошибка")
                msg.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


def white_black(img4, h, w, indicator):
    for hi in range(0, h):
        for wi in range(0, w):
            (b, g, r) = img4[hi, wi]
            if g <= b <= g + 15 or g >= b >= g - 15:
                if g <= r <= g + 15 or g >= r >= g - 15:
                    if r < indicator and g < indicator and b < indicator:
                        img4[hi, wi] = (0, 0, 0)
                    else:
                        img4[hi, wi] = (255, 255, 255)
                else:
                    img4[hi, wi] = (255, 255, 255)
            else:
                img4[hi, wi] = (255, 255, 255)
    return img4


def image_resize(way):
    image = cv2.imread(way)
    h, w = image.shape[:2]
    if h > 1500 or w > 1500:
        if h >= w:
            w = int(w * float(1500 / h))
            h = 1500
        else:
            h = int(h * float(1500 / w))
            w = 1500
        image = cv2.resize(image, (w, h))
    return h, w, image


def image_resize2(image):
    h, w = image.shape[:2]
    if h > 1000 or w > 1000:
        if h >= w:
            w = int(w * float(1000 / h))
            h = 1000
        else:
            h = int(h * float(1000 / w))
            w = 1000
        image = cv2.resize(image, (w, h))
    return image


def f2(arr_of_area):
    print("Работа f2()")
    print(len(arr_of_area))
    i = 0
    while i < len(arr_of_area):
        j = 0
        while j < len(arr_of_area):
            j += 1
            try:

                if arr_of_area[i].x2 <= arr_of_area[j].x or arr_of_area[i].y2 <= arr_of_area[j].y:
                    continue
                elif arr_of_area[i].x >= arr_of_area[j].x2 or arr_of_area[i].y >= arr_of_area[j].y2:
                    continue
                elif i == j:
                    continue
                else:
                    if arr_of_area[i].x <= arr_of_area[j].x:
                        xmin = arr_of_area[i].x
                        xmax = arr_of_area[j].x

                    else:
                        xmin = arr_of_area[j].x
                        xmax = arr_of_area[i].x

                    if arr_of_area[i].x2 <= arr_of_area[j].x2:
                        x2min = arr_of_area[i].x2
                        x2max = arr_of_area[j].x2
                    else:
                        x2min = arr_of_area[j].x2
                        x2max = arr_of_area[i].x2

                    if arr_of_area[i].y <= arr_of_area[j].y:
                        ymin = arr_of_area[i].y
                        ymax = arr_of_area[j].y

                    else:
                        ymin = arr_of_area[j].y
                        ymax = arr_of_area[i].y

                    if arr_of_area[i].y2 <= arr_of_area[j].y2:
                        y2max = arr_of_area[j].y2
                        y2min = arr_of_area[i].y2
                    else:
                        y2max = arr_of_area[i].y2
                        y2min = arr_of_area[j].y2
                    wi = x2min - xmax
                    yi = y2min - ymax
                    s = wi * yi
                    si = (arr_of_area[i].y2 - arr_of_area[i].y) * (arr_of_area[i].x2 - arr_of_area[i].x)
                    sj = (arr_of_area[j].y2 - arr_of_area[j].y) * (arr_of_area[j].x2 - arr_of_area[j].x)
                    if s >= 0.7 * si or s >= 0.7 * sj:
                        arr_of_area[i].x = xmin
                        arr_of_area[i].y = ymin
                        arr_of_area[i].x2 = x2max
                        arr_of_area[i].y2 = y2max
                        arr_of_area[i].p = arr_of_area[i].p + arr_of_area[j].p - arr_of_area[i].p * arr_of_area[j].p
                        arr_of_area.pop(j)

            except IndexError:
                continue
        i += 1
    print(len(arr_of_area))
    return arr_of_area


def f21(arr_of_area, arr_of_vd):
    print("Работа f21()")
    i = 0
    while i < len(arr_of_area):
        j = 0
        while j < len(arr_of_vd):
            j += 1
            try:
                if arr_of_area[i].x2 <= arr_of_vd[j].x or arr_of_area[i].y2 <= arr_of_vd[j].y:
                    continue
                elif arr_of_area[i].x >= arr_of_vd[j].x2 or arr_of_area[i].y >= arr_of_vd[j].y2:
                    continue
                else:
                    arr_of_area[i].p = arr_of_area[i].p + arr_of_vd[j].p - arr_of_area[i].p * arr_of_vd[j].p
            except IndexError:
                continue
        i += 1
    return arr_of_area


class Area:
    def __init__(self, x, y, x2, y2, p):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.p = p


def f1(template1, stepen_shodstva, h1, w1, only_cat):
    arr_of_area = []
    res = cv2.matchTemplate(only_cat, template1, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= stepen_shodstva)
    if loc:
        for pt in zip(*loc[::-1]):
            x = int(pt[0])
            y = int(pt[1])
            area1 = Area(x, y, x + h1, y + w1, stepen_shodstva)
            arr_of_area.append(area1)
        return arr_of_area
    else:
        print("нет совпадений")
        print("Снизте порог сходства")


def f4(l1, l2):
    t = 0
    while t < len(l2):
        p = 0
        while p < len(l1):
            try:
                if l2[t].x == l1[p].x and l2[t].y == l1[t].y:
                    l2.pop(t)
                    break
                p += 1
            except IndexError:
                break
        t += 1
    return l2


def f3(name_of_temple, stepen_shodstva, only_cat):
    print("Работа f3()")
    list_f3 = []
    template1 = cv2.imread(name_of_temple)
    template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
    h, w = template1.shape[:2]
    for P in range(100, int(stepen_shodstva * 100), -10):
        print("Вероятность = {}".format(P))
        print("Размеры:")
        for i in range(100, 50, -10):
            w1 = int(w * i * 0.01)
            h1 = int(h * i * 0.01)
            print("h = {} w={}".format(h1, w1))
            if w1 <= 0 and h1 <= 0:
                continue
            dim = (w1, h1)
            template1 = cv2.resize(template1, dim, interpolation=cv2.INTER_AREA)
            list_f3 += f4(list_f3, f1(template1, float(P / 100), h1, w1, only_cat))
    return list_f3


def viola_jones(gray):
    list_of_vd = []
    micro_cascade = cv2.CascadeClassifier("cascade.xml")
    for minn in range(20, 0, -1):
        print(minn)
        micro = micro_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=minn,
            minSize=(10, 10)
        )
        list1 = []
        for (x, y, w, h) in micro:
            area1 = Area(x, y, x + h, y + w, float(minn / 20))
            list1.append(area1)
        list_of_vd += list1
    print(len(list_of_vd))
    return list_of_vd


def rectangle(only_cat, img2):
    list_of_rect = []
    contours0, hierarchy = cv2.findContours(only_cat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        x3 = box[2][0]
        y3 = box[2][1]
        x4 = box[3][0]
        y4 = box[3][1]
        s = float(1 / 2 * (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - x2 * y1 - x3 * y2 - x4 * y3 - x1 * y4))
        if s >= 1000:
            l1 = float((abs(x1 - x2) + abs(y1 - y2)) ** (1 / 2))
            l2 = float((abs(x3 - x2) + abs(y3 - y2)) ** (1 / 2))
            if abs(l1 - l2) < 3:
                area1 = Area(x3, y3, x1, y1, 0.5)
                list_of_rect.append(area1)
                cv2.rectangle(img2, (x3, y3), (x1, y1), (0, 255, 0), 10)
    return list_of_rect


def mymain(path):
    matrix = []
    h, w, img = image_resize(path)
    _, _, img2 = image_resize(path)
    _, _, img3 = image_resize(path)
    _, _, img4 = image_resize(path)
    _, _, img5 = image_resize(path)
    img4 = white_black(img4, h, w, 130)
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    rat, only_cat = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    matrix += f3("template\\TEMPLE-1.png", 0.3, only_cat)
    matrix += f3("template\\TEMPLE-2.png", 0.3, only_cat)
    for retc in matrix:
        cv2.rectangle(img2, (retc.x, retc.y), (retc.x2, retc.y2), (0, 255, 255), 2)

    while 1:
        lenght = len(matrix)
        matrix = f2(matrix)
        if len(matrix) == lenght:
            break
    a = 1

    templatek = 0.6
    for i in range(0, len(matrix)):
        matrix[i].p *= templatek
    micro = viola_jones(gray)
    matrix = f21(matrix, micro)
    area_rect = rectangle(only_cat, img2)
    matrix = f21(matrix, area_rect)
    for retc in micro:
        cv2.rectangle(img2, (retc.x, retc.y), (retc.x2, retc.y2), (255, 0, 0), 10)
    for retc in matrix:
        str1 = str(int(retc.p*100))
        str1 += "%"
        print(str1)
        cv2.putText(img3, str1, (retc.x2 + 10, retc.y + 10), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
        cv2.rectangle(img3, (retc.x, retc.y), (retc.x2, retc.y2), (0, 255, 255), 5)
        cropped = img5[retc.y:retc.y2, retc.x:retc.x2]
        a += 1
        picturename = "Result2" + str(a) + ".jpg"
        cropped = image_resize2(cropped)
        cv2.putText(cropped, str1, (0, 25), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
        cv2.imwrite(picturename, cropped)
        Img_Name_List.append(picturename)
    img3 = image_resize2(img3)
    cv2.imwrite("Result2.jpg", img3)
    img2 = image_resize2(img2)
    cv2.imwrite("Result.jpg", img2)
    return 0


if __name__ == '__main__':
    main()

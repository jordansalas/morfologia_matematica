import cv2
import numpy as np

class Granulometria:
    def __init__(self, file_path):
        self.list_image = []
        self.file_path = file_path
        self.img = cv2.imread(self.file_path)
        self.list_image.append(self.img)

    def draw(self):
        numpy_horizontal = np.hstack(self.list_image)
        cv2.imshow('output', numpy_horizontal)
        #cv2.imshow('output', self.img)
        cv2.waitKey(0)
        cv2.destroyWindow('ouput')

    def removeNoise(self):
        img_remove = cv2.fastNlMeansDenoising(self.img, None, 10, 10, 21)
        #img_remove = cv2.bilateralFilter(img_remove, 9, 75, 75)
        img_remove = cv2.adaptiveThreshold(img_remove, 255, 0, 1, 5, 2)
        self.list_image.append(img_remove)

def main():
    gra = Granulometria('image1.PNG')
    img_list = gra.removeNoise()
    gra.draw()

    print("FIN")

if __name__ == "__main__":
    main()

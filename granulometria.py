import cv2
import numpy as np

class Granulometria:
    def __init__(self, file_path):
        self.list_image = []
        self.file_path = file_path
        self.img = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        img_3ch = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.list_image.append(img_3ch)

    def draw(self):
        numpy_horizontal = np.hstack(self.list_image)
        cv2.imshow('output', numpy_horizontal)
        cv2.waitKey(0)
        cv2.destroyWindow('ouput')

    def removeNoise(self):
        kernel = np.ones((3, 3), np.uint8)
        kernel_ero = np.ones((5, 5), np.uint8)
        img_remove = cv2.bilateralFilter(self.img, 70, 75, 75)
        img_remove = cv2.fastNlMeansDenoising(img_remove, None, 15, 7, 21)
        cv2.imwrite('filteredImage.jpg', img_remove)
        #ret, img_remove = cv2.threshold(img_remove, 127, 255, cv2.THRESH_BINARY)
        #img_remove = cv2.adaptiveThreshold(img_remove, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                   cv2.THRESH_BINARY, 35, 13)
        #img_remove = cv2.adaptiveThreshold(img_remove, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                   cv2.THRESH_BINARY, 55, 55)
        ret2, img_remove = cv2.threshold(img_remove,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_remove = cv2.erode(img_remove, kernel_ero, iterations = 1)
        img_remove = cv2.dilate(img_remove, kernel, iterations = 1)
        img_3ch = cv2.cvtColor(img_remove, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('threshouldImage.jpg', img_remove)
        self.list_image.append(img_3ch)
        return img_remove

    def contours(self, img):
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.copy(self.img)
        img_3ch = cv2.cvtColor(img_contours, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_3ch, contours, -1, (0,255,0))

        cv2.imwrite('objectImage.jpg', img_3ch)
        self.list_image.append(img_3ch)


def main():
    gra = Granulometria('image1.PNG')
    img = gra.removeNoise()
    gra.contours(img)
    gra.draw()

    print("FIN")

if __name__ == "__main__":
    main()

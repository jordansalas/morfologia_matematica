import cv2

def draw(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

def main():
    draw("lena.jpg")

if __name__ == "__main__":
    main()

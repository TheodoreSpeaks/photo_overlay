
if __name__ == '__main__':
    pass


def run_haar():
    img = cv2.imread("thispersondoesnotexist.jpeg")

    sunglasses = cv2.imread('sunglasses.png')
    plt.imshow(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye = eye_classifier.detectMultiScale(gray, 1.3, 9)

    if len(eye) == 0:
        pass

    print(eye[0])
    (x,y,w,h) = eye[1]
    # for (x,y,w,h) in eye:
    if h > 0 and w > 0:
        print(x)
        print(y)
        print(img.shape)
        print(sunglasses.shape)


        h, w = int(h*3), int(w*3)
        y -= 120
        x -= 20

        img_roi = img[y:y+h, x:x+w]

        print(w, h)

        print(img_roi.shape)
        print(sunglasses.shape)
        sunglasses_small = cv2.resize(sunglasses, (w, h),  interpolation=cv2.INTER_AREA)
        gray_sunglasses = cv2.cvtColor(sunglasses_small, cv2.COLOR_BGR2GRAY)
        print(sunglasses_small.shape)

        ret, mask = cv2.threshold(gray_sunglasses, 230, 255,  cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        masked_face = cv2.bitwise_and(sunglasses_small, sunglasses_small, mask=mask)

        masked_frame = cv2.bitwise_and(img_roi,  img_roi, mask=mask_inv)

        print("result")
        print(masked_frame.shape)
        print(masked_face.shape)

        img[y:y+h, x:x+w] = cv2.add(masked_face,  masked_frame)

    plt.figure()

    plt.imshow(img)

import glob
import cv2
import numpy

image_list = []
eigen_face_number = 10


def callback(value):
    new_output = average_face
    for i in range(0, eigen_face_number):
        new_wight = cv2.getTrackbarPos('W' + str(i), 'Controller')
        weight = new_wight - 150  # get changes
        new_output = numpy.add(new_output, weight * eigen_vector_list[i])

    new_output = cv2.resize(new_output, (0, 0), fx=4, fy=4) # increase quality of image
    cv2.imshow('Output', new_output)


# reading data set
for name in glob.glob('DataSet/*.jpg'):
    image = cv2.imread(name)
    normal_image = numpy.float32(image) / 255.0
    image_list.append(normal_image)

print('Number of Images: ', len(image_list))
print('Number of Selected Eigen Faces: ', eigen_face_number)

# vectoring image
size = image_list[0].shape
vectored_image_list = numpy.zeros((len(image_list), size[0] * size[1] * size[2]), dtype=numpy.float32)
for i in range(0, len(image_list)):
    vectored_image_list[i] = image_list[i].flatten()

# calculate eigen vectors
avg_face, eigen_vector = cv2.PCACompute(vectored_image_list, mean=None, maxComponents=eigen_face_number)

# convert vectorized data to original size
eigen_vector_list = []
for ev in eigen_vector:
    eigen_vector_list.append(ev.reshape(size))
average_face = avg_face.reshape(size)

# create controller
cv2.namedWindow('Controller', cv2.WINDOW_NORMAL)
for i in range(0, eigen_face_number):
    cv2.createTrackbar('W' + str(i), 'Controller', 150, 300, callback)

# show output
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 250, 250)
output = cv2.resize(average_face, (0, 0), fx=4, fy=4)  # increase quality of image
cv2.imshow('Output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()

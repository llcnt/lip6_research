import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

import torchvision.transforms as transforms

import torchvision.datasets as dsets
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

METHOD = 'uniform'
radius = 1
n_points = 8 * radius

trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
training_set = dsets.CIFAR10(root='data', train=True,  transform=trans, download=True)
testing_set = dsets.CIFAR10(root='data', train=False, transform=trans, download=True)

#%%


def lbp_piece_vec(img, method, p, r, regions=4):
    '''
    :param img: input grayscale image
    :param method: LBP method
    :param p: number of neighbors
    :param r: radius
    :param regions: number of regions divided
    :return: lbp vector
    '''
    # crop the 32x32 image into 4 8x8 regions
    pieces = []
    vectors = []
    for i in range(regions):
        for j in range(regions):
            pieces.append(img[8*i:8*(i+1), 8*j:8*(j+1)])

    for piece in pieces:
        lbp = local_binary_pattern(piece, p, r, method)
        his, _ = np.histogram(lbp, density=True, bins=p+2, range=(0, p+2))
        vectors.append(his)

    return np.concatenate(vectors)

def lbp_vec(img, method, p, r):
    '''
    :param img: input grayscale image
    :param method: LBP method
    :param p: number of neighbors
    :param r: radius
    :return: lbp vector
    '''
    # convert a single image into graph directly
    lbp = local_binary_pattern(img, p, r, method)
    his, _ = np.histogram(lbp, density=True, bins=p+2, range=(0, p+2))
    return his

def LBP_data(training_set, testing_set):
    '''
    :param training_set:
    :param testing_set:
    :return: array like training and testing data
    '''
    train_lbp_vectors = []
    train_labels = []

    for img, lab in training_set:
        img = img.numpy().squeeze()
        vector = lbp_vec(img, METHOD, n_points, radius)
        train_lbp_vectors.append(np.expand_dims(vector, axis=0))
        train_labels.append(lab)

    train_img = np.concatenate(train_lbp_vectors, axis=0)
    train_lab = np.array(train_labels)


    test_lbp_vectors = []
    test_labels = []

    for img, lab in testing_set:
        img = img.numpy().squeeze()
        vector = lbp_vec(img, METHOD, n_points, radius)
        test_lbp_vectors.append(np.expand_dims(vector, axis=0))
        test_labels.append(lab)

    test_img = np.concatenate(test_lbp_vectors, axis=0)
    test_lab = np.array(test_labels)

    return train_img, train_lab, test_img, test_lab


#%%
train_img, train_lab, test_img, test_lab = LBP_data(training_set, testing_set)
np.save('train_img.npy', train_img)
np.save('train_lab.npy', train_lab)
np.save('test_img.npy', test_img)
np.save('test_lab.npy', test_lab)

#%%
train_img, train_lab, test_img, test_lab = np.load('train_img.npy'), np.load('train_lab.npy'), np.load('test_img.npy'), np.load('test_lab.npy')

print(type(train_img), train_img.shape)
print(type(test_img), test_img.shape)
print('finished the processing of images')

#%%
classifier = LinearSVC()
classifier.fit(train_img, train_lab)
print('fit the model')
pre_labs = classifier.predict(test_img)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_lab, pre_labs)))

#%%

classifier = KNeighborsClassifier()
classifier.fit(train_img, train_lab)
print('fit the model')
pre_labs = classifier.predict(test_img)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_lab, pre_labs)))

#%%
trans = transforms.ToPILImage()
images, labels = training_set.__getitem__(1)
lbp = local_binary_pattern(images.numpy().squeeze(), n_points, radius, METHOD)
plt.imshow(images.numpy()[0, :, :], cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(lbp.ravel(), density=True, bins=n_points+2, range=(0, n_points+2), color='0.5')
plt.show()
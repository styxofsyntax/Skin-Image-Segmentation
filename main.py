import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# rbg values in mask
dej = [255, 172, 255]
drm = [0, 255, 190]
epi = [160, 48, 112]
ker = [224, 224, 224]
bkg = [0, 0, 0]

# intensities for each layer
bkg_vset = [i for i in range(248, 256)]
epi_vset = [i for i in range(125, 200)]
drm_vset = [i for i in range(20, 60)] + [i for i in range(185, 253)]
ker_vset = [i for i in range(235, 256)]
dej_vset = [i for i in range(40, 250)]

np.set_printoptions(suppress=True)

def list_files(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def CCA(image, vSet):
    label = np.zeros(image.shape, dtype=int)
    label_counter = 0

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] in vSet:
                uniqueL = uniqueNeighbors(label, [i, j])
                uniqueL = [i for i in uniqueL if i != 0]

                if len(uniqueL) == 0:
                    label_counter += 1
                    label[i][j] = label_counter
                    continue

                if len(uniqueL) == 1:
                    label[i][j] = uniqueL[0]

                else:
                    label[i][j] = uniqueL[0]
                    replace_indices = np.isin(label, uniqueL[1:])

                    label[replace_indices] = uniqueL[0]

    return label_counter - 1, label


def uniqueNeighbors(image, index):
    direction = [[0, -1], [-1, 0], [-1, -1], [-1, 1]]
    neighbors = []

    for x in direction:
        i = index[0] + x[0]
        j = index[1] + x[1]

        if i < 0 or i >= image.shape[0]:
            continue
        elif j < 0 or j >= image.shape[1]:
            continue

        neighbors.append(int(image[i][j]))

    return np.sort(np.unique(neighbors))


def detectBackground(img):
    ret, thresh = cv.threshold(img, bkg_vset[0], 255, cv.THRESH_BINARY)
    num_of_labels, labels = CCA(thresh, [255])
    output = np.zeros_like(img)

    label_freq = np.bincount(labels.flatten(), minlength=num_of_labels)
    max_label = np.argmax(label_freq[1:]) + 1

    if np.sum(label_freq[max_label]) >= 2000:
        output[labels == max_label] = np.uint8(1)

    return output


def detectLayer(img, vset, mostFreq=False, threshold=180):
    num_of_labels, labels = CCA(img, vset)
    output = np.zeros_like(img)

    label_freq = np.bincount(labels.flatten(), minlength=num_of_labels)

    if mostFreq:
        label_freq = np.bincount(labels.flatten(), minlength=num_of_labels)
        max_label = np.argmax(label_freq[1:]) + 1

        if np.sum(label_freq[max_label]) >= 2000:
            output[labels == max_label] = np.uint8(1)

    else:
        for label in range(1, num_of_labels):
            if label_freq[label] > threshold:
                output[labels == label] = np.uint8(1)

    return output


def postProcessing(img, layer_code, threshold=220):
    image = np.copy(img)

    num_of_labels, labels = CCA(img, [0])
    label_counts = np.bincount(labels.flatten(), minlength=num_of_labels)

    for label in range(1, num_of_labels):
        if label_counts[label] <= threshold:
            image[labels == label] = layer_code

    return image


def applyColor(img):
    background_img_color = (np.zeros((512, 512, 3), dtype=np.uint8) * 255)

    background_img_color[img == 1] = bkg
    background_img_color[img == 2] = drm
    background_img_color[img == 3] = epi
    background_img_color[img == 4] = ker
    background_img_color[img == 5] = dej

    return background_img_color


def dice_coff(my_mask, original_mask):
    total_area = original_mask.shape[0] * original_mask.shape[1]
    # Computing number of overlapping pixels
    overlap = 0
    for i in range(original_mask.shape[0]):
        for j in range(original_mask.shape[1]):
            if (my_mask[i, j] == original_mask[i, j]).all():
                overlap += 1
    print(overlap, total_area)
    return overlap / total_area


def combineDEJ(img, width=45):
    img_pad = np.pad(img, ((width, width), (width, width)),
                     mode='constant', constant_values=0)
    img_to_change = np.copy(img_pad)

    for i in range(width, img_pad.shape[0]):
        for j in range(width, img_pad.shape[1]):
            neighborhood = img_pad[i - width: i + width, j - width: j + width]

            count_2 = np.count_nonzero(neighborhood == 2)
            count_3 = np.count_nonzero(neighborhood == 3)

            if count_3 != 0:
                ratio_drm_epi = count_2 / count_3
                if (0.2 < ratio_drm_epi <= 1):
                    img_to_change[i, j] = 5

    return img_to_change[width:-width, width:-width]


def layerProcessing(img):
    background_img_raw = detectBackground(img)
    background_img = postProcessing(background_img_raw, 1, 50)

    print("background done!")

    drm_img_raw = detectLayer(img, drm_vset, True) * 2
    total_pixels_drm = np.count_nonzero(drm_img_raw)
    drm_img = postProcessing(
        drm_img_raw, 2, threshold=int(total_pixels_drm * 0.1))

    print("Dermis (Green) done!")

    epi_img_raw = detectLayer(img, [i for i in range(110, 180)], True) * 3

    print("Epidermis (Purple) done!")

    ker_img = detectLayer(img, ker_vset, threshold=100) * 4

    print("Keratin (Gray) done!")

    output = np.copy(background_img)

    output[(background_img == 0) & (ker_img == 4)] = 4
    output[(background_img == 0) & (drm_img == 2)] = 2
    output[(background_img == 0) & (epi_img_raw == 3)] = 3

    output[output == 0] = 3  # filling empty spaces with purple

    output = combineDEJ(output, 20)  # adding pink junction

    print("Dermal-Epidermal Junction (Pink) done!")

    output_color = applyColor(output)

    return output_color


filenames = list_files("Test/Tissue")
for fName in filenames:
    img = cv.imread(f"Test/Tissue/{fName}", 0)
    print(img.shape)
    output = layerProcessing(img)
    cv.imwrite(f"Test_Output/{fName[:-4]}.png", output)

from skimage import exposure
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_fill_holes

from skimage.filters import threshold_otsu, threshold_adaptive, sobel
from skimage.transform import resize
from skimage.morphology import binary_dilation, remove_small_objects, dilation, binary_closing, disk
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
plt.gray()

import matplotlib.patches as patches
import pickle
from keras.models import model_from_json
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd
import uuid


def imglabel(im):
    """Locates and returns a list of individual characters in the image"""

    grayim = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    grad = sobel(grayim)
    threshold = threshold_otsu(grad)
    newim = grad>threshold
    newim = binary_closing(newim, selem=disk(4))
    newim = remove_small_objects(newim, min_size=400)
  
    labeled_digits, _ = ndi.label(newim)

  
    resized_segmented_ims = []
    coords = []
    for i in np.unique(labeled_digits)[1:]:

        try:
            y, x = np.where(labeled_digits == i)
            coords.append([x.min(), y.min(), x.max(), y.max()])
            
            char = grayim[y.min()-2: y.max()+2, x.min()-2:x.max()+2].astype(float)
            char /= char.max()
            char = -char + 1


            perc = 20. / max(char.shape[0], char.shape[1])
            w,h = int(char.shape[0]*perc), int(char.shape[1]*perc)
        
            # Pad the image           CHANGE TO // for P3, / for P2
            wpad = (28 - w)/2
            hpad = (28 - h)/2

            w_off=0
            h_off=0
            if h + 2*hpad == 27:
                h_off = 1
            if w + 2*wpad == 27:
                w_off = 1


            char = resize(char, (w,h))


            thresh = threshold_otsu(char)
            char[char < thresh] = 0.0

            # Binarize?
            char[char >= thresh] = 1.0
            

            char = np.pad(char, ((wpad, wpad+w_off), (hpad, hpad+h_off)), mode='constant', constant_values=0.)

            char = binary_dilation(char, selem=disk(1))

            resized_segmented_ims.append(char)
            
        except Exception as e:
            print e
        
    return coords, resized_segmented_ims


def scan_image(im):
    """Returns position and value of all digits found in image"""

    coords, segmented_ims = imglabel(im)


    X = np.array(segmented_ims).reshape(len(segmented_ims), 1,  28, 28)
    keras = model_from_json(open('keras_v3_arch.json').read())
    keras.load_weights('keras_v3_weights.h5')

    keras.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    #Predict Classes
    y_pred = keras.predict_proba(X, verbose=0)

    return coords, y_pred


def save_segmented_img(filepath):

    fig, ax = plt.subplots(1,1, figsize=(12, 16))

    im = plt.imread(filepath)
    plt.imshow(im)
    
    coords, probs = scan_image(im)

    norm = colors.Normalize(vmin=.80, vmax=1.00, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdYlGn"))

    for coord, prob in zip(coords, probs):

        xmin, ymin, xmax, ymax = coord
        
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),   # (x,y)
                xmax - xmin,    # width
                ymax - ymin,    # height
                fill=True,
                color = mapper.to_rgba(prob.max()),
                alpha=.2

        ))
        ax.annotate(str(prob.argmax()), xy=(coord[0], coord[1]))

    table = tableizer(coords, probs)
    path = "static/" + str(uuid.uuid4()) + ".png"
    fig.savefig(path, bbox_inches='tight')
    return path, table

def tableizer(results, label_list):
    label_list = label_list.argmax(axis=1)

    for id in range(len(results)):
        results[id].append(label_list[id])
    
    
    k=1
    for i in range(len(results)-1):
        x_min1, y_min1, x_max1, y_max1 = results[i][:4]
        x_min2, y_min2, x_max2, y_max2 = results[i+1][:4]
        list1 = range(y_min1, y_max1)
        list2 = range(y_min2, y_max2)
        zlist = [j for j in list1 if j in list2]
        if len(zlist) > 0:
            results[i].append(k)
        else:
            results[i].append(k)
            k += 1

    results[-1].append(k)
    
    
    sorted_list = sorted(results, key = lambda x: (x[5], x[0]))
    
    p = 1
    for x in range(len(sorted_list)-1):
        x_min1, y_min1, x_max1, y_max1 = sorted_list[x][0:4]
        x_min2, y_min2, x_max2, y_max2 = sorted_list[x+1][0:4]
        height1 = (y_max1 - y_min1)*0.8
        height2 = (y_max2 - y_min2)*0.8
        if x_min2 - height1 > x_max1 and x_max1 + height2 < x_min2 or x_min2 < x_max1:
            sorted_list[x].append(p)
            p+=1
        else:
            sorted_list[x].append(p)

    sorted_list[-1].append(p)
    
    col_nums = list(filter(lambda x: x[5] == 1, sorted_list))

    unique_cols = []
    for x in range(len(col_nums)):
        unique_cols.append(col_nums[x][-1])
        
    unique_cols = list(set(unique_cols))

    unique_cells = []
    for x in range(len(sorted_list)):
        unique_cells.append(sorted_list[x][-1])

        
    unique_cells = list(set(unique_cells))
    
    
    bounds = []
    for col in unique_cols:
        digits = filter(lambda x: x[6] == col, sorted_list)
        digits = list(digits)
        max_y = digits[0][3]; min_y = digits[0][1]
        bheight = int((max_y - min_y)*0.40)
        max_x = digits[-1][2]; min_x = digits[0][0]
        bounds.append(range(min_x-bheight, max_x+bheight))
    
    
    column_bound_tup = list((zip(unique_cols, bounds)))

    column_bound_list = []
    for x in column_bound_tup:
        column_bound_list.append(list(x))
        
    cell_bounds = []
    for cell in unique_cells:
        cell_digits = filter(lambda x: x[6] == cell, sorted_list )
        cell_digits = list(cell_digits)
        max_x = cell_digits[-1][2]; min_x = cell_digits[0][0]
        cell_bounds.append(range(min_x, max_x))

    col_labels = []
    for cell in cell_bounds:
        for col in column_bound_list:
            zlist = [i for i in cell if i in col[1]]
            if len(zlist) > 0:
                col_labels.append(col[0])
                
    column_labeled_digits =[]
    for id in range(len(col_labels)):
        col_lab_digits = filter(lambda x: x[6] == id+1, sorted_list )
        col_lab_digits = list(col_lab_digits)
        for cell in col_lab_digits:
            column_labeled_digits.append(col_labels[id])
            
    for id in range(len(column_labeled_digits)):
        sorted_list[id].append(column_labeled_digits[id])
        
    cell_full_digits = []
    for cell_id in unique_cells:
        cell_digits = filter(lambda x: x[6] == cell_id, sorted_list )
        cell_digits = list(cell_digits)

        digit = ""
        for cell in cell_digits:
            digit += str(cell[4]) 

        cell_full_digits.append((digit, cell_digits[-1][5], cell_digits[-1][7]))
        
    frows = []
    for cell in cell_full_digits:
        frows.append(cell[1])

    fcols = []
    for cell in cell_full_digits:
        fcols.append(cell[2])

    dimensions = (max(frows), max(fcols))

    initial = np.empty(dimensions)*np.nan

    for digit in cell_full_digits:
        d, row, col = digit
        initial[row - 1][col - 1] = d
        
    final = pd.DataFrame(initial)
    
    return final





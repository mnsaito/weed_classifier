# Standard Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

from PIL import Image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score

# Lime Imports
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable

store_path = os.path.join(os.pardir, os.pardir, 'reports', 'figures')

# Compare perennials with weeds, randomly selected from training data
def compare_plants():
    data_path = os.path.join(os.pardir, os.pardir, 'data', 'train')
    perennial_path = os.path.join(data_path, 'perennials')
    weed_path = os.path.join(data_path, 'weeds')
    plants_compared = os.path.join(store_path, 'plants_compared')
    
    perennial_file = random.choice(os.listdir(perennial_path))
    perennial_img = mpimg.imread(os.path.join(perennial_path, perennial_file))
    weed_file = random.choice(os.listdir(weed_path))
    weed_img = mpimg.imread(os.path.join(weed_path, weed_file))

    plt.figure(figsize = (20, 8))
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Perennial', fontdict={'size': 18})
    plt.imshow(perennial_img)
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Weed', fontdict = {'size': 18})
    plt.imshow(weed_img);
    plt.savefig(plants_compared)
    return

# Function to plot loss, accuracy and precision during training
def visualize_training_results(results):
    """Input results = model.fit
    requires both accuracy and precision as metrics
     
    """
    
    loss_plot = os.path.join(store_path, 'loss_plot')
    acc_plot = os.path.join(store_path, 'acc_plot')
    prec_plot = os.path.join(store_path, 'prec_plot')

    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(loss_plot)
    plt.show()
    
    plt.figure()
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(acc_plot)
    plt.show()
        
    plt.figure()
    plt.plot(history['val_precision'])
    plt.plot(history['precision'])
    plt.legend(['val_precision', 'precision'])
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.savefig(prec_plot)
    plt.show();
    return

def create_confusion_matrix(model, generator):
    """Input model and generator
    Creates confusion matrix     
    """
    c_matrix = os.path.join(store_path, 'confusion_matrix')
    preds = (model.predict(generator) > 0.5).astype('int32')
    true_labels = generator.classes
    labels = list(generator.class_indices.keys())
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    fig, ax = plt.subplots(figsize = (8, 8))
    disp.plot(ax = ax)
    plt.grid(False)
    plt.savefig(c_matrix)
    plt.show();
    return

def get_metrics(model, generator):
    """Input model and generator
    Prints accuracy and recall     
    """
    preds = (model.predict(generator) > 0.5).astype('int32')
    true_labels = generator.classes
    print('Accuracy:', accuracy_score(true_labels, preds))
    print('Precision:', precision_score(true_labels, preds))
    return

def get_lime(model, generator, batch_no, plant, path):
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20, 10))
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(generator[0][0][batch_no].astype('double'), 
                                             model.predict, top_labels = 1,
                                             hide_color = 0, num_samples = 1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only = False, num_features = 10, 
                                                hide_rest = False)
    ax1.set_title(plant, fontdict={'size': 18})
    ax2.set_title('Lime (pro in green; con in red)', fontdict={'size': 18})
    ax1.imshow(generator[0][0][batch_no])
    ax2.imshow(mark_boundaries(temp, mask))
    plt.savefig(path)
    plt.show();
    return

def display_lime(model, generator):
    """Input model and generator
    Displays Lime for 3 random x-rays selected from generator using model
    """
    peren_flag = 0
    weed_flag = 0
    
    while peren_flag + weed_flag < 2:
        # Randomly select image in generator
        batch_no = random.choice(range(len(generator[0][0])))

        if peren_flag == 0:
            if generator[0][1][batch_no] == 0:
                plant = 'Perennial'
                peren_flag = 1
                path = os.path.join(store_path, 'lime_peren')
                get_lime(model, generator, batch_no, plant, path)
        elif weed_flag == 0:
            if generator[0][1][batch_no] == 1:
                plant = 'Weed'
                weed_flag = 1
                path = os.path.join(store_path, 'lime_weed')
                get_lime(model, generator, batch_no, plant, path)
    return






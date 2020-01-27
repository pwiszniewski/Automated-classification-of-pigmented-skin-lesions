# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:45:05 2019

@author: SW
"""

import numpy as np
import numpy.ma as ma
from scipy.stats import kurtosis
from scipy.stats import skew
import cv2
import matplotlib.pyplot as plt
import math


def calc_parameters(img_org, show_params=False, show_hist=False, show_img=False):
    img_dst = np.copy(img_org)
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_sat = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)[:,:,1]
    
    # Processing
    img_blur = cv2.medianBlur(img_gray, 7)
    # img_blur = cv2.GaussianBlur(img_gray, (7,7), 0)
    
    height, width = img_org.shape[0], img_org.shape[1]
    img_black = np.zeros((height,width,1), dtype=np.uint8)
    # img_black = cv2.circle(img_black, (int(width/2), int(height/2)), 
    #                        int(height/2), 255, -1)
    cv2.circle(img_black, (int(width/2), int(height/2)), 
                           int(height/2), 255, -1)
#    cv2.imshow(' ', img_black)
#    cv2.waitKey(0)
    # hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    # _, img_binar = cv2.threshold(hsv[:,:,2], 120, 255, cv2.THRESH_BINARY)
    # img_black = cv2.bitwise_or(img_binar, img_black)
    # cv2.ellipse(img_black,(int(width/2), int(height/2)),(int(0.9*width/2), int(0.9*height/2)),0,360,0,255,-1)
    # cv2.imshow(' ', img_black)
    # cv2.waitKey(0)
    
    img_masked = ma.masked_array(img_blur, img_black>0) #cv2.bitwise_and(img_blur, img_black)
    ret, _ = cv2.threshold(img_masked,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, img_binar = cv2.threshold(img_blur, ret, 255, cv2.THRESH_BINARY_INV)
    img_binar = cv2.bitwise_and(img_binar, img_black)
    if np.sum(img_binar) < 200*255:
        # dst = concatenate_images(img_binar, img_masked, img_dst)
        # cv2.imshow('pre', dst)
        
        ret = ret + 40
        _, img_binar = cv2.threshold(img_blur, ret, 255, cv2.THRESH_BINARY_INV)
        img_binar = cv2.bitwise_and(img_binar, img_black)

        # dst = concatenate_images(img_binar, img_masked, img_dst)
        # cv2.imshow('post', dst)
            
    kernel_er = np.ones((3,3),np.uint8)
    img_eroded = cv2.erode(img_binar, kernel_er, iterations = 3)
    kernel_cl = np.ones((11,11),np.uint8)
    img_closed = cv2.morphologyEx(img_eroded, cv2.MORPH_CLOSE, kernel_cl)
    kernel_grad = np.ones((3,3),np.uint8)
    img_grad = cv2.morphologyEx(img_closed, cv2.MORPH_GRADIENT, kernel_grad)
    
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    mask = cv2.bitwise_not(img_grad)
    img_masked = cv2.bitwise_and(img_dst,img_dst,mask = mask)
    
    # Find contours
    if np.sum(img_closed) > 20*255:
        image, contours, hierarchy = cv2.findContours(img_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        image, contours, hierarchy = cv2.findContours(img_binar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area and perimeter
    max_area = 0
    max_contour = None
    for c in contours: 
        cont_area = cv2.contourArea(c)
        if cont_area > max_area:
            max_contour = c
            max_area = cont_area
    # dst = concatenate_images(img_binar, img_grad, img_dst)
    # cv2.imshow('post', dst)
    # cv2.waitKey(0)

    perimeter = cv2.arcLength(max_contour,True)   
    (x,y),radius = cv2.minEnclosingCircle(max_contour)
    
    ellipse = cv2.fitEllipse(max_contour)
    axes = ellipse[1]
    minor_ell, major_ell = axes
    min_maj_ell_ratio = minor_ell / major_ell
    perimeter_ell = np.pi*( 3/2*(minor_ell+major_ell) - np.sqrt(minor_ell*major_ell) )
    perimeter_ratio =  perimeter_ell / perimeter
    center = (int(x),int(y))
    radius = int(radius)
    circle_area = np.pi*radius**2
    circ_area_ratio = max_area / circle_area
    
    # Histogram
    hist_mask = np.zeros(img_org.shape)
    hist_mask = cv2.fillPoly(hist_mask, pts=[max_contour], color=(255,255,255)).astype(np.uint8)[:,:,0]
    img_hist = cv2.bitwise_and(img_org, img_org, mask=hist_mask)
    
    color = ('b','g','r')
    histograms = []
    for i,col in enumerate(color):
        hist = cv2.calcHist([img_org],[i],hist_mask,[256],[0,256])
        histograms.append(hist)
    #    print(np.argmax(hist))
        if show_hist:
            plt.plot(hist,color = col)
            plt.xlim([0,256])
    if show_hist:
        plt.show()
    
    img_hist_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_hist_gray = cv2.bitwise_and(img_hist_gray, img_hist_gray, mask=hist_mask)
    
    hist = cv2.calcHist([img_hist_gray],[0],hist_mask,[256],[0,256])
    histograms.append(hist)
    if show_hist:
        plt.plot(hist, color = 'm')
        plt.xlim([0,256])
        plt.show()
    
    #data = histograms[1]
    #for d in data:
    
    #plt.hist(data, bins=60)
    hist_params = ['mean', 'var', 'skew', 'kurt']
    hist_params_dict = {param: [] for param in hist_params}
    hue = cv2.cvtColor(img_hist, cv2.COLOR_BGR2HSV)[:,:,0]
    hist_types = ['blue', 'green', 'red', 'gray', 'hue']
    dataset = [img_hist[:,:,0], img_hist[:,:,1], img_hist[:,:,2], img_hist_gray, hue]
    for hist_type, data in zip(hist_types, dataset):
        data = data[data>0].ravel()
        hist_params_dict[f'{hist_params[0]}'].append(np.mean(data))
        hist_params_dict[f'{hist_params[1]}'].append(np.var(data))
        hist_params_dict[f'{hist_params[2]}'].append(skew(data))
        hist_params_dict[f'{hist_params[3]}'].append(kurtosis(data))
        
    params_dict = {}
    # Calculate hu moments
    retval	= cv2.moments(hist_mask)
    hu = cv2.HuMoments(retval, 7)
    for i in range(0,7):
        if hu[i] != 0:
            hu[i] = -1* math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))
        params_dict[f'hu{i}'] = float(hu[i])
        
    params_dict['max_area'] = int(max_area)
    params_dict['circ_area_ratio'] = circ_area_ratio
    params_dict['perimeter'] = int(perimeter)
    params_dict['min_maj_ell_ratio'] = min_maj_ell_ratio
    params_dict['perimeter_ratio'] = perimeter_ratio
    for p in hist_params_dict:
        for i,colorscale in enumerate(hist_types):
            params_dict[p+'_'+colorscale] = hist_params_dict[p][i]
         
    if show_params:
        print('\t', ' '.join(f'{ht:6}' for ht in hist_types))
        print('mean:\t', ' '.join('{:6.2f}'.format(p) for p in hist_params_dict[f'{hist_params[0]}']))
        print('var:\t', ' '.join('{:6.2f}'.format(p) for p in hist_params_dict[f'{hist_params[1]}']))
        print('skew:\t', ' '.join('{:6.2f}'.format(p) for p in hist_params_dict[f'{hist_params[2]}']))
        print('kurt:\t', ' '.join('{:6.2f}'.format(p) for p in hist_params_dict[f'{hist_params[3]}']))
        # print('kurt:\t', ' '.join('{:6.2f}'.format(p) for p in hist_params_dict[f'{hist_params[3]}']))
        print('max contour area:', int(max_area))
        print('max contour length:', int(perimeter))
        print('circle area ratio:', int(circ_area_ratio*100))
        print('hu moments:')
        for i,h in enumerate(hu):
            print(f'\thu{i}: {h[0]}')
    
    if show_img:
        img_dst = cv2.ellipse(img_dst, ellipse, (0, 0, 255), 2)
        img_dst = cv2.circle(img_dst,center,radius,(255,0,0),2)
        img_dst = cv2.drawContours(img_dst, contours, -1, (255,255,255), 2)
#        cv2.imshow('img_org',img_org)
#        cv2.imshow('binar', img_binar)
#        cv2.imshow('gradient', img_grad)
#        cv2.imshow('img_masked', img_masked)
#        cv2.imshow('closing', img_closed)
#        cv2.imshow('img_hist',img_hist)
#        cv2.imshow('img_hist_gray',img_hist_gray)
        # cv2.imshow('hist_mask',hist_mask)
        # cv2.imshow('img_dst',img_dst)
        
        dst = concatenate_images(img_blur, img_binar, img_eroded,
                                  img_closed, hist_mask, img_dst)
#        dst = concatenate_images(img_blur, img_masked, hist_mask,
#                                  img_binar, img_closed, img_dst)
        cv2.imshow(' ', dst)
        # cv2.imshow(' ', hsv[:,:,0])
        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
    return histograms, params_dict
    

def concatenate_images(*images):
    images_high = [img if len(np.shape(img)) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images[:3]]
    images_low = [img if len(np.shape(img)) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images[3:]]
    dst_high = np.hstack(images_high)
    if len(images_low) == 3:
        dst_low = np.hstack(images_low)
        dst = np.vstack([dst_high, dst_low])
    else:
        dst = dst_high
    return dst

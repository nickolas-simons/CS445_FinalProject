from os import path
import math

# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN as db
from itertools import combinations

#Applies Mask to the image
def prep_image(img, chroma_key,sigma,ksize, threshold):
    oned_fil = cv2.getGaussianKernel(ksize, sigma) # 1D kernel
    twod_fil = oned_fil*np.transpose(oned_fil)

    im_fil_low = img.copy()#cv2.filter2D(img,-1,twod_fil)
    
    dim = 3 if len(im_fil_low.shape) == 3 else 1

    for i in range (im_fil_low.shape[0]):
        for j in range(im_fil_low.shape[1]):
            color = img[i][j]
            difference = np.abs(chroma_key-color)
            im_fil_low[i][j] = np.ones(dim) if (np.sum(difference) < threshold) else np.zeros(dim)
    
    im_fil_low = np.clip(cv2.filter2D(im_fil_low,-1,twod_fil),0,1)
    plt.imshow(im_fil_low)
    plt.show()
    
    return im_fil_low

#Function that calculates Hough and its Helper functions are listed below
from itertools import product
def compute_line(line):
    """Returns line coefficients (a, b, c) from points: ax + by + c = 0"""
    a = line[1] - line[3]
    b = line[2] - line[0]
    c = line[0]*line[3] - line[2]*line[1]
    return a, b, -c

def intersection(L1, L2):
    """Find intersection point of two lines given in ax + by + c = 0 form"""
    D = L1[0] * L2[1] - L1[1] * L2[0]
    if D == 0:
        return None  # parallel lines
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    x = Dx / D
    y = Dy / D
    return x, y

def hough(img):
    mask = prep_image(img,(0,1,0),10,(int)(img.shape[0]/10),0.95)
    masked_img =  (mask*img * 255).astype(np.uint8)
    plt.imshow(masked_img)
    plt.show()

    masked_gray =  cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    edges = cv2.Canny(masked_gray,25,75)
    plt.imshow(edges)
    plt.show()
    
    lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi / 180,threshold=img.shape[0]//10,minLineLength=img.shape[0]/4,maxLineGap=img.shape[0]//10)
    line_detection_img = img.copy()

    thetas = []
    for x1,y1,x2,y2 in lines[:,0]:
        theta = np.arctan2(y2-y1, x2-x1) + 3*np.pi/2
        thetas.append([theta])
    thetas = np.array(thetas)

    labels = db(eps=0.1, min_samples=1).fit_predict(thetas)

    vertical = []
    horizontal = []
    vert_dir = np.zeros(2,dtype=np.float64)
    horz_dir = np.zeros(2)
    for i in range(len(lines)):
        x1,y1,x2,y2 = lines[i][0]
        if labels[i] == 1:
            vertical.append(lines[i][0])
            vert_dir += [x2 - x1, y2 - y1]
            color = [1,0,0]
        elif labels[i] == 0:
            horizontal.append(lines[i][0])
            horz_dir += [x2 - x1, y2 - y1]
            color = [0,0,1]
        else:
            v1 = np.array([x2 - x1, y2 - y1],dtype=np.float64)
            v1 /= np.linalg.norm(v1)
            normalized_horz_dir = horz_dir/np.linalg.norm(horz_dir)
            normalized_vert_dir = vert_dir/np.linalg.norm(vert_dir)

            if abs(np.dot(v1, normalized_vert_dir)) > 0.95:
                vertical.append(lines[i][0])
                vert_dir += [x2 - x1, y2 - y1]
                color = [1,0,0]
            elif abs(np.dot(v1, normalized_horz_dir)) > 0.95:
                horizontal.append(lines[i][0])
                horz_dir += [x2 - x1, y2 - y1]
                color = [0,0,1]
            else:
                color = [0,1,0]
        cv2.line(line_detection_img, (x1, y1), (x2, y2), color, 3)
    
    points = []
    for a, b in product(vertical, horizontal):
        L1 = compute_line(a)
        L2 = compute_line(b)
        pt = intersection(L1, L2)
        if pt:
            points.append(pt)
            plt.scatter(pt[0], pt[1], color='pink', s=5, marker='o')
    points = np.array(points)
    plt.imshow(line_detection_img)
    plt.show()

    from sklearn.cluster import KMeans
    if points.size > 81:
        kmeans = KMeans(n_clusters=81, n_init='auto')
        labels = kmeans.fit_predict(points)
        points = kmeans.cluster_centers_
    plt.imshow(img)
    plt.scatter(points[:,0],points[:,1], color='yellow', s=5, marker='o')
    plt.show()

    corners = np.zeros((4,2))
    hull = cv2.convexHull(points.astype(np.float32))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4:
        corners = approx[:, 0, :]  # shape (4, 2)
    
    return corners

#Function for chroma key method is below
def chroma_key(img):
    mask = prep_image(img,[0,1,0],30,15, 0.95)
    masked_img = mask*img

    top = [-1,-1]
    bottom = [9999,9999]
    left = [9999,9999]
    right = [-1,-1]
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if masked_img[j][i][2] > 0.015:
                if(j > top[1]):
                    top = [i,j]
                if(j < bottom[1]):
                    bottom = [i,j]
                if(i < left[0]):
                    left = [i,j]
                if(i > right[0]):
                    right = [i,j]
    
    return [top,right,bottom,left]

#Function for FastLineDetector and helper functions below
def othello_prep_image(img):
    chroma_key=np.array([0, 1, 0])  
    sigma=30
    ksize=15
    threshold=0.95

    oned_fil = cv2.getGaussianKernel(ksize, sigma)
    twod_fil = oned_fil @ oned_fil.T
    # Step 2: Prepare the mask based on chroma key
    mask = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = img[i, j].astype(np.float32) / 255.0  # Normalize to [0, 1]
            difference = np.abs(chroma_key - color)
            mask[i, j] = np.ones(3) if np.sum(difference) < threshold else np.zeros(3)

    # Step 3: Apply the Gaussian filter
    mask = np.clip(cv2.filter2D(mask, -1, twod_fil), 0, 1)

    # Step 4: Apply the mask to the original image
    masked_img = (mask * (img.astype(np.float32) / 255.0)) * 255
    masked_img = masked_img.astype(np.uint8)

    return masked_img

def get_dynamic_length_threshold(image, lines, scale_factor=0.15, min_length=20):
    # Get image dimensions
    height, width = image.shape[:2]
    diagonal_length = np.sqrt(width**2 + height**2)
    
    # Use the diagonal length to set a base threshold
    base_threshold = int(diagonal_length * scale_factor)
    
    # Collect all line lengths
    line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for line in lines for x1, y1, x2, y2 in [line[0]]]
    avg_length = np.mean(line_lengths) if line_lengths else 0
    
    # Set the threshold based on the average line length
    adaptive_threshold = max(min_length, int(avg_length * 0.5))
    
    # Return the larger of the two thresholds to avoid cutting too many lines
    return max(base_threshold, adaptive_threshold)


def fast_line_detector(img):
    # mask = prep_image(img,(0,1,0),10,(int)(img.shape[0]/10),0.95)
    # masked_img =  (mask*img * 255).astype(np.uint8)

    masked_img = othello_prep_image(img)
    plt.figure(figsize=(3, 3))  # Adjust the size as needed
    plt.imshow(masked_img)
    plt.title('masked_image')
    plt.axis("off")  # Optional, hides the axes
    plt.show()
    resize_factor = 0.5
    resized_image = cv2.resize(masked_img, (0, 0), fx=resize_factor, fy=resize_factor)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Use GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 1: Use Canny edge detection to highlight edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    fld = cv2.ximgproc.createFastLineDetector()

    # Step 2: Detect lines
    lines = fld.detect(edges)
    #print(f"lines: {lines}, lines type: {type(lines)}")

    # Use dynamic threshold for line length filtering
    dynamic_threshold = get_dynamic_length_threshold(resized_image, lines, scale_factor=0.15)
    print(f"Dynamic Threshold: {dynamic_threshold}")
    endpoints = []
    line_image = resized_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Apply dynamic length threshold
            if length > dynamic_threshold:
                endpoints.append(((x1, y1), (x2, y2)))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Step 3: Find all intersections of the longest lines (Represent corners)
    corners = []
    image_height, image_width = resized_image.shape[:2]
    for (p1, p2), (q1, q2) in combinations(endpoints, 2):
        # Line 1 points
        x1, y1 = p1
        x2, y2 = p2
        # Line 2 points
        x3, y3 = q1
        x4, y4 = q2
        # Line intersection formula
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue  # Parallel lines
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        # Filter points within image bounds
        if 0 <= px <= image_width and 0 <= py <= image_height:
            corners.append((int(px), int(py)))
    
    filtered_corners = []
    min_distance = 50  # Minimum distance to consider two points as separate corners
    for corner in corners:
        if all(np.linalg.norm(np.array(corner) - np.array(existing)) > min_distance for existing in filtered_corners):
            filtered_corners.append(corner)
   
    scale_factor_x = img.shape[1] / resized_image.shape[1]
    scale_factor_y = img.shape[0] / resized_image.shape[0]

    original_corners = []
    for corner in filtered_corners:
    # Scale the corner coordinates back to the original image size
        original_corner = [int(corner[0] * scale_factor_x), int(corner[1] * scale_factor_y)]
        original_corners.append(original_corner)
    # Ensure corners are sorted in clockwise order
    if original_corners:
        # Calculate the centroid
        centroid = np.mean(original_corners, axis=0)

        # Sort corners based on their angle relative to the centroid
        def angle_from_centroid(point):
            dx = point[0] - centroid[0]
            dy = point[1] - centroid[1]
            return np.arctan2(dy, dx)

        original_corners.sort(key=angle_from_centroid)

        original_corners = original_corners[-2:] + original_corners[:-2]

    #print(f"Sorted corners (clockwise): {original_corners}")
    return original_corners




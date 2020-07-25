#!/usr/bin/env python3

#import libraries
# import matplotlib.pyplot as plt
import numpy as np
import  sys
import random
import cv2
from operator import itemgetter
import time
          
#########################################################################
################# Part 1 functions start here ###########################
#########################################################################

#function to read image
def read_image(image_path, gray = True):
    """
    Args
    
    image_path: Image path with filename e.g.,part1-images/eiffel.jpg
    gray: Boolean Flag to read images in gray scale
    
    Returns:
    Image in python readable format, a 2D or 3D array
    
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path)

#function to get orb features of an image, default of 1000 features
def orb_features(img, nfs = 1000):
    """
    Args
    
    img: Image in form of a 2D array
    nfs: Number of keypoints to extract from an image
    
    Returns:
    keypoints: Keypoints in the image
    descriptors: 32 dimensional descriptor corresponding to each keypoint

    """
    orb = cv2.ORB_create(nfeatures=nfs)
    # detect features 
    (keypoints, descriptors) = orb.detectAndCompute(img, None)
    return keypoints, descriptors

#function to calculate hamming distance between two descriptors
def ham_dist(descrp1, descrp2):
    """
    Args
    
    descrp1: descriptor vector/list
    descrp2: descriptor vector/list same dimensional as descrp1
    
    Returns:
    dist: hamming distance between descrp1 and descrp2

    """
    dist = cv2.norm(descrp1, descrp2, cv2.NORM_HAMMING)
    return dist

#function which returns strong keypoints and descriptors from the given based on response
def strong_keypoints(keypoints, descriptors):
    """
    Args
    
    keypoints: keypoints of an image
    descriptors: descriptors of an image
    
    Returns:
    kp: strong keypoint based on response thresholding
    desc: descriptors corresponding to kp

    """
    threshold = np.array(list(map(lambda x:x.response, keypoints))).mean()
    indices = [ind for ind, val in enumerate(keypoints) if val.response > threshold]
    kp = [keypoints[i] for i in indices]
    desc = [descriptors[i] for i in indices]
    return kp, desc

#function to match the keypoints between two images and return the count of matching points
def match(img1, img2, threshold = 0.8):
    """
    Args
    
    img1: Image 1
    img2: Image 2
    
    Returns:
    match_count: number of matching keypoints
    dis: chamfer distance between keypoints of both images

    """
    match_count = 0
    mat_kp1 = []
    mat_kp2 = []
    
    k1, d1 = orb_features(img1)
    kp1, desc1 = strong_keypoints(k1, d1)
#    kp1, desc1 = k1, d1 #uncommente if we want to explore all keypoints
    
    k2, d2 = orb_features(img2)
    kp2, desc2 = strong_keypoints(k2, d2)
#    kp2, desc2 = k2, d2 #uncomment if we want to explore all keypoints
    
    dis = 0
    for i in range(len(kp1)):
        match_tup = []
        for j in range(len(kp2)):
            distance = ham_dist(desc1[i], desc2[j])
            match_tup.append((distance, kp1[i], kp2[j]))
            
        best2 = sorted(match_tup,key=itemgetter(0))[:2]
        first = best2[0][0]
        second = best2[1][0]
        ratio = first/second
        
        if ratio < threshold:
            match_count += 1
            dis += first
            mat_kp1.append(best2[0][1])
            mat_kp2.append(best2[0][2])
        
    return match_count, dis, mat_kp1, mat_kp2


#function to draw lines between matching points of 2 images
def draw_matchlines(img1, img2, mat_kp1, mat_kp2):
    """
    Args
    
    img1: Image 1
    img2: Image 2
    mat_kp1: Matching keypoints of image 1
    mat_kp2: Matching keypoints of image 2
    
    Returns:
    image: Image with img1 and img2 sid-by-side with lines joining keypoints 

    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if len(img1.shape) == len(img2.shape):
        if len(img1.shape) == 3:
            # if img1 and img2 are RGB
            image = np.zeros((max(h1,h2), w1+w2,3), np.uint8) 
            image[0:h1, 0:w1,:] = img1
            image[0:h2, w1:w1+w2,:] = img2
        else:
            # if imag1 and img2 are grayscale
            image = np.zeros((max(h1,h2), w1+w2), np.uint8) 
            image[0:h1, 0:w1] = img1
            image[0:h2, w1:w1+w2] = img2
    
    else:
        print("Both images should be either grayscale or RGB")
        return None     
    
    #keypoint matching lines
    for k1, k2 in list(zip(mat_kp1, mat_kp2)):
        img1_p = (int(k1.pt[1]), int(k1.pt[0]))
        img2_p = (int(k2.pt[1]), int(w1+k1.pt[0]))
        cv2.line(image,(img1_p[1],img1_p[0]),(img2_p[1],img2_p[0]),(255,255,255),2)
        
    return image


#function to calculate similarity matrix for list of images given
def similarity_mat(images, use_count_flag = True):
    """
    Args
    
    images: List of images
    use_count_flag: Boolean flag, to use match count in similarity matrix
                    else uses chamfer distance
    
    Returns:
    matrix: similarity matrix 

    """
    matrix  = np.zeros((len(images), len(images)))
    
    if use_count_flag:
        #this puts count of matches in the similarity matrix
        for i in range(len(images)):
            for j in range(i+1):
                matrix[i][j] = match(images[i],images[j])[0]
                matrix[j][i] = matrix[i][j]
    else:
        #this puts chamfer distance in the similarity matrix
        for i in range(len(images)):
            for j in range(i+1):
                matrix[i][j] = match(images[i],images[j])[1] + match(images[j],images[i])[1]
                matrix[j][i] = matrix[i][j]
                
    return matrix

 
#function to cluster similar images
def kmeans(n_clusters, img_sim_matrix, iterations = 100):
    """
    Args
    
    n_clusters: Number of clusters (K)
    img_sim_matrix: similarity matrix of images
    iterations: number of iterations to run kmeans clustering
    
    Returns:
    final_clusters: dictionary of cluster number and correspoinding image position 

    """
    # eigen vectors of similarity matrix are used for clustering
    eigen_values, eigen_vectors = np.linalg.eigh(img_sim_matrix)
    
    data = eigen_vectors #[:,2:] #we can select only top few eigen vectors to speedup clustering
    cluster_centers = data[np.random.choice(data.shape[0], size=n_clusters, replace=False), :]
    
    # some parts of below code in this function is adapted from
    # https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
    for i in range(iterations+1):
        dist_matrix = np.array([]).reshape(data.shape[0],0)
        for k in range(n_clusters):
            tempDist=np.sum((data-cluster_centers[k])**2,axis=1)
            dist_matrix=np.c_[dist_matrix,tempDist]
        C=np.argmin(dist_matrix,axis=1)+1
        Y={}
        for k in range(n_clusters):
            Y[k+1]=np.array([]).reshape(data.shape[1],0)
        for j in range(data.shape[0]):
            Y[C[j]]=np.c_[Y[C[j]],data[j]]
             
        for k in range(n_clusters):
            Y[k+1]=Y[k+1].T
            
        for k in range(n_clusters):
             cluster_centers[k]=np.mean(Y[k+1],axis=0)
    
    # final cluster assignment
    final_clusters = {i:C[i] for i in range(len(C))}
    
    return final_clusters

#function to evaluate accuracy of clustering 
def evaluation(pred_clusters, files):
    """
    Args
    
    pred_clusters: dictionary of cluster number and correspoinding image position
    files: image filenames order maintained as per image position
    
    Returns:
    accuracy: pairwise clustering accuracy 

    """
    N = len(pred_clusters)
    TP = 0
    TN = 0
    
    for i in range(N):
        for j in range(N):
            if i != j:
                act_cluster1 = files[i].strip().split('_')[0]
                act_cluster2 = files[j].strip().split('_')[0]
                if act_cluster1 == act_cluster2:
                    if pred_clusters[i] == pred_clusters[j]:
                        TP += 1
                else:
                    if pred_clusters[i] != pred_clusters[j]:
                        TN += 1
                        
    accuracy = (TP +TN)/(N*(N-1))
    print('Number of True positives are {}'.format(TP))
    print('Number of True negatives are {}'.format(TN))
    print('Accuracy of clustering is {}'.format(accuracy))
    
    return accuracy

#function to write clustering output to a file
def cluster_outfile(clusters, files, outfile_name = "output_file.txt"):
    """
    Args
    
    clusters: dictionary of cluster number and correspoinding image position
    files: image filenames order maintained as per image position
    outfile_name: output filename along with file extension
    
    Returns:
    None
    writes image file names in different lines corresponding to clusters 
    to output file 

    """
    K = len(clusters)
    with open(outfile_name, "w") as f:
        for i in range(1,K+1):
            indices = [k for k,v in clusters.items() if v == i]
            ff = [files[ind] for ind in indices]
            
            
            write = ' '.join(ff)
            f.write(write+"\n")
                
    f.close()                   

#########################################################################
################# Part 1 functions end here #############################
#########################################################################


#########################################################################
################# Part 2 functions start here ###########################
#########################################################################


#Function to calculate near by pixels after warping
def near_by_pixels(matrix,col,row):
    """
    Args
    
    matrix: Tranformation matrix
    col: column id of pixel in original space
    row: row id of pixel in original space
    
    Returns:
    a: row coordinate in warped space
    b: col coordinate in warped space
    b_ceil : higher col coordinate in warped space
    a_ceil : higher row coordinate in warped space
    b_floor : lower col coordinate in warped space
    a_floor : lower row coordinate in warped space
    
    """
    new = np.dot(matrix,[col,row,1])
    col_new = new[0]/new[2]
    row_new = new[1]/new[2] 
    a = row_new - np.floor(row_new)
    b = col_new - np.floor(col_new)
    b_ceil,a_ceil = int(np.ceil(col_new)),int(np.ceil(row_new))
    b_floor,a_floor = int(np.floor(col_new)),int(np.floor(row_new))
    return a,b,b_ceil,a_ceil,b_floor,a_floor

#Bilinear interpolation to extrapolte pixel values
def bilinear_interpolation(inv_matrix ,col,row,image):
    """
    Args
    
    inv_matrix: iversed Tranformation matrix
    col: column id of pixel in warped space
    row: row id of pixel in warped space
    image: original image
    
    Returns:
    bli_value: pixel value at [row,col] in warped image 
    
    """
    a,b,b_ceil,a_ceil,b_floor,a_floor = near_by_pixels(inv_matrix ,col,row)
    bli_value = (1-b)*(1-a)*image[a_floor,b_floor] + \
                                        (1-b)*(a)*image[a_ceil,b_floor]+ \
                                        b*(1-a)*image[a_floor,b_ceil] + \
                                        b*a*image[a_ceil,b_ceil]
    return bli_value
   
#Function to transform the image using the transformation matrix
def transform(image,matrix):
    """
    Args
    
    matrix: Tranformation matrix
    image: original image
    
    Returns:
    warp_image: warped image 
    
    """    
    h,w,d = image.shape
    warp_image = np.zeros(image.shape)           
    inv_matrix = np.linalg.inv(matrix) 
    for col in range(w):
        for row in range(h):
            # inverse warping 
            a,b,b_ceil,a_ceil,b_floor,a_floor = near_by_pixels(inv_matrix ,col,row)
            if 0<= b_floor and b_ceil < w  and 0<= a_floor and a_ceil < h:
                #bilinear interpolation
                warp_image[row,col,:] = bilinear_interpolation(inv_matrix ,col,row,image)


    return warp_image


# Function to find Tranformation matrix based on feature correspondings
def tranformation_matrix(n,p_1,p_2):
    """
    Args
    n: Type of tranformation ( n=1 :translation, n = 2: Euclidean,
                               n = 3: affine, and n = 4  projective )
    p_1: feature correspondings in image 1 
    p_2: feature correspondings in image 2 
    
    Returns
    matrix: Transformation matrix 
    
    """     
    if n==1:
        #only translation
        tx = p_2[0][0]-p_1[0][0]
        ty = p_2[0][1]-p_1[0][1]
        matrix = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    elif n==2:
        #Euclidean (rigid) transformation,
        #AX = X' lets create A matrix solve system of equations
        A = [[p_1[0][0], -p_1[0][1], 1, 0 ],
             [p_1[0][1],  p_1[0][0], 0, 1],
             [p_1[1][0], -p_1[1][1], 1, 0 ],
             [p_1[1][1],  p_1[1][0], 0, 1]]
        X_prime = p_2.flatten()
        matrix = np.linalg.solve(A, X_prime)
        matrix = np.array([[matrix[0],-matrix[1],matrix[2]],[matrix[1],matrix[0],matrix[2]],[0,0,1]])
    elif n==3:
        #affine transformation
        A = [[p_1[0][0], p_1[0][1],  1,     0 ,        0,      0],
             [  0,           0,      0, p_1[0][0], p_1[0][1],  1],
             [p_1[1][0], p_1[1][1],  1,     0 ,        0,      0],
             [  0,           0,      0, p_1[1][0], p_1[1][1],  1],
             [p_1[2][0], p_1[2][1],  1,     0 ,        0,      0],
             [  0,           0,      0, p_1[2][0], p_1[2][1],  1]]
        X_prime = p_2.flatten()
        matrix = np.linalg.solve(A, X_prime)
        matrix = np.array([matrix[0:3],matrix[3:],[0,0,1]])
    elif n==4:
        # projective Transformation
        A = [[p_1[0][0], p_1[0][1], 1,     0,         0,     0, -p_2[0][0]*p_1[0][0], -p_2[0][0]*p_1[0][1]],
             [    0,         0,     0, p_1[0][0], p_1[0][1], 1, -p_2[0][1]*p_1[0][0], -p_2[0][1]*p_1[0][1]],
             [p_1[1][0], p_1[1][1], 1,     0,         0,     0, -p_2[1][0]*p_1[1][0], -p_2[1][0]*p_1[1][1]],
             [    0,         0,     0, p_1[1][0], p_1[1][1], 1, -p_2[1][1]*p_1[1][0], -p_2[1][1]*p_1[1][1]],
             [p_1[2][0], p_1[2][1], 1,     0,         0,     0, -p_2[2][0]*p_1[2][0], -p_2[2][0]*p_1[2][1]],
             [    0,         0,     0, p_1[2][0], p_1[2][1], 1, -p_2[2][1]*p_1[2][0], -p_2[2][1]*p_1[2][1]],
             [p_1[3][0], p_1[3][1], 1,     0,         0,     0, -p_2[3][0]*p_1[3][0], -p_2[3][0]*p_1[3][1]],
             [    0,         0,     0, p_1[3][0], p_1[3][1], 1, -p_2[3][1]*p_1[3][0], -p_2[3][1]*p_1[3][1]]]
        
        X_prime = p_2.flatten()
        matrix = np.linalg.solve(A, X_prime)
        matrix = np.array([matrix[0:3],matrix[3:6],np.append(matrix[6:],1)])
    else:
        print("please enter the n values in the range of [1,4]" )
    
    return matrix

#########################################################################
################# Part 2 functions end here #############################
#########################################################################


#########################################################################
################# Part 3 functions start here ###########################
#########################################################################


#Calculates votes for an hypothesis
def hypothesis_votes(match_p_r,match_p_l,Inlier_threshold,matrix):
    """
    Args
    
    match_p_r: Matching points of Right image
    match_p_l: Matching points of Left image 
    Inlier_threshold: Squared distance threshold to consider as Inliers
    e_limit: Inliers ratio threshold for the transform to be valid 
    matrix:  transformation matrix
    
    Returns:
    e: support inliers ratio for the hypothesis   
    
    """
    votes = 0    
    for j in range(len(match_p_r)):
        pr = match_p_r[j]
        pl = match_p_l[j]            
        pr_t = matrix.dot(np.array([pr[0],pr[1],1]))   
        if pr_t[2]==0:
            continue
        pr_t = np.array([pr_t[0]/pr_t[2],pr_t[1]/pr_t[2]])
        dst = np.linalg.norm(pr_t-pl)
        
        if dst < Inlier_threshold:
            votes += 1
    e = votes/len(match_p_r)
    return e 

# RANSAC Function to find best transformation matrix from image 2 -> image 1
# Adapted some part of RANSAC function from below link
# https://stackoverflow.com/questions/4655334/ransac-algorithm
def ransac(match_p_r,match_p_l,iters,Inlier_threshold,e_limit,n=4):
    """
    Args
    
    match_p_r: Matching points of Right image
    match_p_l: Matching points of Left image 
    iters: number of iterations to run
    Inlier_threshold: Squared distance threshold to consider as Inliers
    e_limit: Inliers ratio threshold for the transform to be valid 
    n: Number of points required to calculate transformation matrix
    
    Returns:
    best_model: Matrix that best describes Transformation from Right image to Left image   
    
    """
    best_model = np.array([])
    
    if len(match_p_r) < n:
        # if match points in image_1 are less than n, then cant find Transformation matrix.
        # print("Match Points are less than",str(n),"can't perform RANSAC")
        return np.array([])
    best_e = 0 #Inlier ratio
    for i in range(iters):  
        #randomly selecting 4 matching points between 2 images
        rand_indices = random.sample(range(0, len(match_p_r)), n)
        base_points = match_p_r[rand_indices] 
        final_points = match_p_l[rand_indices]
        
        #Calculating matrix hypothesis
        try:
            matrix = np.array(tranformation_matrix(n,base_points,final_points))
        except:
            continue  
        #calculting inliers ratio for hypothesis
        e =hypothesis_votes(match_p_r,match_p_l,Inlier_threshold,matrix)
        #comparing ratios for best model till now
        if best_e < e:
            best_e = e
            best_model = matrix
     #checking whether best_e obtained is > than threshold else 
     #recalculating matrix with reducing threshold by 0.05   
    if best_e > e_limit :   
        return best_model
    else:
        return ransac(match_p_r,match_p_l,iters,Inlier_threshold,e_limit-0.05,n)

#Crops the image(removes black)
def crop_image(pan_image):
    idx = np.argwhere(np.all(pan_image[..., :] == 0, axis=0))                
    pan_image = np.delete(pan_image, idx, axis=1)
    return pan_image    

#Function for stiching two images together using the transformation matrix
def panorama(image1,image2,matrix):
    
    """
    Args
    
    image1: Left image
    image2: Right Image
    matrix: Transformation matrix calculated using RANSAC Method
    
    Returns:
    pan_image: Returns Panorama image constructed using Left and Right images 
    
    """

#referred to https://github.com/tsherlock/panorama/blob/master/pano_stitcher.py
#Took the logic to decide on panorama image size

    # Finding min and max row, col of image2 projection on image1
    h, w, z = image2.shape
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(matrix, p)
    
    row_list = p_prime[1] / p_prime[2]
    col_list = p_prime[0] / p_prime[2]
    row_min = min(row_list)
    col_min = min(col_list)
    row_max = max(row_list)
    col_max = max(col_list)
    
    new_row_min = min(0,np.round(row_min))
    new_col_min = min(0,np.round(col_min))
    new_row_max = max(image1.shape[0],np.round(row_max))
    new_col_max = max(image1.shape[1],np.round(col_max))
    
    new_mat = np.array([[1, 0, -1 * new_col_min], [0, 1, -1 * new_row_min], [0, 0, 1]])
    matrix = np.dot(new_mat, matrix)
    
    # createing a new image with width of image 1 + image 2
    # and height as max of image 1 & image 2
    pan_width = int(new_col_max-new_col_min)
    pan_height = int(new_row_max-new_row_min)
    pan_depth = image1.shape[2]
    pan_image = np.zeros((pan_height,pan_width,pan_depth))
    inv_matrix = np.linalg.inv(matrix)
    # Warping the image2 and stiching with image1
    #Assigng image1 to the left part of Panorama image
    pan_image[int(-new_row_min):len(image1)+int(-new_row_min),0:len(image1[0])] = image1
    
    #Update the commom part between image1 and warpped image2 with average values
    #and remaining right side part with corresponding part of warpped image2
    
    for col in range(pan_width):
        for row in range(pan_height):
            x,y,y_ceil,x_ceil,y_floor,x_floor = near_by_pixels(inv_matrix ,col,row)
            if (x_ceil>=0 and x_ceil<image2.shape[0]  and x_floor>=0 and x_floor<image2.shape[0] and y_ceil>=0 and y_ceil<image2.shape[1]  and y_floor>=0 and y_floor<image2.shape[1] ):
                bi_pixel =    bilinear_interpolation(inv_matrix ,col,row,image2)
                #checking for overlap
                
                if not sum(pan_image[row,col]):
                    pan_image[row,col] = bi_pixel
                
#                if sum(pan_image[row,col]):
#                    pan_image[row,col] = (pan_image[row,col]+bi_pixel)/2
#                    #pixel.astype('int')
#                else:
#                    pan_image[row,col] = bi_pixel
                    
    pan_image = crop_image(pan_image)
    return pan_image

      
        
 #Main function
    print ("Code Execution started")
if __name__ == '__main__':
    part = sys.argv[1]
    
    if part == "part1":
        start_time = time.time()
        print ("------------ Started Part1")
        k = int(sys.argv[2])
        image_list = sys.argv[3:]
        out_file = image_list.pop()
          
        
        filenames = image_list
        
        if k <= len(image_list):
        
            #read all images
            imgs = [read_image(filenames[i]) for i in range(len(filenames))]
            
            #generate a similarity matrix
            sim_matrix = similarity_mat(imgs, use_count_flag = False)
            
            #clustering images
            clusters = kmeans(k, sim_matrix, iterations = 500)
            
            #evaluate clustering
            eval_metric = evaluation(clusters, filenames)
            
            #write clustered images filename into different lines in out_file
            cluster_outfile(clusters, filenames, outfile_name = out_file)
            
        else:
            print("Number of images you provide should atleast be equal to number of clusters")
            
        print("--------- Total time taken: {} minutes".format((time.time()-start_time)/60))
        print("Ended Part1 ------------")    
    
    elif part =="part2":   
        print ("------------ Started Part2")
#        img = imageio.imread("part2-images/lincoln.jpg")
#        matrix = np.array([[.907, .258, -182], [-.153, 1.44, 58], [-.000306, 0.000731, 1]])
#        new_image = transform(img, matrix)
#        plt.imsave("lincol_transformed.png",new_image.astype('uint8'))
        n,img_1,img_2,img_output = int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
        p = sys.argv[6:]
        p_1 = np.array([list(map(int,p[i].split(','))) for i in range(0,len(p),2)])
        p_2 = np.array([list(map(int,p[i].split(','))) for i in range(1,len(p),2)])
        image_1 = cv2.imread(img_1)
        new_matrix = tranformation_matrix(n,p_1,p_2)
        print(new_matrix)
        Estimated_img_2=transform(image_1,new_matrix)
        cv2.imwrite(img_output,Estimated_img_2.astype('int'))
        print("Ended Part2 ------------") 
        
    elif part == "part3":
        print ("------------ Started Part3")
        
        image_1,image_2,output = sys.argv[2],sys.argv[3],sys.argv[4]
        
        #reading greyscale images
        image1 =  cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
        
        #get the matching points
        count,dis, match1,match2 = match(image1,image2)
        
        #reading RGB images
        image1 =  cv2.imread(image_1)
        image2 = cv2.imread(image_2)
        
        points1 = np.array([i.pt for i in match1])
        points2 = np.array([i.pt for i in match2])
        
        #RANSAC parameters
        #Calculating number of rounds(iterations required  
        # if p =0.99 , s = 4 and assuming outlier ratio not greater than  0.7
        N = int(np.log(1-0.99)/(np.log(1-np.power(1-0.8,4))))  #2875
        Inlier_threshold = 3  #threshold for error
        ratio = 0.8 # inliers ratio limit
        n = 4 #no of points to find the tranform matrix
        
        #Finding best transformation matrix from image 2 -> image 1
        transform_matrix = ransac(points2,points1,N,Inlier_threshold,ratio,n)
        
        if transform_matrix.size == 0:
            print("Match Points are less than",str(n),"can't perform RANSAC")
        
        else:
            # Warping & Stitching : 
            # Doing panorama using the transformation matrix and original images
            output_image = panorama(image1,image2,transform_matrix)
            
            #Saving the output to drive
            cv2.imwrite(output,output_image.astype('int')) 
        print("Ended Part3 ------------") 
        

        
        
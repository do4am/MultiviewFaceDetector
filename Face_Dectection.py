################### IMPORT NECCESARY PACKAGES #######################
#Package to do computation
import numpy as np
import random as rd

#Package to work with Image
import ntpath
from PIL import Image, ImageDraw, ImageFont

#Package to work with heat-map printing
import matplotlib.pyplot as plt
import matplotlib.cm as CM

#Packages to work with XMLs files
from lxml import etree as ET
from math import pow
from nms import nms_average, nms_max

#Package to work with Caffe
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#####################################################################
######################### SETTING PATHS #############################

#path to filelist of testing images
FILELIST_PATH = 'Data\\TestingSet\\newFileList1.txt'

#path to fully-conv network model
MODEL_PATH = 'Data\\Fully-Convolutional Network\\fully_conv.prototxt'

#path to fine-tuned caffemodel
MODEL_PARAMETERS_PATH = 'Data\\Fully-Convolutional Network\\alexNet2_finetunned_evaluation_newmean_iter_441_full_conv.caffemodel'

#path to save temporary modified network model when the code run
MODIFIED_MODEL_SAVE_PATH = 'Temp\\ffc2-deploy.prototxt'

#path to meanfile of data
MEAN_FILE_PATH = 'Data\\Fully-Convolutional Network\\mean_evaluation.npy'

#####################################################################
###################### SETTING PARAMETERS ###########################

#step to slide windows across the images (in pixels)
STRIDE = 32

#smallest size of input image that allow to feed into the network (size of 227 x 227 for alex net) (in pixels)
WINDOW_SIZE = 227

#The smallest windows' size = Upscaling_factor / 227 =  13 pixels (13 x 13) (in pixels)
UPSCALLING_FACTOR = 2975

#enlarge the smallest window size gradually by a given rate
DOWN_SAMPLING_RATE = 2**(-1/3) #cube root of 0.5

#threshold of score
OUTPUT_SCORE_THRESH = 0.8 #0.85 - v1 , 0.8 - v2

#The IOU of the max-score bounding box to other bounding boxes which is greater than a threshold will be removed
OVERLAPPING_THRESH = 0.8 #0.8 - This Threshold is to be used with NMS _ AVERAGE scheme, remove bounding box that has confidence < 20%

#Window's score in the cluster after filtered by NMS_max  lower the over max score of the cluster 10% will be removed
CLUSTER_SCORE_THRESH= 0.05 #0.05

#The score of the final bounding boxes should be greater than this number or they will be removed
CONFIDENCE_THRESH = 0.9  #0.945 - v2 , 0.9 - v1

#Count number of images in testing set
INPUT_IMG_COUNTER = 0

#Count number of faces detected (given the fact that only 1 face in 1 image)
FACE_DETECTED_COUNTER = 0

#Sum of all Scores from each faces detected
SUM_SCORE = 0

#Plot heatmap one time
RunOne = 1
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

root = ET.Element("dataset")
#save all box detected into .txt file
#resultData = open("result_fullname.txt","w+")
resultData = open("Results\\Logs\\resultText.txt","w+")

#####################################################################
###################### FUNCTION DEFINITION ##########################

# Generating of Bounding Box with given featureMap
def makeBoundingBoxes(featureMaps, scale):
    boundingBox = []
    for (x, y), score in np.ndenumerate(featureMaps):
        #adjust threshold to experiment which one is the best outcome
       if(score >= OUTPUT_SCORE_THRESH):

           #boundingbox array store all min and max locations of (x,y) and score. Then rescale all to fit with the original images.
            boundingBox.append([float(STRIDE * y)/ scale, float(x * STRIDE)/scale,
                                float(STRIDE * y + WINDOW_SIZE - 1)/scale, float(STRIDE * x + WINDOW_SIZE - 1)/scale,
                                score])

    return boundingBox
#####################################################################
############################ MAIN PART ##############################

#Loop through all Testing Images given in FILELIST
for thisImg in open(FILELIST_PATH).readlines():
    name = 0
    #array to store all scales from original images
    scale_factors = [] #empty array to store all scaling factor of an image
    thisImgPath = thisImg.strip() #remove all possible white space
    img = Image.open(thisImgPath) #open images
    width, height = img.size #take width and height of image

    ThisImgName = ntpath.basename(thisImgPath)

    #finding larger dimension between width and height. Upscalling will be applied with respect to the this dimension
    min_dim = width if width <= height else height
    max_dim = width if width >= height else height

    max_scale = UPSCALLING_FACTOR / max_dim

    if max_scale == 1: #no scalling if the larger dimension are at the maximum size (which is 2975 for this project)
        scale_factors.append(1)
    elif max_scale > 1: #larger dimensions of original image is not at maximum size, then factors should be greater than 1
        scale_factors.append(max_scale)

    min_dim = min_dim * DOWN_SAMPLING_RATE
    rate_stack = 1 #number of division to the Down_sampling_rate

    #Generating all scaling factor with a constrains that the smaller dimension of feeding image cannot have size that is
    #smaller than set WINDOW_SIZE
    while min_dim >= WINDOW_SIZE:
        scale_factors.append(pow(DOWN_SAMPLING_RATE, rate_stack))
        min_dim = min_dim * DOWN_SAMPLING_RATE
        rate_stack += 1

    #to store all locations and score of all the detected bounding boxes.
    total_bboxes = []

    #loop through each scale_factor in the generated scales_factors
    for sf in scale_factors:
        # resize images by current scale_factor, first enlarge the larger size of image to 2975 then reduce it by (1/0.793700526)
        w, h = int(width * sf), int(height * sf)
        scale_img = img.resize((w, h))
        scale_img.save("Temp\\tmp.jpg") #save the scaled image to a temp files

        #####
        '''
        im2 = scale_img
        draw2 = ImageDraw.Draw(im2)
        draw2.rectangle((0,0,227,227), outline=colors[2])
        draw2.rectangle((1, 1, 226, 226), outline=colors[2])
        draw2.rectangle((2, 2, 225, 225), outline=colors[2])
        draw2.rectangle((3, 3, 224, 224), outline=colors[2])
        draw2.rectangle((4, 4, 223, 223), outline=colors[2])
        draw2.rectangle((5, 5, 222, 222), outline=colors[2])
        draw2.rectangle((6, 6, 221, 221), outline=colors[2])

        im2.save("Results\\Heatmaps\\{}.jpg".format(str(name+1)))'''
        #####

        #print 'size:', scale_img.size[0], scale_img.size[1]
        #Load the model structure (prototxt) files to a variable
        prototxt = open(MODEL_PATH, 'r')

        #since I cannot make it work with the command ".blobs['data'].reshape(1, 3, w, h)" in caffe package, I have to choose this dummy way and it
        #drastically decreases the speed of face detection

        #### dummy starts
        #Modifying line 5 and 6 in the network.prototxt files and save a copy version to the set path
        new_line = ""
        for i, line in enumerate(prototxt):
            if i == 5:
                new_line += "input_dim: " + str(scale_img.size[1]) + "\n"
            elif i == 6:
                new_line += "input_dim: " + str(scale_img.size[0]) + "\n"
            else:
                new_line += line

        output = open(MODIFIED_MODEL_SAVE_PATH, 'w')
        output.write(new_line)
        output.close()
        prototxt.close()

        #### dummy ends

        #Load fully-convolutional AlexNet network with all neurons' parameters
        net_full_conv = caffe.Net(MODIFIED_MODEL_SAVE_PATH,
                                  MODEL_PARAMETERS_PATH,
                                  caffe.TEST)

        # load input and configure preprocessing
        im = caffe.io.load_image("Temp\\tmp.jpg") #this method uses sliding window schema
        #net_full_conv.blobs['data'].reshape(1, 3, w, h) #this command should have done all the modification to the network.prototxt files

        #caffe procedure after loading a new neural network.
        transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(MEAN_FILE_PATH).mean(1).mean(1))
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_channel_swap('data', (2, 1, 0))  #RGB to BGR
        transformer.set_raw_scale('data', 255.0)

        #generate classification maps or feature maps including their locations (in regarding to the image) and scores
        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        '''
        #ploting heatmap if need
        name = name + 1

        fig = plt.figure()
        im2 = plt.imshow(out['prob'][0,1], cmap = plt.get_cmap('jet'))
        pos = fig.add_axes([0.93, 0.21, 0.02, 0.56])
        fig.colorbar(im2, cax=pos)
        plt.savefig("Results\\Heatmaps\\" + str(name))'''

        #bounding boxes around the predicted faces locations
        bboxes = makeBoundingBoxes(out['prob'][0, 1], sf)

        if (bboxes):
            total_bboxes.extend(bboxes)

    #the following scheme is : NMS_AVERAGE scheme. Perform greater than NMS_MAX scheme.
    all_bboxes = np.array(total_bboxes)
    preserved_bboxes = nms_max(all_bboxes, OVERLAPPING_THRESH)
    preserved_bboxes = nms_average(np.array(preserved_bboxes), CLUSTER_SCORE_THRESH)

    '''#the following scheme is : NMS_MAX scheme.
    all_bboxes = np.array(total_bboxes)
    preserved_bboxes = nms_max(all_bboxes, OVERLAPPING_THRESH)'''


    #write image name onto opened .txt files
    resultData.write(thisImgPath + "  \n")

    #using Draw method in Object ImageDraw
    draw = ImageDraw.Draw(img)

    '''###
    box_counter = 0
    temp = np.zeros(5)
    for box in preserved_bboxes:
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
        if box[4] >= temp[4]:
            temp = box
        box_counter += 1

    ttfont = ImageFont.truetype("arial.ttf", 20)
    draw.text((temp[0], temp[1]), "{0:.2f}".format(temp[4]), font=ttfont, fill=(255, 255, 102, 255))
    img.save("Results\\Images\\" + "1" + ".jpg")
    print (box_counter)'''
    ###

    for box in preserved_bboxes:
        
        #if box[4] > CONFIDENCE_THRESH: #used with NMS_MAX scheme

            #draw boxes with colour
            draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))

            #add score of the box to a conner
            ttfont = ImageFont.truetype("arial.ttf", 20)
            draw.text((box[0], box[1]), "{0:.2f}".format(box[4]), font=ttfont, fill=(255,255,102,255))

            #save image to a path within the working space
            img.save("Results\\Images\\" + ThisImgName)

            FACE_DETECTED_COUNTER += 1
            SUM_SCORE = SUM_SCORE + box[4]

            #write boxes information to .txt files
            resultData.write("<box> < top: %d  left: %d  width : %d  height: %d >\n\n" %(box[1], box[0], box[2], box[3]))

            # write boxes information to .xml files
            imgsChild = ET.SubElement(root, "images")
            imgChild = ET.SubElement(imgsChild, "image")
            imgChild.set('file', "Images\\" + ThisImgName)
            boxChild = ET.SubElement(imgChild, "box")
            boxChild.set('top', str(int(box[1])))
            boxChild.set('left', str(int(box[0])))
            boxChild.set('width', str(int(box[2] - box[0])))
            boxChild.set('height', str(int(box[3] - box[1])))

            part = ET.SubElement(boxChild, "part", field='empty')

    INPUT_IMG_COUNTER += 1
    #RunOne = 1 - comment out to be used when plotting heat map

#close result.txt
resultData.close()
tree = ET.ElementTree(root)


print("Total faces detected: ", FACE_DETECTED_COUNTER)
print("Total score: ", SUM_SCORE)
print("Average score: ", (SUM_SCORE / FACE_DETECTED_COUNTER))
print("Total Testing Images: ", INPUT_IMG_COUNTER)


tree.write("Results\\Logs\\resultXML.xml", pretty_print = True)

#####################################################################
#####################################################################
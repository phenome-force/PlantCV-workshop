# !/usr/bin/python
import sys, traceback
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    args = parser.parse_args()
    return args

### Main workflow
def main():
    # Get options
    args = options()
    pcv.params.debug = args.debug  # Set debug mode
    pcv.params.debug_outdir = args.outdir  # Set output directory
    
    # Read image
    img, path, filename = pcv.readimage(filename=args.image)
    
    ####################################################
    ### SEGMENTATION 
    # Saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=80, max_value=255, object_type='light')
    
    # Blue-yellow channel 
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=134, max_value=255, object_type='light')

    # Green-magenta
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    a_thresh = pcv.threshold.binary(gray_img=a, threshold=122, max_value=255, object_type='dark')
    
    # Combine 
    bs = pcv.logical_or(bin_img1=s_thresh, bin_img2=b_thresh)
    bsa = pcv.logical_or(bin_img1=bs, bin_img2=a_thresh)
    
    # Rough cleaning 
    bsa_fill1 = pcv.fill(bin_img=bsa, size=15) # Fill small noise
    bsa_fill2 = pcv.dilate(gray_img=bsa_fill1, ksize=3, i=3) # Dilate to join objects
    bsa_fill3 = pcv.fill(bin_img=bsa_fill2, size=250) # Fill large noise 
    
    # USE CUSTOM ROI to avoid soil while being aggressive with plant pixels kept 
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=bsa_fill3) # Identify contours 
    roi_custom, roi_hier_custom = pcv.roi.custom(img=img, vertices=[[1122,1560], [1385,1560], [1385,1685], [1890 , 1744], [1890 , 25], [600 , 25], [615 , 1744], [1085,1685]]) # Make a custom polygon ROI
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, roi_custom, roi_hier_custom, id_objects, obj_hierarchy, 'cutto') 
    
    # Clean up and fill any gaps within plant mask 
    filled_mask1 = pcv.fill(bin_img=kept_mask, size=350)
    filled_mask2 = pcv.closing(gray_img=filled_mask1)
    
    # Identify final plant mask and combine objects
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=filled_mask2)
    obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy)
    
    ####################################################
    ### OPTIONAL SHAPE AND COLOR ANALYSIS 
    #shape_img = pcv.analyze_object(img=img, obj=obj, mask=mask)
    #color_histogram = pcv.analyze_color(rgb_img=img, mask=mask, hist_plot_type='hsv')
    
    ####################################################
    ### MORPHOLOGY WORKFLOW PART
    pcv.params.text_size = 1.5
    pcv.params.text_thickness = 5
    pcv.params.line_thickness = 10
    skel = pcv.morphology.skeletonize(mask=mask)
    # Prune back barbs off the skeleton
    pruned, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skel, size=30, mask=mask)
    pruned, segmented_img, segment_objects = pcv.morphology.prune(skel_img=pruned, size=3, mask=mask)
    # Sort segments into leaf and stem 
    leaf_objects, other_objects = pcv.morphology.segment_sort(skel_img=pruned, objects=segment_objects, mask=mask)
    # Identify segments     
	segmented_img, labeled_id_img = pcv.morphology.segment_id(skel_img=pruned, objects=leaf_objects, mask=mask)
    # Extract relative leaf angles 
	labeled_angle_img = pcv.morphology.segment_insertion_angle(skel_img=pruned, segmented_img=segmented_img, leaf_objects=leaf_objects, stem_objects=other_objects, size=22)
	
	####################################################
    ### OPTIONAL ALTERNATIVE MEASUREMENTS  
    ## Fill in leaf objects and collect area information 
    #filled_img = pcv.morphology.fill_segments(mask=mask, objects=leaf_objects)
    ## Measure path lengths of segments     
    #labeled_img2 = pcv.morphology.segment_path_length(segmented_img=segmented_img, objects=leaf_objects)
	## Measure leaf euclidean lengths (will overwrite internodes)
    #labeled_img3 = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_objects)
    ## Measure curvature of segments      
    #labeled_img4 = pcv.morphology.segment_curvature(segmented_img=segmented_img, objects=leaf_objects)
    ## Measure the angle of segments      
    #labeled_img5 = pcv.morphology.segment_angle(segmented_img=segmented_img, objects=leaf_objects)
    ## Measure the tangent angles of segments      
    #labeled_img6 = pcv.morphology.segment_tangent_angle(segmented_img=segmented_img, objects=leaf_objects, size=25)
    ## Measure stem characteristics 
    #stem_img = pcv.morphology.analyze_stem(rgb_img=img, stem_objects=other_objects)

    ####################################################
    ### WRITE DATA
    pcv.print_results(filename=args.result)

if __name__ == '__main__':
    main()

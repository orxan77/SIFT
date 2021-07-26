#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;

const char* keys =
"{ help h |                  | Print help message. }"
"{ input1 | dataset4/obj3.png | Path to input image 1. }"
"{ input2 | dataset4/scene2.png | Path to input image 2. }";

Mat img_obj_colored, img_scene_colored;
Mat img_object, img_scene;

std::vector<KeyPoint> keypoints_object, keypoints_scene;
Mat descriptors_object, descriptors_scene;

/**
 Function to show the keyppoints on the color image found  by SIFT algorithm
 */
void showKeypoints(){
    Mat res;
    drawKeypoints(img_obj_colored, keypoints_object, res, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("Keypoints", res);
//    imwrite("/Users/orkhanbayramli/Desktop/res.png", res);
}

int main( int argc, char* argv[] )
{
    //-- Step 0: Load images as Mat arrays and convert their colorspace to grayscale
    CommandLineParser parser( argc, argv, keys );

    img_obj_colored = imread( samples::findFile( parser.get<String>("input1")));
    img_scene_colored = imread( samples::findFile( parser.get<String>("input2")));

    /// Convert to grayscale
    cvtColor(img_obj_colored, img_object, COLOR_BGR2GRAY);
    cvtColor(img_scene_colored, img_scene, COLOR_BGR2GRAY);

    /// Sanity check
    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //--Step 0: end
    
    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    Ptr<SIFT> detector = SIFT::create(0, 3, 0.015, 10, 1.6);

    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    showKeypoints();
    //-- Step 1: end
    
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 3 );
    //-- Step 2: end
    
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        //we create an image by using the good matches[i].queryIdx which is the index of good keypoint.
        //Then retrieve this keypoint by querying it from the keypoints vector. It will give us
        //the keypoint itself then we retrieve the (x,y) location of that keypoint by using the pt attribute
        //of that object. Lastly, the point itself is stored in the obj vector in the form of Point2f.
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    
    // We have the (x,y) locations of obj and scene keypoints stored in obj and scene, respectively.
    // We get the Homography matrix which will help us to detect the coordinations of object in the
    // scenery.
    Mat H = findHomography(obj, scene, RANSAC);
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    // get the locations of obj corners in the scene by using homography. the corner locations
    // are stored in the scene_corners vector.
    perspectiveTransform( obj_corners, scene_corners, H);
    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
         scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
         scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
         scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
         scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches );
    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif

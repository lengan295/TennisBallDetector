#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void xuly(Point&);

int main()
{
	VideoCapture capture(0); // open the default camera
    
	if( !capture.isOpened() )
    {
            //printf( "ERROR: capture is NULL \n" );
            getchar();
            return -1;
    }

	Mat image, hsv_frame, thresholded;

	int t1min=32, t1max=45, t2min=0, t2max=255, t3min=0, t3max=255;

	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 3;
	Mat element = getStructuringElement( erosion_type,
                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                    Point( erosion_size, erosion_size ) );

	while(image.empty())
		capture >> image; 

      
	while(1)
	{
		capture >> image;
		
		Scalar hsv_min = cvScalar(t1min, t2min, t3min, 0);
		Scalar hsv_max = cvScalar(t1max, t2max ,t3max, 0);

		// Covert color space to HSV as it is much easier to filter colors in the HSV color-space.
        cvtColor(image, hsv_frame, CV_BGR2GRAY);

		//////////////////////////////////////////////////////////////
		// Filter out colors which are out of range.
		inRange(hsv_frame, hsv_min, hsv_max, thresholded);
		
		/////// @dilation and erosion //////////
		erode( thresholded, thresholded, element );
		dilate( thresholded, thresholded, element );

		/// Blur
		GaussianBlur( thresholded, thresholded, Size(9, 9), 2, 2 );

		vector<Vec3f> circles;

		/// Apply the Hough Transform to find the circles
		HoughCircles( thresholded, circles, CV_HOUGH_GRADIENT, 2, thresholded.rows/8, 200, 100, 0, 0 );
		
		int c = circles.size();
		if(c>0)
		{
			Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));		
			xuly(center);
		}

		if( waitKey(30) >= 0 ) break;
	}
	return 0;
}

void xuly(Point& center)
{
}
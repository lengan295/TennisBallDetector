#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

VideoCapture capture(0); // open the default camera
Mat image, hsv_frame, thresholded;

int t1min=32, t1max=45, t2min=0, t2max=255, t3min=0, t3max=255;
int interval = 10;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
		 Point3_<uchar>* p = hsv_frame.ptr<Point3_<uchar> >(y,x);
         printf("%d, %d, %d\n",p->z,p->y,p->x);

		 t1min = p->z - interval/2;
		 t1max = p->z + interval/2;
		 /*t2min = p->y - 3*interval;
		 t3min = p->x - 3*interval;

		 
		 t2max = p->y + 3*interval;
		 t3max = p->x + 3*interval;*/
     }
}



void main()
{
	/////////////Start the camera///////////////////
	if( !capture.isOpened() )
    {
            printf( "ERROR: capture is NULL \n" );
            getchar();
            return;
    }
	
	while(image.empty())
		capture >> image; 

	/////////////end - Start the camera///////////////////


	////////////Initate the windows///////////////////

	//char TrackbarName1[50]="t1min";
    //char TrackbarName2[50]="t1max";

	cvNamedWindow( "Original", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "HSV", CV_WINDOW_AUTOSIZE );
	cvNamedWindow("After Color Filtering",CV_WINDOW_AUTOSIZE);

	setMouseCallback("Original", CallBackFunc, NULL);
	cvCreateTrackbar( "Interval", "Original", &interval, 250 , NULL );

	// Create Trackbars	
    //cvCreateTrackbar( TrackbarName1, "After Color Filtering", &t1min, 260 , NULL );
    //cvCreateTrackbar( TrackbarName2, "After Color Filtering", &t1max, 260,  NULL  );
    
	////////////end - Initate the windows///////////////////

	///////////Init variables for erosion and dilation//////

	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 3;

	Mat element = getStructuringElement( erosion_type,
                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                    Point( erosion_size, erosion_size ) );

	////////end - Init variables for erosion and dilation//////

	////////the main loop/////////

	while(1)
	{
		capture >> image;
		
		Scalar hsv_min = cvScalar(t1min, t2min, t3min, 0);
		Scalar hsv_max = cvScalar(t1max, t2max ,t3max, 0);

		// Covert color space to HSV as it is much easier to filter colors in the HSV color-space.
        cvtColor(image, hsv_frame, CV_BGR2HSV);

		//////////////////////////////////////////////////////////////
		// Filter out colors which are out of range.
		inRange(hsv_frame, hsv_min, hsv_max, thresholded);
		
		
		/*////// @dilation and erosion //////////
		erode( thresholded, thresholded, element );
		dilate( thresholded, thresholded, element );

		/// Blur
		GaussianBlur( thresholded, thresholded, Size(9, 9), 2, 2 );

		vector<Vec3f> circles;

		/// Apply the Hough Transform to find the circles
		HoughCircles( thresholded, circles, CV_HOUGH_GRADIENT, 2, thresholded.rows/8, 200, 100, 0, 0 );
		
		int c = circles.size();
		//printf("%d\n",circles.size());
		if(c>0)
		{
			Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
			int radius = cvRound(circles[0][2]);
			// circle outline
			circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}

		/*
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// circle outline
			circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}*/		
           
        imshow( "Original", image ); // Original stream with detected ball overlay
		imshow( "HSV", hsv_frame ); 
        imshow( "After Color Filtering", thresholded ); // The stream after color filtering
         
		if( waitKey(30) >= 0 ) break;
	}
}
// TennisBallDetector_full.cpp : Defines the entry point for the console application.
//
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void detect_mat();
void detect();

int main2()
{
	detect_mat();
	return 0;
}

void detect_mat()
{
	VideoCapture capture(0); // open the default camera
    
	if( !capture.isOpened() )
    {
            printf( "ERROR: capture is NULL \n" );
            getchar();
            return;
    }

	Mat image, hsv_frame, thresholded;

	int t1min=32, t1max=45, t2min=0, t2max=255, t3min=0, t3max=255;

	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 3;
	Mat element = getStructuringElement( erosion_type,
                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                    Point( erosion_size, erosion_size ) );

	char TrackbarName1[50]="t1min";
    char TrackbarName2[50]="t1max";

	while(image.empty())
		capture >> image; 

	cvNamedWindow( "Original", CV_WINDOW_AUTOSIZE );
	cvNamedWindow("After Color Filtering",CV_WINDOW_AUTOSIZE);

	// Create Trackbars	
    cvCreateTrackbar( TrackbarName1, "After Color Filtering", &t1min, 260 , NULL );
    cvCreateTrackbar( TrackbarName2, "After Color Filtering", &t1max, 260,  NULL  );
      
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
		
		/////// @dilation and erosion //////////
		erode( thresholded, thresholded, element );
		dilate( thresholded, thresholded, element );

		/// Blur
		GaussianBlur( thresholded, thresholded, Size(9, 9), 2, 2 );

		vector<Vec3f> circles;

		/// Apply the Hough Transform to find the circles
		HoughCircles( thresholded, circles, CV_HOUGH_GRADIENT, 2, thresholded.rows/8, 200, 100, 0, 0 );
		
		int c = circles.size();
		printf("%d\n",circles.size());
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
        imshow( "After Color Filtering", thresholded ); // The stream after color filtering
         
		if( waitKey(30) >= 0 ) break;
	}
}

void detect()
{
	//IplImage *image = cvLoadImage("ball.jpg");
	CvCapture* capture = cvCaptureFromCAM( 0 );
    
    if( !capture )
    {
            printf( "ERROR: capture is NULL \n" );
            getchar();
            return;
    }

	IplImage* image = cvQueryFrame( capture );;
	
	while(!image)
		image = cvQueryFrame( capture );

	int t1min=32, t1max=45, t2min=0, t2max=255, t3min=0, t3max=255;
	// get the image data
     int height    = image->height;
     int width     = image->width;
     int step      = image->widthStep;
      
     // capture size - 
    CvSize size = cvSize(width,height);
    
    // Initialize different images that are going to be used in the program
    IplImage*  hsv_frame    = cvCreateImage(size, IPL_DEPTH_8U, 3); // image converted to HSV plane
	IplImage*  thresholded   = cvCreateImage(size, IPL_DEPTH_8U, 1); // final thresholded image

	cvNamedWindow( "Original", CV_WINDOW_AUTOSIZE );
	cvNamedWindow("After Color Filtering",CV_WINDOW_AUTOSIZE);

	// Create Trackbars
    char TrackbarName1[50]="t1min";
    char TrackbarName2[50]="t1max";
	
    cvCreateTrackbar( TrackbarName1, "After Color Filtering", &t1min, 260 , NULL );
    cvCreateTrackbar( TrackbarName2, "After Color Filtering", &t1max, 260,  NULL  );
      
	while(1)
	{
		image = cvQueryFrame( capture );
		CvScalar hsv_min = cvScalar(t1min, t2min, t3min, 0);
		CvScalar hsv_max = cvScalar(t1max, t2max ,t3max, 0);

		// Covert color space to HSV as it is much easier to filter colors in the HSV color-space.
        cvCvtColor(image, hsv_frame, CV_BGR2HSV);

		// Filter out colors which are out of range.
		cvInRangeS(hsv_frame, hsv_min, hsv_max, thresholded);
		//fastNlMeansDenoising(thresholded,thresholded);
		
		//Split image into its 3 one dimensional images
        //cvSplit( hsv_frame,thresholded1, thresholded2, thresholded3, NULL );

		// Memory for hough circles
        CvMemStorage* storage = cvCreateMemStorage(0);
        
        // hough detector works better with some smoothing of the image
        cvSmooth( thresholded, thresholded, CV_GAUSSIAN, 9, 9 );
        
        //hough transform to detect circle
        CvSeq* circles = cvHoughCircles(thresholded, storage, CV_HOUGH_GRADIENT, 2,
                                        thresholded->height/4, 50, 180, 5, 0);
		int n = circles->total;
		printf("%d \n",n);
		if(n>0)
		{
			float* p = (float*)cvGetSeqElem( circles, 0 );
            //printf("Ball! x=%f y=%f r=%f\n\r",p[0],p[1],p[2] );
           
            // draw a circle with the centre and the radius obtained from the hough transform
            cvCircle( image, cvPoint(cvRound(p[0]),cvRound(p[1])),  //plot centre
                                    2, CV_RGB(255,255,255),-1, 8, 0 );
            cvCircle( image, cvPoint(cvRound(p[0]),cvRound(p[1])),  //plot circle
                                    cvRound(p[2]), CV_RGB(0,255,0), 2, 8, 0 );
		}
		/*
		for (int i = 0; i < (n>5?5:n) ; i++)
        {   //get the parameters of circles detected
            float* p = (float*)cvGetSeqElem( circles, i );
            //printf("Ball! x=%f y=%f r=%f\n\r",p[0],p[1],p[2] );
           
            // draw a circle with the centre and the radius obtained from the hough transform
            cvCircle( image, cvPoint(cvRound(p[0]),cvRound(p[1])),  //plot centre
                                    2, CV_RGB(255,255,255),-1, 8, 0 );
            cvCircle( image, cvPoint(cvRound(p[0]),cvRound(p[1])),  //plot circle
                                    cvRound(p[2]), CV_RGB(0,255,0), 2, 8, 0 );
        }
		*/
           
         
           
         cvShowImage( "Original", image ); // Original stream with detected ball overlay
         cvShowImage( "After Color Filtering", thresholded ); // The stream after color filtering
         
        //cvReleaseMemStorage(&storage);
		if( (cvWaitKey(10) & 255) == 27 ) break;
	}
	//cvWaitKey(0);
	
	cvReleaseCapture( &capture );
}
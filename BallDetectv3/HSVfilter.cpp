
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//IplImage *image = cvLoadImage("ball.jpg");
	CvCapture* capture = cvCaptureFromCAM( 0 );
    
    if( !capture )
    {
            printf( "ERROR: capture is NULL \n" );
            getchar();
            return -1;
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
    return 0;
}
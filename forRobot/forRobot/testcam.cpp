#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <time.h>
using namespace cv;

int main2()
{
        VideoCapture capture(0); // open the default camera

        if( !capture.isOpened() )
		{
            printf( "ERROR: capture is NULL \n" );
            getchar();
            return -1;
		}

        Mat image; //, hsv_frame, thresholded;

        int c = 0;
        while(image.empty() && c<10000)
        {
                c++;
                capture >> image;
        }
        if(c>=10000)
        {
                printf("Khong doc duoc anh \n");
                getchar();
				return -1;
        }
        else
		{
				printf("Co ve OK :) \n");
				//getchar();
		}

		int i=1;
		char name[8] = "";
		char path[20] = "test/";
		time_t t0, t1, t2;
		time(&t0);
		time(&t1);
		time(&t2);
		while(difftime(t2,t0)<60)
		{
			time(&t2);
			if(difftime(t2,t1)>3)
			{
				time(&t1);
				sprintf(name,"%d.jpg",i);
				//strcat(name,".jpg");
				//strcat(path,name);
				capture >> image;
				printf("%d -> %s\n",i,name);
				i++;
				imwrite(name,image);
			}
		}
}



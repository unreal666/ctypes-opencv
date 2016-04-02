ctypes-opencv is a package that brings Willow Garage's (formerly Intel's) Open Source Computer Vision Library (OpenCV) to Python. OpenCV is a collection of algorithms and sample code for various computer vision problems. The goal of ctypes-opencv is to provide Python access to all documented functionality of OpenCV.

## Advantages ##

  * Complete interface to OpenCV's CXCORE, CV, HighGUI components.
  * Pythonic interface. OpenCV's objects are safely deleted when not used. No need to invoke cvRelease...().
  * Pure Python package. Neither C/C++ compiler nor OpenCV's source code is needed.
  * Support for both versions 1.0 and 1.1pre1 of OpenCV.
  * Cross platform, running on any OS that OpenCV can be installed, including: Windows, Linux, and Mac OS X.
  * Ability to convert between OpenCV's arrays (CvMatND, CvMat, and IplImage) and arrays used in wxWidgets, NumPy, PyGTK, and PIL.

## Disadvantage ##

  * No support for OpenCV's ML component yet. This is a limitation of the current ctypes package. The ML component contains mostly C++ classes. ctypes at the moment cannot wrap C++ classes and their member functions. This limitation will be overcome in the future. [A new Python wrapper using Boost.Python and NumPy](http://code.google.com/p/pyopencv/) has been under development.

## Example ##

### Python code for demonstrating K-Means Clustering ###
```
from ctypes_opencv import *
MAX_CLUSTERS=5

if __name__ == "__main__":

    color_tab = [CV_RGB(255,0,0),CV_RGB(0,255,0),CV_RGB(100,100,255), CV_RGB(255,0,255),CV_RGB(255,255,0)]
    img = cvCreateImage(cvSize(500, 500), 8, 3)
    rng = cvRNG(-1)
    cvNamedWindow( "clusters", 1 )
        
    while True:
        cluster_count = cvRandInt(rng)%(MAX_CLUSTERS-1) + 2
        sample_count = cvRandInt(rng)%999 + 1
        points = cvCreateMat(sample_count, 1, CV_32FC2)
        clusters = cvCreateMat(sample_count, 1, CV_32SC1)
        
        # generate random sample from multigaussian distribution
        for k in range(cluster_count):
            first = k*sample_count/cluster_count
            last = (k+1)*sample_count/cluster_count if k != cluster_count else sample_count
            if first < last:
                cvRandArr(rng, cvGetRows(points, None, first, last), CV_RAND_NORMAL,
                    cvScalar(cvRandInt(rng)%img.width,cvRandInt(rng)%img.height), cvScalar(img.width*0.1,img.height*0.1))
        cvRandShuffle( points, rng )
        
        # K Means Clustering
        cvKMeans2(points, cluster_count, clusters, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0))

        cvZero( img )
        for i in range(sample_count):
            pt = points[i,0]
            cvCircle(img, cvPoint(cvRound(pt[0]), cvRound(pt[1])), 2, color_tab[clusters[i,0]], CV_FILLED, CV_AA, 0)
        
        cvShowImage( "clusters", img )

        if '%c' % (cvWaitKey(0) & 255) in ['\x1b','q','Q']: # 'ESC'
            break
```

### Equivalent C code for demonstrating K-Means Clustering ###
```
#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

int main( int argc, char** argv )
{
    #define MAX_CLUSTERS 5
    CvScalar color_tab[MAX_CLUSTERS];
    IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
    CvRNG rng = cvRNG(-1);
    CvPoint ipt;

    color_tab[0] = CV_RGB(255,0,0);
    color_tab[1] = CV_RGB(0,255,0);
    color_tab[2] = CV_RGB(100,100,255);
    color_tab[3] = CV_RGB(255,0,255);
    color_tab[4] = CV_RGB(255,255,0);

    cvNamedWindow( "clusters", 1 );
        
    for(;;)
    {
        char key;
        int k, cluster_count = cvRandInt(&rng)%MAX_CLUSTERS + 1;
        int i, sample_count = cvRandInt(&rng)%1000 + 1;
        CvMat* points = cvCreateMat( sample_count, 1, CV_32FC2 );
        CvMat* clusters = cvCreateMat( sample_count, 1, CV_32SC1 );
        
        /* generate random sample from multigaussian distribution */
        for( k = 0; k < cluster_count; k++ )
        {
            CvPoint center;
            CvMat point_chunk;
            center.x = cvRandInt(&rng)%img->width;
            center.y = cvRandInt(&rng)%img->height;
            cvGetRows( points, &point_chunk, k*sample_count/cluster_count,
                       k == cluster_count - 1 ? sample_count :
                       (k+1)*sample_count/cluster_count, 1 );
                        
            cvRandArr( &rng, &point_chunk, CV_RAND_NORMAL,

                       cvScalar(center.x,center.y,0,0),
                       cvScalar(img->width*0.1,img->height*0.1,0,0));
        }

        /* shuffle samples */
        for( i = 0; i < sample_count/2; i++ )
        {
            CvPoint2D32f* pt1 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng)%sample_count;
            CvPoint2D32f* pt2 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng)%sample_count;
            CvPoint2D32f temp;
            CV_SWAP( *pt1, *pt2, temp );
        }

        cvKMeans2( points, cluster_count, clusters,
                   cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));

        cvZero( img );

        for( i = 0; i < sample_count; i++ )
        {
            int cluster_idx = clusters->data.i[i];
            ipt.x = (int)points->data.fl[i*2];
            ipt.y = (int)points->data.fl[i*2+1];
            cvCircle( img, ipt, 2, color_tab[cluster_idx], CV_FILLED, CV_AA, 0 );
        }

        cvReleaseMat( &points );
        cvReleaseMat( &clusters );

        cvShowImage( "clusters", img );

        key = (char) cvWaitKey(0);
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    
    cvDestroyWindow( "clusters" );
    cvReleaseImage(&img);
    return 0;
}
```

## Related Python wrappers for OpenCV ##

OpenCV has its own swig-based Python wrapper. However, it has conflicts in memory management between C/C++ and Python, and hence is not suitable for large projects. It is also particularly hard to maintain and develop.

Another project called CVtypes was pioneered by Michael Otto and is currently maintained by Gary Bishop (at http://wwwx.cs.unc.edu/~gb/wp/blog/2007/02/04/python-opencv-wrapper-using-ctypes/). The wrapper is based on ctypes. It supports a large set of OpenCV's functions and a limited set of OpenCV's structures.

I used to provide some improvements to CVtypes here and there. While Gary Bishop was a kind professor, I felt not so nice to keep asking him to update his code. Therefore, I decided to branch from his CVtypes, and the result is this project. ctypes-opencv supports a fairly complete set of OpenCV's structures and functions. More importantly, I have put a lot of efforts in making ctypes-opencv faster, better memory-managed, and easier to use, by not only adopting but also improving the pythonic interface introduced by OpenCV's developers. Nevertheless, credits should also go to OpenCV developers, and CVtypes' authors and contributors. I intend to eventually merge back to Gary Bishop's CVtypes when the project is mature enough.


## Bugs and Commentary ##

> Please send information on issues of usage to Minh-Tri Pham <pmtri80@gmail.com>, post a message to [ctypes-opencv's discussion group](http://groups.google.com/group/ctypes-opencv), or create an issue in the Issues pannel.
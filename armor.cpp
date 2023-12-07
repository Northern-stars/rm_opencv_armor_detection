#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;


int hmin = 18, hmax = 67, smin = 0, smax = 99, vmin = 240, vmax = 255;
int kernelsize = 14, gaussx = 5, gaussy = 6, gausscoresize = 4, cannyx = 164, cannyy = 164, Corpointmax=8, Corpointmin=4, arealimit=4,hw=3,light=235;//red
/*int hmin=65, hmax=120,smin=20,smax=255,vmin=85,vmax=255;
int kernelsize=14,gaussx=5,gaussy=4,gausscoresize=4,cannyx=164,cannyy=164,Corpointmax=8,Corpointmin=4,arealimit=4,hw=3,light=235; //blue*/



vector<Point2f> calculate(Point2f pu, Point2f pd)
{
	Point2f midPoint = (pu + pd) / 2;
	Point2f delta = midPoint - pu;
	delta =delta* (double)(125 / 55);
	return { midPoint - delta,midPoint + delta };
}
void getContours(Mat img, Mat imgOri, Mat imgout)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect>boundbox(contours.size());
	vector<RotatedRect> rotatedrectbox(contours.size());
	vector<int> angles(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		float peri = arcLength(contours[i], true);
		if (area > arealimit*50) { approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
		}
		int objCor = (int)conPoly[i].size();
		if (objCor<=Corpointmax && objCor>=Corpointmin){
			boundbox[i] = boundingRect(conPoly[i]);
			if (boundbox[i].height / boundbox[i].width > hw/2) {
				rotatedrectbox[i] = minAreaRect(conPoly[i]);
				angles[i] = rotatedrectbox[i].angle;
				rectangle(imgout, boundbox[i].tl(), boundbox[i].br(), Scalar(255, 0, 255), 2);
			}
		}
	}
	vector<int> flag(rotatedrectbox.size(),0);
	for (int i = 0; i < rotatedrectbox.size(); i++)
	{
		int min = i;
		for (int j = i; j < rotatedrectbox.size(); j++) 
		{
			if (rotatedrectbox[j].center.x < rotatedrectbox[min].center.x) { min = j; }
		}
		RotatedRect tre = rotatedrectbox[min];
		rotatedrectbox[min] = rotatedrectbox[i];
		rotatedrectbox[i] = tre;
	}
	int pointer = 0;
	vector<Point2i> resul(rotatedrectbox.size());
	for (int i = 0; i < rotatedrectbox.size(); i++) 
	{
		if (flag[i] == 1 ||(rotatedrectbox[i].center.x==0 and rotatedrectbox[i].center.y==0)) { continue; }
		for (int j = 0;j<rotatedrectbox.size();j++)
		{
			if (flag[j] == 1 || i==j|| (rotatedrectbox[j].center.x == 0 and rotatedrectbox[j].center.y == 0)) { continue; }
			int deltaangle = rotatedrectbox[i].angle - rotatedrectbox[j].angle;
			int deltay = abs(rotatedrectbox[i].center.y - rotatedrectbox[j].center.y);
			if (-10 < deltaangle && deltaangle < 10 &&deltay<150)
			{
				flag[i] = 1;
				flag[j] = 1;
				resul[pointer].x = i;
				resul[pointer].y = j;
				pointer+=1;
				break;
			}
		}
	}
	if (pointer == 0) { cout << "No result" << endl; }
	else 
	{
		for (int i = 0; i != pointer; i++)
		{
			//cout << rotatedrectbox[resul[i].x].center << "and" << rotatedrectbox[resul[i].y].center<<endl;
			//Point deltapoint;
			//deltapoint.x = rotatedrectbox[resul[i].x].center.x - rotatedrectbox[resul[i].y].center.x;
			//deltapoint.y = rotatedrectbox[resul[i].x].center.y - rotatedrectbox[resul[i].y].center.y;
			Point2f cornerpointi[4]; Point2f cornerpointj[4];
			if (rotatedrectbox[resul[i].x].size.width > rotatedrectbox[resul[i].x].size.height) 
			{
				rotatedrectbox[resul[i].x].points(cornerpointi);
				Point2f transfer[4];
				transfer[0] = cornerpointi[3];
				for (int j = 0; j < 3; j++)
				{
					transfer[j+1] = cornerpointi[j];
				}
				for (int j = 0; j < 4; j++)
				{
					cornerpointi[j] = transfer[j];
				}

			}
			else { rotatedrectbox[resul[i].x].points(cornerpointi); }
			/*for (int j = 0; j < 4; j++)
			{
				cout << "这是第" << j << "坐标为" << cornerpointi[j] << endl;
			}
			cout << rotatedrectbox[resul[i].x].size.width << "i宽" << endl;
			cout << rotatedrectbox[resul[i].x].size.height << "i高" << endl;*/
			if (rotatedrectbox[resul[i].y].size.width > rotatedrectbox[resul[i].y].size.height)
			{
				rotatedrectbox[resul[i].y].points(cornerpointj);
				Point2f transfer[4];
				transfer[0] = cornerpointj[3];
				for (int j = 0; j < 3; j++)
				{
					transfer[j+1] = cornerpointj[j];
				}
				for (int j = 0; j < 4; j++)
				{
					cornerpointj[j] = transfer[j];
				}
			}
			else { rotatedrectbox[resul[i].y].points(cornerpointj); }
			/*for (int j = 0; j < 4; j++)
			{
				cout << "这是第" << j << "坐标为" << cornerpointj[j] << endl;
			}
			cout << rotatedrectbox[resul[i].y].size.width << "j宽" << endl;
			cout << rotatedrectbox[resul[i].y].size.height << "j高" << endl;*/
			vector<Point2f> Leftside = calculate(cornerpointi[1], cornerpointi[0]);
			vector<Point2f> Rightside = calculate(cornerpointj[2], cornerpointj[3]);
			Point2f final[4];
			final[0] = Leftside[0]; final[1] = Rightside[0]; final[2] = Leftside[1]; final[3] = Rightside[1];
			Point2f target[4] = {{0,0},{270,0},{0,250},{270,250}};
			Mat imgWarp;
			Mat matrix = getPerspectiveTransform(final, target);
			warpPerspective(imgOri, imgWarp, matrix, Point(270, 250));
			imshow("Warp", imgWarp);
			//cout << "//" << endl;
			//waitKey(1000);
		}
	}
}

int main()
{
	string path = "C:\\Users\\north\\Desktop\\learning materials and works\\Data\\1.jpg";
	//VideoCapture cap(path); //For video
	//Mat img; //For video
	Mat img = imread(path); //For img
	//while (true) {
		//cap.read(img); //For video
		Mat imgHSV, mask, imgErode, imgGauss, imgDil, imgCanny, imgRec, imgGray, imgB;
		cvtColor(img, imgHSV, COLOR_BGR2HSV);
		cvtColor(img, imgGray, COLOR_BGR2GRAY);
		//imshow("Gray", imgGray);
		/*namedWindow("trackbar", (640, 400));
		createTrackbar("Hmin", "trackbar", &hmin, 359);
		createTrackbar("Hmax", "trackbar", &hmax, 359);
		createTrackbar("Smin", "trackbar", &smin, 255);
		createTrackbar("Smax", "trackbar", &smax, 255);
		createTrackbar("Vmin", "trackbar", &vmin, 255);
		createTrackbar("Vmax", "trackbar", &vmax, 255);
		createTrackbar("kernelsize", "trackbar", &kernelsize, 100);
		createTrackbar("gaussx", "trackbar", &gaussx, 200);
		createTrackbar("gaussy", "trackbar", &gaussy, 200);
		createTrackbar("gausscoresize", "trackbar", &gausscoresize, 99);
		createTrackbar("cannyx", "trackbar", &cannyx, 500);
		createTrackbar("cannyy", "trackbar", &cannyy, 500);
		createTrackbar("Corpointmin", "trackbar", &Corpointmin, 30);
		createTrackbar("Corpointmax", "trackbar", &Corpointmax, 30);
		createTrackbar("Area", "trackbar", &arealimit, 10000);
		createTrackbar("HW", "trackbar", &hw, 1000);
		createTrackbar("light", "trackbar", &light, 255);*/
		//while (true)  //For img
		//{ //For img
		//imshow("Image", img);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelsize, kernelsize));
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);
		//threshold(imgGray, imgB, light, 255, THRESH_BINARY); //For gray
		//imshow("Binary", imgB);  //For gray
		inRange(imgHSV, lower, upper, mask);  
		//imshow("Mask", mask);
		GaussianBlur(mask, imgGauss, Size(2 * gausscoresize + 1, 2 * gausscoresize + 1), gaussx, gaussy); //For HSV
		//GaussianBlur(imgB, imgGauss, Size(2 * gausscoresize + 1, 2 * gausscoresize + 1), gaussx, gaussy); // For Gray
		Canny(imgGauss, imgCanny, cannyx, cannyy);
		//imshow("Canny", imgCanny);
		dilate(imgCanny, imgDil, kernel);
		erode(imgDil, imgErode, kernel);
		//imshow("ImgDil", imgDil);
		//imshow("ImgErode", imgErode);
		imgRec = img.clone();
		getContours(imgErode, img, imgRec);
		imshow("ImgRec", imgRec);
		//cout << "//" << endl;
		//waitKey(1); //For video
		waitKey(0); //For img
	//} // For video
	//} //For img
	return 0;
}
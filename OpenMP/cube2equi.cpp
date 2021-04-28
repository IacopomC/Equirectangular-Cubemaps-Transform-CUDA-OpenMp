#include <opencv2/opencv.hpp>

using namespace std;

void cube2equi(cv::Mat_<cv::Vec3b>& posy, cv::Mat_<cv::Vec3b>& negx, cv::Mat_<cv::Vec3b>& posx, cv::Mat_<cv::Vec3b>& negz, cv::Mat_<cv::Vec3b>& negy, cv::Mat_<cv::Vec3b>& posz, cv::Mat_<cv::Vec3b>& dst)
{
	float PI = 3.14159265358979323846;

	float dims = negx.size().width;

	float xPixel = 0;
	float yPixel = 0;
	float yTemp = 0;
	float imageSelect = 0;
	float outputWidth = dims * 4;
	float outputHeight = dims * 2;

	float increment = (dims * 2) / 100;
	int	counter = 0;
	float percentCounter = 0;

	float v;
	float phi;
	float u;
	float theta;
	float x;
	float y;
	float z;
	float a;

	float xx;
	float yy;
	float zz;

	dst.create(outputHeight, outputWidth);
	dst = cv::Vec3b(0, 0, 0);

	for (int j = 0; j < outputHeight; j++)
	{
		if (counter <= j)
		{
			percentCounter += 1;
			counter += increment;
		}

		v = 1.0 - ((float(j)) / (dims * 2));
		phi = v * PI;

		for (int i = 0; i < (int)outputWidth; i++)
		{

			u = (float(i)) / (dims * 4);
			theta = u * 2 * PI;

			// all of these range between 0 and 1
			x = cos(theta) * sin(phi);
			y = sin(theta) * sin(phi);
			z = cos(phi);

			a = max(max(abs(x), abs(y)), abs(z));

			// one of these will equal either - 1 or +1
			xx = x / a;
			yy = y / a;
			zz = z / a;

			// format is left, front, right, back, bottom, top;
			// therefore negx, posz, posx, negz, negy, posy

			// square 1 left
			if (yy == -1)
			{ 
				xPixel = int(((-1.0 * tan(atan(x / y)) + 1.0) / 2.0) * dims);
				yTemp = int(((-1.0 * tan(atan(z / y)) + 1.0) / 2.0) * (dims - 1.0));
				imageSelect = 1;
			}
			// square 2; front
			else if(xx == 1)
			{
				xPixel = int(((tan(atan(y / x)) + 1.0) / 2.0) * dims);
				yTemp = int(((tan(atan(z / x)) + 1.0) / 2.0) * dims);
				imageSelect = 2;
			}
			// square 3; right
			else if(yy == 1) 
			{
				xPixel = int(((-1 * tan(atan(x / y)) + 1.0) / 2.0) * dims);
				yTemp = int(((tan(atan(z / y)) + 1.0) / 2.0) * (dims - 1));
				imageSelect = 3;
			}
			// square 4; back
			else if (xx == -1) {
				xPixel = int(((tan(atan(y / x)) + 1.0) / 2.0) * dims);
				yTemp = int(((-1 * tan(atan(z / x)) + 1.0) / 2.0) * (dims - 1));
				imageSelect = 4;
			}
			// square 5; bottom
			else if (zz == 1)
			{
				xPixel = int(((tan(atan(y / z)) + 1.0) / 2.0) * dims);
				yTemp = int(((-1 * tan(atan(x / z)) + 1.0) / 2.0) * (dims - 1));
				imageSelect = 5;
			}
			// square 6; top
			else if (zz == -1)
			{
				xPixel = int(((-1 * tan(atan(y / z)) + 1.0) / 2.0) * dims);
				yTemp = int(((-1 * tan(atan(x / z)) + 1.0) / 2.0) * (dims - 1));
				imageSelect = 6;
			}

			yPixel = yTemp > dims - 1 ? (dims - 1) : yTemp;

			if (yPixel > dims - 1)
				yPixel = dims - 1;
			if (xPixel > dims - 1)
				xPixel = dims - 1;

			if (imageSelect == 1)
				dst.at<cv::Vec3b>(j, i) = posy.at<cv::Vec3b>(int(yPixel), int(xPixel));
			else if (imageSelect == 2)
				dst.at<cv::Vec3b>(j, i) = posx.at<cv::Vec3b>(int(yPixel), int(xPixel));
			else if(imageSelect == 3)
				dst.at<cv::Vec3b>(j, i) = negy.at<cv::Vec3b>(int(yPixel), int(xPixel));
			else if(imageSelect == 4)
				dst.at<cv::Vec3b>(j, i) = negx.at<cv::Vec3b>(int(yPixel), int(xPixel));
			else if(imageSelect == 5)
				dst.at<cv::Vec3b>(j, i) = negz.at<cv::Vec3b>(int(yPixel), int(xPixel));
			else if(imageSelect == 6)
				dst.at<cv::Vec3b>(j, i) = posz.at<cv::Vec3b>(int(yPixel), int(xPixel));
		}
	}
}
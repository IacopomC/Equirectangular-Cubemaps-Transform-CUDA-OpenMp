#include <opencv2/opencv.hpp>

//----------------------------------------------------------------------------------------------------------
//
// takes an equirectangular imageand converts this to a cube map image of the following format
//
//	+----+----+----+
//	| Y+ | X+ | Y- |
//	+----+----+----+
//	| X- | Z- | Z+ |
//	+----+----+----+
//
// which when unfolded should take the following format
//
//	+----+
//	| Z+ |
//	+----+----+----+----+
//	| X+ | Y- | X- | Y+ |
//	+----+----+----+----+
//	| Z- |
//	+----+
//
//-------------------------------------------------------------------------------------------------------

float getTheta(float x, float y)
{
    float PI = 3.14159265358979323846;

    float rtn = 0;

    if (y < 0)
    {
        rtn = atan2(y, x) * -1;
    }
    else
    {
        rtn = PI + (PI - atan2(y, x));
    }
    return rtn;
}

void equi2cube(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst)
{
	float PI = 3.14159265358979323846;

    cv::Size s = src.size();
    float inputHeight = s.height;
    float inputWidth = s.width;

    float sqr = inputWidth / 4.0;
    float outputWidth = sqr * 3;
    float outputHeight = sqr * 2;

    dst.create(outputHeight, outputWidth);
    dst = cv::Vec3b(0, 0, 0);

    float tx = 0;
    float ty = 0;
    float x = 0;
    float y = 0;
    float z = 0;

	float rho = 0;
	float normTheta = 0;
	float normPhi = 0;

	float iX;
	float iY;

	// iterate over pixels output image
	// height inclusive
    #pragma omp parallel for
    for (int loopY = 1; loopY <= outputHeight; loopY++)
    {
		// width inclusive
        for (int loopX = 1; loopX <= outputWidth; loopX++)
        {
			// local coordinates for the cube map face.
            tx = 0;
            ty = 0;

			// normalized local coordinates
            x = 0;
            y = 0;
            z = 0;

			// top half
			if (loopY < sqr + 1) {

				// top left box[Y + ]
				if (loopX < sqr + 1) {

					tx = loopX;
					ty = loopY;
					x = tx - 0.5 * sqr;
					y = 0.5 * sqr;
					z = ty - 0.5 * sqr;
				}
				// top middle[X + ]
				else if (loopX < 2 * sqr + 1) {

					tx = loopX - sqr;
					ty = loopY;
					x = 0.5 * sqr;
					y = (tx - 0.5 * sqr) * -1;
					z = ty - 0.5 * sqr;
				}
				// top right[Y - ]
				else{

					tx = loopX - sqr * 2;
					ty = loopY;
					x = (tx - 0.5 * sqr) * -1;
					y = -0.5 * sqr;
					z = ty - 0.5 * sqr;
				}
			}
			// bottom half
			else {
				// bottom left box[X - ]
				if (loopX < sqr + 1) {

					tx = loopX;
					ty = loopY - sqr;
					x = int(-0.5 * sqr);
					y = int(tx - 0.5 * sqr);
					z = int(ty - 0.5 * sqr);

				}
				// bottom middle[Z - ]
				else if (loopX < 2 * sqr + 1) {

					tx = loopX - sqr;
					ty = loopY - sqr;
					x = (ty - 0.5 * sqr) * -1;
					y = (tx - 0.5 * sqr) * -1;
					z = 0.5 * sqr;
				}
				// bottom right[Z + ]
				else {

					tx = loopX - sqr * 2;
					ty = loopY - sqr;
					x = ty - 0.5 * sqr;
					y = (tx - 0.5 * sqr) * -1;
					z = -0.5 * sqr;
				}
			}
			// now find out the polar coordinates

			rho = sqrt(x * x + y * y + z * z);
			normTheta = getTheta(x, y) / (2 * PI);
			normPhi = (PI - acos(z / rho)) / PI;

			iX = normTheta * inputWidth;
			iY = normPhi * inputHeight;

			// catch possible overflows

			if (iX >= inputWidth) {
				iX = iX - (inputWidth);
			}
			if (iY >= inputHeight) {
				iY = iY - (inputHeight);
			}

			dst.at<cv::Vec3b>(loopY-1, loopX-1) = src.at<cv::Vec3b>(int(iY), int(iX));
			
        }
    }
}

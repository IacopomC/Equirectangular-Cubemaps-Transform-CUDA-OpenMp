#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float getTheta(float x, float y)
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

__global__ void equi2cube(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int drows, int dcols, int srows, int scols)
{
	float PI = 3.14159265358979323846;

	float inputHeight = srows;
	float inputWidth = scols;

	float sqr = inputWidth / 4.0;

	int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

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
	// height and width inclusive
	if (dst_x < dcols && dst_y < drows)
	{
		dst_x += 1;
		dst_y += 1;
		// local coordinates for the cube map face.
		tx = 0;
		ty = 0;

		// normalized local coordinates
		x = 0;
		y = 0;
		z = 0;

		// top half
		if (dst_y < sqr + 1) {

			// top left box[Y + ]
			if (dst_x < sqr + 1) {

				tx = dst_x;
				ty = dst_y;
				x = tx - 0.5 * sqr;
				y = 0.5 * sqr;
				z = ty - 0.5 * sqr;
			}
			// top middle[X + ]
			else if (dst_x < 2 * sqr + 1) {

				tx = dst_x - sqr;
				ty = dst_y;
				x = 0.5 * sqr;
				y = (tx - 0.5 * sqr) * -1;
				z = ty - 0.5 * sqr;
			}
			// top right[Y - ]
			else {

				tx = dst_x - sqr * 2;
				ty = dst_y;
				x = (tx - 0.5 * sqr) * -1;
				y = -0.5 * sqr;
				z = ty - 0.5 * sqr;
			}
		}
		// bottom half
		else {
			// bottom left box[X - ]
			if (dst_x < sqr + 1) {

				tx = dst_x;
				ty = dst_y - sqr;
				x = int(-0.5 * sqr);
				y = int(tx - 0.5 * sqr);
				z = int(ty - 0.5 * sqr);

			}
			// bottom middle[Z - ]
			else if (dst_x < 2 * sqr + 1) {

				tx = dst_x - sqr;
				ty = dst_y - sqr;
				x = (ty - 0.5 * sqr) * -1;
				y = (tx - 0.5 * sqr) * -1;
				z = 0.5 * sqr;
			}
			// bottom right[Z + ]
			else {

				tx = dst_x - sqr * 2;
				ty = dst_y - sqr;
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

		dst(dst_y - 1, dst_x - 1).x = src(int(iY), int(iX)).x;
		dst(dst_y - 1, dst_x - 1).y = src(int(iY), int(iX)).y;
		dst(dst_y - 1, dst_x - 1).z = src(int(iY), int(iX)).z;
	}

}

int divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void equi2cubeCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{

	const dim3 block(32, 32);
	const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

	equi2cube << <grid, block >> > (src, dst, dst.rows, dst.cols, src.rows, src.cols);

}
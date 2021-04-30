#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

int divUp(int a, int b);

__device__ float max_val(float x, float y)
{
	return x > y ? x : y;
}

__global__ void cube2equi(const cv::cuda::PtrStep<uchar3> posy, const cv::cuda::PtrStep<uchar3> negx, const cv::cuda::PtrStep<uchar3> posx,
						  const cv::cuda::PtrStep<uchar3> negz, const cv::cuda::PtrStep<uchar3> negy, const cv::cuda::PtrStep<uchar3> posz,
						  cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int dims)
{
	float PI = 3.14159265358979323846;

	float xPixel = 0;
	float yPixel = 0;
	float yTemp = 0;
	float imageSelect = 0;

	float increment = (dims * 2) / 100;
	float counter = 0;
	float percentCounter = 0;

	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

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
	
	if (dst_x < cols && dst_y < rows)
	{

		if (counter <= dst_y)
		{
			percentCounter += 1;
			counter += increment;
		}

		v = 1.0 - ((float(dst_y)) / (dims * 2));
		phi = v * PI;


		u = (float(dst_x)) / (dims * 4);
		theta = u * 2 * PI;

		// all of these range between 0 and 1
		x = cos(theta) * sin(phi);
		y = sin(theta) * sin(phi);
		z = cos(phi);

		a = max_val(max_val(abs(x), abs(y)), abs(z));

		// one of these will equal either - 1 or +1
		xx = x / a;
		yy = y / a;
		zz = z / a;

		// format is left, front, right, back, bottom, top;
		// therefore negx, posz, posx, negz, negy, posy

		// square 1 left
		if (yy == -1)
		{
			xPixel = (((-1.0 * tan(atan(x / y)) + 1.0) / 2.0) * dims);
			yTemp = (((-1.0 * tan(atan(z / y)) + 1.0) / 2.0) * (dims - 1.0));
			imageSelect = 1;
		}
		// square 2; front
		else if (xx == 1)
		{
			xPixel = (((tan(atan(y / x)) + 1.0) / 2.0) * dims);
			yTemp = (((tan(atan(z / x)) + 1.0) / 2.0) * dims);
			imageSelect = 2;
		}
		// square 3; right
		else if (yy == 1)
		{
			xPixel = (((-1 * tan(atan(x / y)) + 1.0) / 2.0) * dims);
			yTemp = (((tan(atan(z / y)) + 1.0) / 2.0) * (dims - 1));
			imageSelect = 3;
		}
		// square 4; back
		else if (xx == -1) {
			xPixel = (((tan(atan(y / x)) + 1.0) / 2.0) * dims);
			yTemp = (((-1 * tan(atan(z / x)) + 1.0) / 2.0) * (dims - 1));
			imageSelect = 4;
		}
		// square 5; bottom
		else if (zz == 1)
		{
			xPixel = (((tan(atan(y / z)) + 1.0) / 2.0) * dims);
			yTemp = (((-1 * tan(atan(x / z)) + 1.0) / 2.0) * (dims - 1));
			imageSelect = 5;
		}
		// square 6; top
		else if (zz == -1)
		{
			xPixel = (((-1 * tan(atan(y / z)) + 1.0) / 2.0) * dims);
			yTemp = (((-1 * tan(atan(x / z)) + 1.0) / 2.0) * (dims - 1));
			imageSelect = 6;
		}

		yPixel = yTemp > dims - 1 ? (dims - 1) : yTemp;

		if (yPixel > dims - 1)
			yPixel = dims - 1;
		if (xPixel > dims - 1)
			xPixel = dims - 1;

		if (imageSelect == 1)
		{
			dst(dst_y, dst_x).x = posy(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = posy(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = posy(int(yPixel), int(xPixel)).z;
		}
		else if (imageSelect == 2)
		{
			dst(dst_y, dst_x).x = posx(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = posx(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = posx(int(yPixel), int(xPixel)).z;
		}
		else if (imageSelect == 3)
		{
			dst(dst_y, dst_x).x = negy(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = negy(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = negy(int(yPixel), int(xPixel)).z;
		}
		else if (imageSelect == 4)
		{
			dst(dst_y, dst_x).x = negx(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = negx(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = negx(int(yPixel), int(xPixel)).z;
		}
		else if (imageSelect == 5)
		{
			dst(dst_y, dst_x).x = negz(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = negz(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = negz(int(yPixel), int(xPixel)).z;
		}
		else if (imageSelect == 6)
		{
			dst(dst_y, dst_x).x = posz(int(yPixel), int(xPixel)).x;
			dst(dst_y, dst_x).y = posz(int(yPixel), int(xPixel)).y;
			dst(dst_y, dst_x).z = posz(int(yPixel), int(xPixel)).z;
		}
	}
	
}

void cube2equiCUDA(cv::cuda::GpuMat& posy, cv::cuda::GpuMat& negx, cv::cuda::GpuMat& posx, cv::cuda::GpuMat& negz, cv::cuda::GpuMat& negy, cv::cuda::GpuMat& posz, cv::cuda::GpuMat& dst, int dims)
{

	const dim3 block(32, 32);
	const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

	cube2equi << <grid, block >> > (posy, negx, posx, negz, negy, posz, dst, dst.rows, dst.cols, dims);

}
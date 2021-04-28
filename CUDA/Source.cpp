#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void equi2cubeCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

void cube2equiCUDA(cv::cuda::GpuMat& posy, cv::cuda::GpuMat& negx, cv::cuda::GpuMat& posx, cv::cuda::GpuMat& negz, cv::cuda::GpuMat& negy, cv::cuda::GpuMat& posz, cv::cuda::GpuMat& dst, int dims);

int main(int argc, char** argv)
{
    cv::namedWindow("Equirectangular Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("CubeMap Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    // =========== EQUILATERAL TO CUBE MAP ========== //
    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;

    cv::Size s = h_img.size();
    float height = s.height;
    float width = s.width;

    float sqr = width / 4.0;
    float outputWidth = sqr * 3;
    float outputHeight = sqr * 2;

    cv::Size new_Size(outputWidth, outputHeight);
    cv::Mat_<cv::Vec3b> h_result(new_Size);

    h_img = h_img(cv::Range(0, height / 2), cv::Range(0, width)).clone();

    d_img.upload(h_img);
    d_result.upload(h_result);

    equi2cubeCUDA(d_img, d_result);

    // =========== CUBE MAP TO EQUILATERAL ========== //
    cv::Mat_<cv::Vec3b> negx;
    cv::Mat_<cv::Vec3b> posx;
    cv::Mat_<cv::Vec3b> negz;
    cv::Mat_<cv::Vec3b> posz;
    cv::Mat_<cv::Vec3b> negy;
    cv::Mat_<cv::Vec3b> posy;
    cv::cuda::GpuMat d_result_equi;
    cv::cuda::GpuMat d_negx;
    cv::cuda::GpuMat d_posx;
    cv::cuda::GpuMat d_negz;
    cv::cuda::GpuMat d_posz;
    cv::cuda::GpuMat d_negy;
    cv::cuda::GpuMat d_posy;
    
    float cube_width = h_result.size().width / 3.0;
    float cube_height = h_result.size().height / 2.0;

    d_result.download(h_result);

    posy = h_result(cv::Range(0, cube_height), cv::Range(0, cube_width)).clone();
    negx = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(0, cube_width)).clone();
    posx = h_result(cv::Range(0, cube_height), cv::Range(cube_width, 2 * cube_width)).clone();
    negz = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(cube_width, 2 * cube_width)).clone();
    negy = h_result(cv::Range(0, cube_height), cv::Range(2 * cube_width, 3 * cube_width)).clone();
    posz = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(2 * cube_width, 3 * cube_width)).clone();

    float dims = negx.size().width;

    float equiWidth = dims * 4;
    float equiHeight = dims * 2;

    cv::Size new_Size_equi(equiWidth, equiHeight);
    cv::Mat_<cv::Vec3b> h_result_equi(new_Size_equi);

    d_result.download(h_result_equi);

    d_result_equi.upload(h_result_equi);
    d_negx.upload(negx);
    d_posy.upload(posy);
    d_negz.upload(negz);
    d_posx.upload(posx);
    d_posz.upload(posz);
    d_negy.upload(negy);

    cube2equiCUDA(d_posy, d_negx, d_posx, d_negz, d_negy, d_posz, d_result_equi, dims);
    
    cv::imshow("Equirectangular Image", h_img);
    cv::imshow("CubeMap Image", d_result_equi);

    cv::waitKey();
    return 0;
}
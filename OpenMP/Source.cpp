#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void equi2cube(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst);

void cube2equi(cv::Mat_<cv::Vec3b>&  posy, cv::Mat_<cv::Vec3b>&  negx, cv::Mat_<cv::Vec3b>&  posx, cv::Mat_<cv::Vec3b>&  negz, cv::Mat_<cv::Vec3b>&  negy, cv::Mat_<cv::Vec3b>&  posz, cv::Mat_<cv::Vec3b>& dst);

int main(int argc, char** argv)
{
    cv::namedWindow("Equirectangular Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("CubeMap Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    // =========== EQUILATERAL TO CUBE MAP ========== //
    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::Mat_<cv::Vec3b> h_result;
    
    cv::Size s = h_img.size();
    float height = s.height;
    float width = s.width;

    h_img = h_img(cv::Range(0, height / 2), cv::Range(0, width)).clone();

    equi2cube(h_img, h_result);

    // =========== CUBE MAP TO EQUILATERAL ========== //
    cv::Mat_<cv::Vec3b> h__result_equi;
    cv::Mat_<cv::Vec3b> negx;
    cv::Mat_<cv::Vec3b> posx;
    cv::Mat_<cv::Vec3b> negz;
    cv::Mat_<cv::Vec3b> posz;
    cv::Mat_<cv::Vec3b> negy;
    cv::Mat_<cv::Vec3b> posy;

    float cube_width = h_result.size().width / 3.0;
    float cube_height = h_result.size().height / 2.0;
    
    posy = h_result(cv::Range(0, cube_height), cv::Range(0, cube_width)).clone();
    negx = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(0, cube_width)).clone();
    posx = h_result(cv::Range(0, cube_height), cv::Range(cube_width, 2 * cube_width)).clone();
    negz = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(cube_width, 2 * cube_width)).clone();
    negy = h_result(cv::Range(0, cube_height), cv::Range(2 * cube_width, 3 * cube_width)).clone();
    posz = h_result(cv::Range(cube_height, 2 * cube_height), cv::Range(2 * cube_width, 3 * cube_width)).clone();

    cube2equi(posy, negx, posx, negz, negy, posz, h__result_equi);

    cv::imshow("Equirectangular Image", h_img);
    cv::imshow("CubeMap Image", h__result_equi);

    cv::waitKey();
    return 0;
}
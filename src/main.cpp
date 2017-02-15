#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace cv;
using namespace std;

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{
	Eigen::JacobiSVD< _Matrix_Type_ > svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}


int main(int argc, char** argv)
{

	MatrixXf randm = MatrixXf::Random(2, 3);
	cout << "The pseudoinverse of \n" << randm << "\n is \n" << pseudoInverse(randm) << endl;
	cout << "This project brought to you by CMake!" << endl;

	system("pause");

	VideoCapture cap(0); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	while (true)
	{
		Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); 

		if (!bSuccess) //if not able to read, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		imshow("Original", imgOriginal); //show the original image

		if (waitKey(30) == 27) 
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}
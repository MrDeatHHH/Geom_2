#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> 

using namespace cv;
using namespace std;
using namespace std::chrono;

#define PI 3.14159265
#define inf 10000

int* fancyColor(int dx, int dy, int max)
{
	int* color = new int[3];
	int H = int(atan(double(dy) / double(dx)) * 180.0 / PI);
	if (H < 0)
		H += 360;
	else
		H += 180;
	double C = double(abs(dx) + abs(dy)) / double(max);
	double X = C * (1 - abs((H / 60) % 2 - 1));
	switch (H / 60)
	{
	case 0:
		color[0] = int(C * 255);
		color[1] = int(X * 255);
		color[2] = 0;
		break;
	case 1:
		color[0] = int(X * 255);
		color[1] = int(C * 255);
		color[2] = 0;
		break;
	case 2:
		color[0] = 0;
		color[1] = int(C * 255);
		color[2] = int(X * 255);
		break;
	case 3:
		color[0] = 0;
		color[1] = int(X * 255);
		color[2] = int(C * 255);
		break;
	case 4:
		color[0] = int(X * 255);
		color[1] = 0;
		color[2] = int(C * 255);
		break;
	case 5:
		color[0] = int(C * 255);
		color[1] = 0;
		color[2] = int(X * 255);
		break;
	default:
		color[0] = 0;
		color[1] = 0;
		color[2] = 0;
		break;
	}
	return color;
}

int* notfancyColor(int dx, int dy, int max)
{
	int* color = new int[3];
	for (int i = 0; i < 3; ++i)
		color[i] = int(255 * (double(dx) + double(dy)) / double(max));
	return color;
}

int** run(const int modT, const int modK, int** K, const int width, const int height, double** q, double** g)
{
	// Initialize phi, L, R, U, D
	double** L = new double* [modT];
	double** R = new double* [modT];
	double** U = new double* [modT];
	double** D = new double* [modT];
	for (int i = 0; i < modT; ++i)
	{
		L[i] = new double[modK]();
		R[i] = new double[modK]();
		U[i] = new double[modK]();
		D[i] = new double[modK]();
	}
	cout << "Done Init" << endl;

	// Backward
	for (int i = modT - 1; i >= 0; --i)
		if ((i / width != height - 1) && (i % width != width - 1))
			for (int j = 0; j < modK; ++j)
			{
				double maxD = -inf * sqrt(modT);
				double current = 0;
				for (int j_ = 0; j_ < modK; ++j_)
				{
					current = D[i + width][j_] + q[i + width][j_] + g[j][j_];
					if (current > maxD)
						maxD = current;
				}
				D[i][j] = maxD;

				double maxR = -inf * sqrt(modT);
				current = 0;
				for (int j_ = 0; j_ < modK; ++j_)
				{
					current = R[i + 1][j_] + q[i + 1][j_] + g[j][j_];
					if (current > maxR)
						maxR = current;
				}
				R[i][j] = maxR;
			}
		else if (i / width == height - 1)
			for (int j = 0; j < modK; ++j)
				D[i][j] = 0;
		else
			for (int j = 0; j < modK; ++j)
				R[i][j] = 0;
	cout << "Backwards" << endl;

	// Forward
	for (int i = 0; i < modT; ++i)
		if ((i / width != 0) && (i % width != 0))
			for (int j = 0; j < modK; ++j)
			{
				double maxU = -inf * sqrt(modT);
				double current = 0;
				for (int j_ = 0; j_ < modK; ++j_)
				{
					current = U[i - width][j_] + q[i - width][j_] + g[j][j_];
					if (current > maxU)
						maxU = current;
				}
				U[i][j] = maxU;

				double maxL = -inf * sqrt(modT);
				current = 0;
				for (int j_ = 0; j_ < modK; ++j_)
				{
					current = L[i - 1][j_] + q[i - 1][j_] + g[j][j_];
					if (current > maxL)
						maxL = current;
				}
				L[i][j] = maxL;
			}
		else if (i / width == 0)
			for (int j = 0; j < modK; ++j)
				U[i][j] = 0;
		else
			for (int j = 0; j < modK; ++j)
				L[i][j] = 0;
	cout << "Forward" << endl;

	// K best
	int** res = new int* [modT];
	for (int i = 0; i < modT; ++i)
	{
		res[i] = new int[2];
		int k_star = -1;
		double result = -inf * sqrt(modT) * 4;
		double current = 0;
		for (int j = 0; j < modK; ++j)
		{
			current = L[i][j] + R[i][j] + U[i][j] + D[i][j] + q[i][j];
			if (current > result)
			{
				k_star = j;
				result = current;
			}
		}
		res[i][0] = K[k_star][0];
		res[i][1] = K[k_star][1];
	}
	cout << "Best" << endl;
	
	for (int i = 0; i < modT; ++i)
	{
		delete[] L[i];
		delete[] R[i];
		delete[] U[i];
		delete[] D[i];
	}
	delete[] L;
	delete[] R;
	delete[] U;
	delete[] D;

	return res;
}

int main()
{
	Mat Limage_, Limage;
	Limage_ = imread("left.jpg", IMREAD_UNCHANGED);
	cvtColor(Limage_, Limage, COLOR_BGR2GRAY);
	//imshow("Gray image L", Limage);

	Mat Rimage_, Rimage;
	Rimage_ = imread("right.jpg", IMREAD_UNCHANGED);
	cvtColor(Rimage_, Rimage, COLOR_BGR2GRAY);
	//imshow("Gray image R", Rimage);

	auto start = high_resolution_clock::now();

	const int height = Limage.size().height;
	const int width = Limage.size().width;
	const int modT = height * width;

	// Get array from Mat
	int* Lcolors = new int [modT];
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			Lcolors[i * width + j] = int(Limage.at<uchar>(i, j));

	int* Rcolors = new int [modT];
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			Rcolors[i * width + j] = int(Rimage.at<uchar>(i, j));

	const int minDx = -10;
	const int maxDx = 10;
	const int minDy = -5;
	const int maxDy = 5;

	const int sizeX = maxDx - minDx + 1;
	const int sizeY = maxDy - minDy + 1;
	const int maxD = abs(abs(minDx) - (abs(minDx) + abs(maxDx)) / 2) + (abs(minDx) + abs(maxDx)) / 2 +
		             abs(abs(minDy) - (abs(minDy) + abs(maxDy)) / 2) + (abs(minDy) + abs(maxDy)) / 2;
	const int modK = sizeX * sizeY;


	int** K = new int* [modK];
	for (int i = 0; i < sizeX; ++i)
		for (int j = 0; j < sizeY; ++j)
		{
			K[i * sizeY + j] = new int[2];
			K[i * sizeY + j][0] = i + minDx;
			K[i * sizeY + j][1] = j + minDy;
		}

	double** q = new double* [modT];
	for (int i = 0; i < modT; ++i)
	{
		q[i] = new double[modK];
		for (int j = 0; j < modK; ++j)
			if ((i / width + K[j][0] >= 0) &&
				(i / width + K[j][0] < height) &&
				(i % width + K[j][1] >= 0) &&
				(i % width + K[j][1] < width))
				q[i][j] = -abs(Lcolors[i] - Rcolors[i + K[j][0] * width + K[j][1]]);
			else
				q[i][j] = -inf;
	}

	const double alpha = 10;
	double** g = new double* [modK];
	for (int i = 0; i < modK; ++i)
	{
		g[i] = new double[modK];
		for (int j = 0; j < modK; ++j)
			g[i][j] = -alpha * sqrt(pow(K[i][0] - K[j][0], 2) + pow(K[i][1] - K[j][1], 2));
	}

	int** res = run(modT, modK, K, width, height, q, g);
	cout << "Res" << endl;
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
			{
				int* color = fancyColor(res[i * width + j][0], res[i * width + j][1], maxD);
				result[c].at<uchar>(i, j) = uchar(color[c]);
				delete[] color;
			}
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time used: " << double(duration.count()) / 1000000. << endl;

	Mat rez;
	vector<Mat> channels;

	channels.push_back(result[0]);
	channels.push_back(result[1]);
	channels.push_back(result[2]);

	merge(channels, rez);

	namedWindow("Result image", WINDOW_AUTOSIZE);
	imshow("Result image", rez);
	imwrite("res.png", rez);

	waitKey(0);
	return 0;
}
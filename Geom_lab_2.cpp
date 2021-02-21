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

int qFunc(int c1, int c2)
{
	return abs(c1 - c2);
}

int properIndex(int x, const int minDx)
{
	return (x + minDx);
}

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

int** run(const int height, const int width, const int sizeX, const int sizeY, double** q, double** g)
{
	// Initialize phi, L, R, U, D
	const int modT = height * width;
	const int modKs = sizeX * sizeY;
	double** L = new double* [modT];
	double** R = new double* [modT];
	double** U = new double* [modT];
	double** D = new double* [modT];
	for (int ij = 0; ij < modT; ++ij)
	{
		L[ij] = new double[modKs]();
		R[ij] = new double[modKs]();
		U[ij] = new double[modKs]();
		D[ij] = new double[modKs]();
	}
	cout << "Done Init" << endl;
	// Backward
	for (int i = height - 2; i >= 0; --i)
	{
		const int i_ = i * width;
		for (int j = width - 2; j >= 0; --j)
		{
			const int ij = i_ + j;
			for (int k = 0; k < modKs; ++k)
			{
				double maxR = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double R_ = R[ij + 1][k_] + q[ij + 1][k_] + g[k_][k];
					if (R_ > maxR)
						maxR = R_;
				}
				R[i_ + j][k] = maxR;

				double maxD = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double D_ = D[ij + width][k_] + q[ij + width][k_] + g[k_][k];
					if (D_ > maxD)
						maxD = D_;
				}
				D[i_ + j][k] = maxD;
			}
		}
	}
	cout << "Done Backward" << endl;
	// Forward
	for (int i = 1; i < height; ++i)
	{
		const int i_ = i * width;
		for (int j = 1; j < width; ++j)
		{
			const int ij = i_ + j;
			for (int k = 0; k < modKs; ++k)
			{
				double maxL = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double L_ = L[ij - 1][k_] + q[ij - 1][k_] + g[k_][k];
					if (L_ > maxL)
						maxL = L_;
				}
				L[ij][k] = maxL;

				double maxU = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double U_ = U[ij - width][k_] + q[ij - width][k_] + g[k_][k];
					if (U_ > maxU)
						maxU = U_;
				}
				U[ij][k] = maxU;
			}
		}
	}
	cout << "Done Forward" << endl;
	// Best Ks
	int** res = new int* [height];
	for (int i = 0; i < height; ++i)
	{
		res[i] = new int[width]();
	}

	for (int i = 0; i < height; ++i)
	{
		const int i_ = i * width;
		for (int j = 0; j < width; ++j)
		{
			const int ij = i_ + j;
			int k_star = 0;
			double value = -10000000.;
			for (int k_ = 0; k_ < modKs; ++k_)
			{
				const double v_ = L[ij][k_] + R[ij][k_] + q[ij][k_] + R[ij][k_] + U[ij][k_];
				if (v_ > value)
				{
					value = v_;
					k_star = k_;
				}
			}
			res[i][j] = k_star;
		}
	}
	cout << "Done Best Ks" << endl;
	for (int ij = 0; ij < modT; ++ij)
	{
		delete[] L[ij];
		delete[] R[ij];
		delete[] U[ij];
		delete[] D[ij];
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
	Limage_ = imread("new2l.png", IMREAD_UNCHANGED);
	//imshow("Original image", image_);
	cvtColor(Limage_, Limage, COLOR_BGR2GRAY);
	imshow("Gray image", Limage);
	const int Lheight = Limage.size().height;
	const int Lwidth = Limage.size().width;

	Mat Rimage_, Rimage;
	Rimage_ = imread("new2r.png", IMREAD_UNCHANGED);
	//imshow("Original image", image_);
	cvtColor(Rimage_, Rimage, COLOR_BGR2GRAY);
	imshow("Gray image", Rimage);
	const int Rheight = Rimage.size().height;
	const int Rwidth = Rimage.size().width;

	// Get array from Mat
	int** Lcolors = new int* [Lheight];
	for (int i = 0; i < Lheight; ++i)
	{
		Lcolors[i] = new int[Lwidth];
		for (int j = 0; j < Lwidth; ++j)
		{
			Lcolors[i][j] = int(Limage.at<uchar>(i, j));
		}
	}

	int** Rcolors = new int* [Rheight];
	for (int i = 0; i < Rheight; ++i)
	{
		Rcolors[i] = new int[Rwidth];
		for (int j = 0; j < Rwidth; ++j)
		{
			Rcolors[i][j] = int(Rimage.at<uchar>(i, j));
		}
	}

	const int minDx = -1;
	const int maxDx = 9;
	const int minDy = -2;
	const int maxDy = 2;

	const int sizeX = maxDx - minDx + 1;
	const int sizeY = maxDy - minDy + 1;

	auto start = high_resolution_clock::now();

	// G
	double** g = new double* [sizeX * sizeY];
	for (int i = 0; i < sizeX * sizeY; ++i)
	{
		g[i] = new double [sizeX * sizeY]();
		for (int j = 0; j < sizeY; ++j)
			g[i][j] = sqrt(pow((i / sizeX) - (j / sizeX), 2) + pow((i % sizeY) - (j % sizeY), 2));
	}
	cout << "G" << endl;

	// Q
	double** q = new double* [Lheight * Lwidth]();
	for (int i = 0; i < Lheight * Lwidth; ++i)
	{
		q[i] = new double [sizeX * sizeY]();
		for (int j = 0; j < sizeX * sizeY; ++j)
			if ((properIndex((i / Lwidth) + (j / sizeY), minDx) <= Rheight - 1) &&
				(properIndex((i / Lwidth) + (j / sizeY), minDx) >= 0) &&
				(properIndex((i % Lwidth) + (j % sizeY), minDy) <= Rwidth - 1) &&
				(properIndex((i % Lwidth) + (j % sizeY), minDy) >= 0))
				q[i][j] = qFunc(Lcolors[(i / Lwidth)][(i % Lwidth)],
					            Rcolors[properIndex((i / Lwidth) + (j / sizeY), minDx)]
					                   [properIndex((i % Lwidth) + (j % sizeY), minDy)]);
			else
				q[i][j] = 10000000;
	}
	cout << "Q" << endl;

	int** res = run(Lheight, Lwidth, sizeX, sizeY, q, g);
	cout << "Rez" << endl;
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(Lwidth, Lheight), CV_8UC1);
		for (int i = 0; i < Lheight; ++i)
			for (int j = 0; j < Lwidth; ++j)
			{
				int* color = fancyColor(properIndex(res[i][j] / sizeY, minDx), properIndex(res[i][j] % sizeY, minDy), sizeX + sizeY);
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
	imwrite("res1.png", rez);

	waitKey(0);
	return 0;
}
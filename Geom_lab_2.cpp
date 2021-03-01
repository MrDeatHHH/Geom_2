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
#define inf 10000.

int* fancyColor(int dx, int dy, double max)
{
	int* color = new int[3];
	int H = 0;
	double artan = atan(double(dy) / double(dx));
	if (artan == artan)
	{
		H = int(artan * 180.0 / PI);
		if (H < 0)
			H += 360;
	}
	int V = int((100. * sqrt(pow(dx, 2) + pow(dy, 2))) / max);
	int a = (V * (H % 60)) / 60;
	int V_inc = a;
	int V_dec = V - a;
	switch (H / 60)
	{
	case 0:
		color[2] = V;
		color[1] = V_inc;
		color[0] = 0;
		break;
	case 1:
		color[2] = V_dec;
		color[1] = V;
		color[0] = 0;
		break;
	case 2:
		color[2] = 0;
		color[1] = V;
		color[0] = V_inc;
		break;
	case 3:
		color[2] = 0;
		color[1] = V_dec;
		color[0] = V;
		break;
	case 4:
		color[2] = V_inc;
		color[1] = 0;
		color[0] = V;
		break;
	case 5:
		color[2] = V;
		color[1] = 0;
		color[0] = V_dec;
		break;
	default:
		color[0] = 0;
		color[1] = 0;
		color[2] = 0;
		break;
	}
	return color;
}

int* notfancyColor(int dx, int dy, double max)
{
	int* color = new int[3];
	for (int i = 0; i < 3; ++i)
		color[i] = (255 * sqrt(pow(dx, 2) + pow(dy, 2))) / max;
	return color;
}

int*** run(const int sizeX, const int sizeY, int*** K, const int width, const int height, double**** q, double**** g)
{
	// Initialize phi, L, R, U, D
	double**** L = new double*** [width];
	double**** R = new double*** [width];
	double**** U = new double*** [width];
	double**** D = new double*** [width];
	for (int x = 0; x < width; ++x)
	{
		L[x] = new double** [height]();
		R[x] = new double** [height]();
		U[x] = new double** [height]();
		D[x] = new double** [height]();
		for (int y = 0; y < height; ++y)
		{
			L[x][y] = new double* [sizeX]();
			R[x][y] = new double* [sizeX]();
			U[x][y] = new double* [sizeX]();
			D[x][y] = new double* [sizeX]();
			for (int dx = 0; dx < sizeX; ++dx)
			{
				L[x][y][dx] = new double [sizeY]();
				R[x][y][dx] = new double [sizeY]();
				U[x][y][dx] = new double [sizeY]();
				D[x][y][dx] = new double [sizeY]();
			}
		}
	}
	cout << "Done Init" << endl;

	// Forward
	for (int x = 1; x < width; ++x)
	{
		for (int y = 1; y < height; ++y)
		{
			for (int dx1 = 0; dx1 < sizeX; ++dx1)
			{
				for (int dy1 = 0; dy1 < sizeY; ++dy1)
				{
					double maxU = -inf * max(width, height);
					double currentU = 0;
					for (int dx2 = 0; dx2 < sizeX; ++dx2)
					{
						for (int dy2 = 0; dy2 < sizeY; ++dy2)
						{
							currentU = U[x][y - 1][dx2][dy2] + q[x][y - 1][dx2][dy2] + g[dx1][dy1][dx2][dy2];
							if (currentU > maxU)
								maxU = currentU;
						}
					}
					U[x][y][dx1][dy1] = maxU;

					double maxL = -inf * max(width, height);
					double currentL = 0;
					for (int dx2 = 0; dx2 < sizeX; ++dx2)
					{
						for (int dy2 = 0; dy2 < sizeY; ++dy2)
						{
							currentL = L[x - 1][y][dx2][dy2] + q[x - 1][y][dx2][dy2] + g[dx1][dy1][dx2][dy2];
							if (currentL > maxL)
								maxL = currentL;
						}
					}
					L[x][y][dx1][dy1] = maxL;
				}
			}
		}
	}
	cout << "Forward" << endl;

	// Backward
	for (int x = width - 2; x > -1; --x)
	{
		for (int y = height - 2; y > -1; --y)
		{
			for (int dx1 = 0; dx1 < sizeX; ++dx1)
			{
				for (int dy1 = 0; dy1 < sizeY; ++dy1)
				{
					double maxD = -inf * max(width, height);
					double currentD = 0;
					for (int dx2 = 0; dx2 < sizeX; ++dx2)
					{
						for (int dy2 = 0; dy2 < sizeY; ++dy2)
						{
							currentD = D[x][y + 1][dx2][dy2] + q[x][y + 1][dx2][dy2] + g[dx1][dy1][dx2][dy2];
							if (currentD > maxD)
								maxD = currentD;
						}
					}
					D[x][y][dx1][dy1] = maxD;

					double maxR = -inf * max(width, height);
					double currentR = 0;
					for (int dx2 = 0; dx2 < sizeX; ++dx2)
					{
						for (int dy2 = 0; dy2 < sizeY; ++dy2)
						{
							currentR = R[x + 1][y][dx2][dy2] + q[x + 1][y][dx2][dy2] + g[dx1][dy1][dx2][dy2];
							if (currentR > maxR)
								maxR = currentR;
						}
					}
					R[x][y][dx1][dy1] = maxR;
				}
			}
		}
	}
	cout << "Backwards" << endl;

	// K best
	int*** res = new int** [width];
	for (int x = 0; x < width; ++x)
	{
		res[x] = new int* [height];
		for (int y = 0; y < height; ++y)
		{
			res[x][y] = new int[2];
			int best_dx = -1;
			int best_dy = -1;
			double result = -inf * max(width, height) * 4;
			double current = 0;
			for (int dx = 0; dx < sizeX; ++dx)
			{
				for (int dy = 0; dy < sizeY; ++dy)
				{
					current = L[x][y][dx][dy] +
						R[x][y][dx][dy] +
						U[x][y][dx][dy] +
						D[x][y][dx][dy] +
						q[x][y][dx][dy];
					if (current > result)
					{
						best_dx = dx;
						best_dy = dy;
						result = current;
					}
				}
			}
			res[x][y][0] = K[best_dx][best_dy][0];
			res[x][y][1] = K[best_dx][best_dy][1];
		}
	}
	cout << "Best" << endl;
	
	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			for (int dx = 0; dx < sizeX; ++dx)
			{
				delete[] L[x][y][dx];
				delete[] R[x][y][dx];
				delete[] U[x][y][dx];
				delete[] D[x][y][dx];
			}
			delete[] L[x][y];
			delete[] R[x][y];
			delete[] U[x][y];
			delete[] D[x][y];
		}
		delete[] L[x];
		delete[] R[x];
		delete[] U[x];
		delete[] D[x];
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
	Limage_ = imread("sofal.png", IMREAD_UNCHANGED);
	cvtColor(Limage_, Limage, COLOR_BGR2GRAY);
	//imshow("Gray image L", Limage);

	Mat Rimage_, Rimage;
	Rimage_ = imread("sofar.png", IMREAD_UNCHANGED);
	cvtColor(Rimage_, Rimage, COLOR_BGR2GRAY);
	//imshow("Gray image R", Rimage);

	auto start = high_resolution_clock::now();

	const int height = Limage.size().height;
	const int width = Limage.size().width;

	// Get array from Mat
	int** Lcolors = new int* [width];
	for (int x = 0; x < width; ++x)
	{
		Lcolors[x] = new int [height];
		for (int y = 0; y < height; ++y)
			Lcolors[x][y] = int(Limage.at<uchar>(y, x));
	}

	int** Rcolors = new int* [width];
	for (int x = 0; x < width; ++x)
	{
		Rcolors[x] = new int[height];
		for (int y = 0; y < height; ++y)
			Rcolors[x][y] = int(Rimage.at<uchar>(y, x));
	}
	cout << "Read Images" << endl;

	const int minDx = -10;
	const int maxDx = 10;
	const int minDy = -5;
	const int maxDy = 5;

	const int sizeX = maxDx - minDx + 1;
	const int sizeY = maxDy - minDy + 1;
	const double maxD = sqrt(pow(max(abs(minDx), abs(maxDx)), 2) + pow(max(abs(minDy), abs(maxDy)), 2));

	int*** K = new int** [sizeX];
	for (int dx = 0; dx < sizeX; ++dx)
	{
		K[dx] = new int* [sizeY];
		for (int dy = 0; dy < sizeY; ++dy)
		{
			K[dx][dy] = new int[2];
			K[dx][dy][0] = dx + minDx;
			K[dx][dy][1] = dy + minDy;
		}
	}

	double**** q = new double*** [width];
	for (int x = 0; x < width; ++x)
	{
		q[x] = new double** [height];
		for (int y = 0; y < height; ++y)
		{
			q[x][y] = new double* [sizeX];
			for (int dx = 0; dx < sizeX; ++dx)
			{
				q[x][y][dx] = new double [sizeY];
				for (int dy = 0; dy < sizeY; ++dy)
				{
					if ((x + K[dx][dy][0] >= 0) &&
						(x + K[dx][dy][0] < width) &&
						(y + K[dx][dy][1] >= 0) &&
						(y + K[dx][dy][1] < height))
					{
						q[x][y][dx][dy] = -abs(Lcolors[x][y] - Rcolors[x + K[dx][dy][0]][y + K[dx][dy][1]]);
					}
					else
					{
						q[x][y][dx][dy] = -inf;
					}
				}
			}
		}
	}
	cout << "Q" << endl;
	const double alpha = 10;
	double**** g = new double*** [sizeX];
	for (int dx1 = 0; dx1 < sizeX; ++dx1)
	{
		g[dx1] = new double** [sizeY];
		for (int dy1 = 0; dy1 < sizeY; ++dy1)
		{
			g[dx1][dy1] = new double* [sizeX];
			for (int dx2 = 0; dx2 < sizeX; ++dx2)
			{
				g[dx1][dy1][dx2] = new double [sizeY];
				for (int dy2 = 0; dy2 < sizeY; ++dy2)
				{
					g[dx1][dy1][dx2][dy2] = -alpha * sqrt(pow(K[dx1][dy1][0] - K[dx2][dy2][0], 2) +
						                                  pow(K[dx1][dy1][1] - K[dx2][dy2][1], 2));
				}
			}
		}
	}
	cout << "G" << endl;
	int*** res = run(sizeX, sizeY, K, width, height, q, g);
	cout << "Res" << endl;
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y)
			{
				int* color = fancyColor(res[x][y][0], res[x][y][1], maxD);
				result[c].at<uchar>(y, x) = uchar(color[c]);
				delete[] color;
			}
	}

	imwrite("test.png", result[0]);

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
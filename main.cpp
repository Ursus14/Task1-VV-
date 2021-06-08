#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <ratio>
#include <vector>
#include <thread>

#define thrdsCount 8
#define eps 0.000001

using namespace std;

double f(double x);
double exactSol(double a, double b);
pair<double, int> numSol(double a, double b);
pair<double, int> atomicSol(double a, double b);
pair<double, int> criticalSol(double a, double b);
pair<double, int> locksSol(double a, double b);
pair<double, int> reductionSol(double a, double b);
void outAllResult(double* massA, double* massB);

int main()
{
	double* mA = new double[7]{ 0.00001,0.0001,0.001,0.01,0.1,1,10 };
	double* mB = new double[7]{ 0.0001,0.001,0.01,0.1,1,10,100 };

	outAllResult(mA, mB);
	
	return 0;
}

void outAllResult(double* massA, double* massB)
{
	ofstream file;
	file.open("res.txt");
	if (file)
	{
		file << "A;B;tex;Nn;tn;Na;ta;Nc;tc;Nl;tl;Nr;tr" << endl;
		for (int i = 0; i < 7; i++)
		{
			file << massA[i] << ";" << massB[i] << ";";

			//exactSol
			auto t1 = std::chrono::high_resolution_clock::now();
			auto resex = exactSol(massA[i], massB[i]);
			auto t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
			file << fp_ms.count() << ";";

			//num
			t1 = std::chrono::high_resolution_clock::now();
			auto res = numSol(massA[i], massB[i]);
			t2 = std::chrono::high_resolution_clock::now();
			fp_ms = t2 - t1;
			file << res.second << ";"<< fp_ms.count() <<";";

			//atomic
			t1 = std::chrono::high_resolution_clock::now();
			res = atomicSol(massA[i], massB[i]);
			t2 = std::chrono::high_resolution_clock::now();
			fp_ms = t2 - t1;
			file << res.second << ";" << fp_ms.count() << ";";

			//criticalSol
			t1 = std::chrono::high_resolution_clock::now();
			res = criticalSol(massA[i], massB[i]);
			t2 = std::chrono::high_resolution_clock::now();
			fp_ms = t2 - t1;
			file << res.second << ";" << fp_ms.count() << ";";

			//locksSol
			t1 = std::chrono::high_resolution_clock::now();
			res = locksSol(massA[i], massB[i]);
			t2 = std::chrono::high_resolution_clock::now();
			fp_ms = t2 - t1;
			file << res.second << ";" << fp_ms.count() << ";";

			//reductionSol
			t1 = std::chrono::high_resolution_clock::now();
			res = reductionSol(massA[i], massB[i]);
			t2 = std::chrono::high_resolution_clock::now();
			fp_ms = t2 - t1;
			file << res.second << ";" << fp_ms.count();

			file << endl;
		}
	}
	else
	{
		cout << "File not open!" << endl;
	}
}

double f(double x)
{
	return pow(sin(1 / x) / x, 2);
}

double exactSol(double a, double b)
{
	return (1 / 4.0) * (2 * ((b - a) / (a * b)) + sin(2 / b) - sin(2 / a));
}

pair<double,int> numSol(double a, double b)
{
	int n = 1;
	double Jn, Jold, sumf;
	Jn = ((b - a) / n) * ((f(a) / 2.0) + (f(b) / 2.0));
	do
	{
		n++;
		Jold = Jn;
		sumf = 0;
		for (int i = 0; i <= n; i++)
		{
			sumf += f(a + i * (b - a) / (n + 1.0));
		}
		Jn = ((b - a) / n) * ((f(a) / 2.0) + sumf + (f(b) / 2.0));
		if (n == 60000) return { Jn,n };
	} while (abs(Jn - Jold) > eps*abs(Jn));
	
	//cout << "n =" << n << endl;

	return { Jn,n };
}

pair<double, int> atomicSol(double a, double b)
{
	int n = 1;
	double x, Jn, Jold, sumf;
	Jn = ((b - a) / n) * ((f(a) / 2.0) + (f(b) / 2.0));
	do
	{
		n++;
		Jold = Jn;
		sumf = 0;
		#pragma omp parallel for num_threads(thrdsCount)  private(x)
		for (int i = 0; i <= n; i++)
		{
			x = a + i * (b - a) / (n + 1.0);
			#pragma omp atomic
			sumf += f(x);
		}

		Jn = ((b - a) / n) * ((f(a) / 2.0) + sumf + (f(b) / 2.0));
		if (n == 60000) return { Jn,n };
	} while (abs(Jn - Jold) > eps * abs(Jn));

	//cout << "n = " << n << endl;

	return { Jn,n };
}

pair<double, int> criticalSol(double a, double b)
{
	int n = 1;
	double x, Jn, Jold, sumf;
	Jn = ((b - a) / n) * ((f(a) / 2.0) + (f(b) / 2.0));
	do
	{
		n++;
		Jold = Jn;
		sumf = 0;
#pragma omp parallel for num_threads(thrdsCount)  private(x)
		for (int i = 0; i <= n; i++)
		{
			x = a + i * (b - a) / (n + 1.0);
#pragma omp critical
			sumf += f(x);
		}

		Jn = ((b - a) / n) * ((f(a) / 2.0) + sumf + (f(b) / 2.0));
		if (n == 60000) return { Jn,n };
	} while (abs(Jn - Jold) > eps * abs(Jn));

	//cout << "n = " << n << endl;

	return { Jn,n };
}

pair<double, int> locksSol(double a, double b)
{
	int n = 1;
	double x, Jn, Jold, sumf;
	Jn = ((b - a) / n) * ((f(a) / 2.0) + (f(b) / 2.0));
	do
	{
		n++;
		Jold = Jn;
		sumf = 0;
		omp_lock_t lock;
		omp_init_lock(&lock);
#pragma omp parallel for num_threads(thrdsCount)  private(x)
		for (int i = 0; i <= n; i++)
		{
			x = a + i * (b - a) / (n + 1.0);
			omp_set_lock(&lock);
			sumf += f(x);
			omp_unset_lock(&lock);
		}
		omp_destroy_lock(&lock);
		Jn = ((b - a) / n) * ((f(a) / 2.0) + sumf + (f(b) / 2.0));
		if (n == 60000) return { Jn,n };
	} while (abs(Jn - Jold) > eps * abs(Jn));

	//cout << "n = " << n << endl;

	return { Jn,n };
}

pair<double, int> reductionSol(double a, double b)
{
	int n = 1;
	double x, Jn, Jold, sumf;
	Jn = ((b - a) / n) * ((f(a) / 2.0) + (f(b) / 2.0));
	do
	{
		n++;
		Jold = Jn;
		sumf = 0;
#pragma omp parallel num_threads(thrdsCount) private(x)
		{
#pragma omp for reduction(+:sumf)
			for (int i = 0; i <= n; i++)
			{
				x = a + i * (b - a) / (n + 1.0);
				sumf += f(x);
			}
		}
		Jn = ((b - a) / n) * ((f(a) / 2.0) + sumf + (f(b) / 2.0));
		if (n == 60000) return { Jn,n };
	} while (abs(Jn - Jold) > eps * abs(Jn));

	//cout << "n = " << n << endl;

	return { Jn,n };
}
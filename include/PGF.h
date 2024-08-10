#pragma once
#include <cmath>
#include "complex_bessel.h"
static constexpr double M_PI_ = 3.14159265358979323846264338327950288419716939937510L; // pi

template<typename T>
T __1D_LGF__(T x, T y, T z, T Lx, double epi = 1e-10)
{
// Shift to [0, L)
	if (x < 0) x += Lx;
	if (x >= Lx) x -= Lx;

	// Further shift to [0, L/2]
	if (x > Lx / 2) x = Lx - x;

	double epsilon = epi / Lx;
	double rho = std::sqrt(y * y + z * z);
	double p = -std::log(rho) / (2 * M_PI_ * Lx);
	int Mmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * rho));
	for(int m = 1; m <= Mmax; m++)
	{
		p += cyl_bessel_k(0, 2 * m * M_PI_ * rho / Lx) * \
			std::cos(2 * M_PI_ * m * x / Lx) / \
			(M_PI_ * Lx);
	}
	return (T)p;
}

template<typename T>
T __2D_LGF__(T x, T y, T z, T Lx, T Ly, double epi = 1e-10)
{
	// Shift to [0, L)
	if (x < 0) x += Lx;
	if (y < 0) y += Ly;
	if (z < 0) z = std::abs(z);
	if (x >= Lx) x -= Lx;
	if (y >= Ly) y -= Ly;

	// Further shift to [0, L/2]
	if (x > Lx / 2) x = Lx - x;
	if (y > Ly / 2) y = Ly - y;

	double epsilon = epi / std::min(Lx, Ly);
	// swap to make y >= x
	if (x < y)
	{
		std::swap(x, y);
		std::swap(Lx, Ly);
	}
	double p = -z / (2 * Lx * Ly);
	p -= std::log(1 - 2 * std::exp(-2 * M_PI_ * std::abs(z) / Ly) * std::cos(2 * M_PI_ * y / Ly) + std::exp(-4 * M_PI_ * std::abs(z) / Ly)) / (4 * M_PI_ * Lx);

	int Mmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * std::sqrt(y * y + z * z)));
	int Nmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * Ly));
	for(int m = 1; m <= Mmax; m++)
	{
		for (int n = -Nmax; n <= Nmax; n++)
		{
			p += cyl_bessel_k(0, 2 * m * M_PI_ * std::sqrt(pow(n * Ly + y, 2) + pow(z, 2)) / Lx) * \
				std::cos(2 * M_PI_ * m * x / Lx) / \
				(M_PI_ * Lx);
		}
	}

	return (T)p;
}

template<typename T>
T __3D_LGF__(T x, T y, T z, T Lx, T Ly, T Lz, double epi = 1e-10)
{
	double epsilon = epi / std::min(Lx, std::min(Ly, Lz));
	// Shift to [0, L)
	if (x < 0) x += Lx;
	if (y < 0) y += Ly;
	if (z < 0) z += Lz;
	if (x >= Lx) x -= Lx;
	if (y >= Ly) y -= Ly;
	if (z >= Lz) z -= Lz;
	// Further shift to [0, L/2]
	if (x > Lx / 2) x = Lx - x;
	if (y > Ly / 2) y = Ly - y;
	if (z > Lz / 2) z = Lz - z;

	// swap coordinate to make z >= y >= x
	if (x < y)
	{
		std::swap(x, y);
		std::swap(Lx, Ly);
	}
	if (y < z)
	{
		std::swap(y, z);
		std::swap(Ly, Lz);
	}
	if (x < y)
	{
		std::swap(x, y);
		std::swap(Lx, Ly);
	}

	double p = (z * z - std::abs(z) * Lz) / (2 * Lx * Ly * Lz);
	int Kmax = (int) ceil(Ly * std::log(1 / epsilon) / (2 * M_PI_ * Lz));
	for (int k = -Kmax; k <= Kmax; k++)
	{
		p -= (std::log(1 - \
			2 * std::exp(-2 * M_PI_ * std::abs(k * Lz + z) / Ly) * cos(2 * M_PI_ * y / Ly) + \
			std::exp(-4 * M_PI_ * std::abs(k * Lz + z) / Ly) \
		)) / (4 * M_PI_ * Lx);
	}
	int Mmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * std::sqrt(y * y + z * z)));
	int Nmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * Ly));
	Kmax = (int)ceil(Lx * std::log(1 / epsilon) / (2 * M_PI_ * Lz));
	for (int m = 1; m <= Mmax; m++)
	{
		for (int k = -Kmax; k <= Kmax; k++)
		{
			for (int n = -Nmax; n <= Nmax; n++)
			{
				p += cyl_bessel_k(0, 2 * m * M_PI_ * std::sqrt(pow(n * Ly + y, 2) + pow(k * Lz + z, 2)) / Lx) * \
					std::cos(2 * M_PI_ * m * x / Lx) / \
					(M_PI_ * Lx);
			}
		}
	}


	return (T)p;
}

template<typename T>
std::complex<T> __1D_PGF__(T x, T y, T z, T Lx, T Ly, T Lz, std::complex<T> Kx, std::complex<T> Ky, std::complex<T> Kz, std::complex<T> K0, double epi = 1e-10)
{
	// let's assume Lx is the periodic direction, if not swap
	if(Ly > 0)
	{
		std::swap(Lx, Ly);
		std::swap(x, y);
		std::swap(Kx, Ky);
	}
	if(Lz > 0)
	{
		std::swap(Lx, Lz);
		std::swap(x, z);
		std::swap(Kx, Kz);
	}
	double epsilon = epi / Lx;
	std::complex<T> sum = std::complex<T>(0, 0);
	T rho = std::sqrt(y * y + z * z);
	int M = Lx * std::log(1 / epsilon) / (2 * M_PI_ * rho);
	auto const_part = std::complex<T>(0, -1 / (4 * Lx));

	// hankel function of the second kind
	for(int m = -M; m <= M; m++)
	{
		std::complex<double> Kxm = Kx + 2 * M_PI_ * m / Lx;
		std::complex<double> exp_part = std::exp(std::complex<double>(0, -1) * Kxm * x);
		std::complex<double> Krm = std::sqrt(K0 * K0 - Kxm * Kxm);
		if(Krm.imag() > 0)
		{
			Krm = -Krm;
		}
		auto z_input = Krm * rho;
		std::complex<double> hankel_part = sp_bessel::besselJ(0, z_input) - std::complex<double>(0, 1) * sp_bessel::besselY(0, z_input);
		sum += exp_part * hankel_part;
	}

	return sum * const_part;
}

template<typename T>
std::complex<T> __2D_PGF__(T x, T y, T z, T Lx, T Ly, T Lz, std::complex<T> Kx, std::complex<T> Ky, std::complex<T> Kz, std::complex<T> K0, double epi = 1e-10)
{	
	// Assume Lx, Ly are periodic directions
	// if not, swap
	if(Lx == 0)
	{
		std::swap(Lx, Lz);
		std::swap(x, z);
		std::swap(Kx, Kz);
	}
	if(Ly == 0)
	{
		std::swap(Ly, Lz);
		std::swap(y, z);
		std::swap(Ky, Kz);
	}
	double epsilon = epi / std::min(Lx, Ly);
	int M = (int)std::sqrt(Lx * Ly * std::log(1 / epsilon) * std::log(1 / epsilon) / (4 * M_PI_ * M_PI_ * z * z));
	int N = M;
	std::complex<T> sum = std::complex<T>(0, 0);
	for(int m = -M; m <= M; m++)
	{
		auto Kxm = Kx + 2 * M_PI_ * m / Lx;
		for(int n = -N; n <= N; n++)
		{
			auto Kyn = Ky + 2 * M_PI_ * n / Ly;
			auto Kzmn = std::sqrt(K0 * K0 - Kxm * Kxm - Kyn * Kyn);
			if(Kzmn.imag() > 0)
			{
				Kzmn = -Kzmn;
			}
			auto denominator = 2.0 * std::complex<T>(0, 1) * Kzmn * Lx * Ly;
			auto exp_part = std::exp(std::complex<T>(0, -1) * (Kxm * x + Kyn * y + Kzmn * std::abs(z)));
			sum += exp_part / denominator;
		}
	}
	return sum;
}

template<typename T>
std::complex<T> __3D_PGF__(T x, T y, T z, T Lx, T Ly, T Lz, std::complex<T> Kx, std::complex<T> Ky, std::complex<T> Kz, std::complex<T> K0, double epi = 1e-10)
{
	// swap the x y z order to make z the largest
	if (abs(x) > abs(z))
	{
		std::swap(x, z);
		std::swap(Kx, Kz);
		std::swap(Lx, Lz);
	
	}
	if (abs(y) > abs(z))
	{
		std::swap(y, z);
		std::swap(Ky, Kz);
		std::swap(Ly, Lz);
	}
	double epsilon = epi / std::min(Lx, std::min(Ly, Lz));
	int M = (int)std::sqrt(Lx * Ly * std::log(1 / epsilon) * std::log(1 / epsilon) / (4 * M_PI_ * M_PI_ * z * z));
	int N = M;
	std::complex<T> sum = std::complex<T>(0, 0);
	for(int m = -M; m <= M; m++)
	{
		auto Kxm = Kx + 2 * M_PI_ * m / Lx;
		for(int n = -N; n <= N; n++)
		{
			auto Kyn = Ky + 2 * M_PI_ * n / Ly;
			auto Kzmn = std::sqrt(K0 * K0 - Kxm * Kxm - Kyn * Kyn);
			if(Kzmn.imag() > 0)
			{
				Kzmn = -Kzmn;
			}
			auto denominator1 = 2.0 * std::complex<T>(0, 1) * Kzmn * Lx * Ly;
			auto exp_part1 = std::exp(std::complex<T>(0, -1) * (Kxm * x + Kyn * y));
			auto outside_braket = exp_part1 / denominator1;
			auto term1 = std::exp(std::complex<T>(0, -1) * Kzmn * std::abs(z));
			auto term2_deniminator = 1. - std::exp(std::complex<T>(0, -1) * (Kzmn - Kz) * Lz);
			auto term2_exp = std::exp(std::complex<T>(0, -1) * (Kzmn - Kz) * Lz) * std::exp(std::complex<T>(0, -1) * Kzmn * z);
			auto term2 = term2_exp / term2_deniminator;
			auto term3_deniminator = 1. - std::exp(std::complex<T>(0, -1) * (Kzmn + Kz) * Lz);
			auto term3_exp = std::exp(std::complex<T>(0, -1) * (Kzmn + Kz) * Lz) * std::exp(std::complex<T>(0, 1) * Kzmn * z);
			auto term3 = term3_exp / term3_deniminator;
			sum += outside_braket * (term1 + term2 + term3);
		}
	}
	return sum;
}

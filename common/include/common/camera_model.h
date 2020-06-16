#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace common
{
struct CameraModelParams
{
	double fx = 0;
	double fy = 0;
	double cx = 0;
	double cy = 0;
	double k1 = 0;
	double k2 = 0;
	double k3 = 0;
	double k4 = 0;
	double p1 = 0;
	double p2 = 0;
};

// KannalaBrandt4Camera
// TODO Simulate tangential distortion as well
template <typename Scalar = double>
class CameraModel
{
   public:
	typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
	typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

	CameraModel(const CameraModelParams& p) : param_(p) {}

	inline Scalar getPolynomial(const Scalar& x, const Scalar& k1,
								const Scalar& k2, const Scalar& k3,
								const Scalar& k4) const
	{
		const Scalar xSqr = x * x;

		const Scalar b0 = k3 + k4 * xSqr;
		const Scalar b1 = k2 + b0 * xSqr;
		const Scalar b2 = k1 + b1 * xSqr;
		const Scalar b3 = b2 * xSqr;
		return Scalar(1) + b3;
	}

	inline Scalar getNormalPolynomial(const Scalar& x) const
	{
		const Scalar& k1 = param_.k1;
		const Scalar& k2 = param_.k2;
		const Scalar& k3 = param_.k3;
		const Scalar& k4 = param_.k4;

		return x * getPolynomial(x, k1, k2, k3, k4);
	}

	inline Scalar getNormalPolynomialDerivative(const Scalar& x) const
	{
		const Scalar& k1 = param_.k1;
		const Scalar& k2 = param_.k2;
		const Scalar& k3 = param_.k3;
		const Scalar& k4 = param_.k4;
		return getPolynomial(x, Scalar(3) * k1, Scalar(5) * k2, Scalar(7) * k3,
							 Scalar(9) * k4);
	}

	inline Vec2 project(const Vec3& p) const
	{
		const Scalar& fx = param_.fx;
		const Scalar& fy = param_.fy;
		const Scalar& cx = param_.cx;
		const Scalar& cy = param_.cy;

		const Scalar& x = p[0];
		const Scalar& y = p[1];
		const Scalar& z = p[2];

		Vec2 res;
		const Scalar r = sqrt(x * x + y * y);

		Scalar d_by_r;
		if (r < Sophus::Constants<Scalar>::epsilon())
		{
			d_by_r = 1.0 / z;
		}
		else
		{
			const Scalar theta = atan2(r, z);
			const Scalar d = getNormalPolynomial(theta);
			d_by_r = d / r;
		}

		res << fx * d_by_r * x + cx, fy * d_by_r * y + cy;
		return res;
	}

	Vec3 unproject(const Vec2& p) const
	{
		const Scalar& fx = param_.fx;
		const Scalar& fy = param_.fy;
		const Scalar& cx = param_.cx;
		const Scalar& cy = param_.cy;

		const Scalar mx = (p(0) - cx) / fx;
		const Scalar my = (p(1) - cy) / fy;
		const Scalar r = sqrt(mx * mx + my * my);

		Scalar thetaGuess = Scalar(1.);

		constexpr int maxNumIterations = 10;
		const Scalar allowedResidual = Scalar(10e-10);
		for (int i = 0; i < maxNumIterations; ++i)
		{
			const Scalar oldPolynomial = getNormalPolynomial(thetaGuess);
			const Scalar oldPolynomialDerivative =
				getNormalPolynomialDerivative(thetaGuess);
			thetaGuess =
				thetaGuess - (oldPolynomial - r) / (oldPolynomialDerivative);

			if (abs(oldPolynomial - r) < allowedResidual)
			{
				break;
			}
		}

		Scalar weight = sin(thetaGuess) / r;
		if (r == Scalar(0))
		{
			weight = Scalar(1);
		}

		Vec3 res;
		res << weight * mx, weight * my, cos(thetaGuess);

		return res;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   private:
	CameraModelParams param_;
};

}  // namespace common
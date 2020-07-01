#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <memory>
#include <opencv2/opencv.hpp>

namespace common
{
template <typename Scalar = double>
struct CameraModelParams
{
	Scalar fx = 0;
	Scalar fy = 0;
	Scalar cx = 0;
	Scalar cy = 0;
	Scalar k1 = 0;
	Scalar k2 = 0;
	Scalar k3 = 0;
	Scalar p1 = 0;
	Scalar p2 = 0;
};

template <typename Scalar = double>
class CameraModel
{
   public:
	typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
	typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

	CameraModel(const CameraModelParams<Scalar> p) : param_(p) {}

	inline Scalar getTangentialDistortion(const Scalar& p1, const Scalar& p2,
										  const Scalar& x, const Scalar& y,
										  const Scalar& r2) const
	{
		return Scalar(2) * p1 * x * y + p2 * (r2 + Scalar(2) * x * x);
	}

	inline Scalar getRadialDistortion(const Scalar& r2) const 
	{
		const Scalar& k1 = param_.k1;
		const Scalar& k2 = param_.k2;
		return Scalar(1) + k1 * r2 + k2 * r2 * r2;
	}

	inline Vec2 project(const Vec3& p) const
	{
		const Scalar& fx = param_.fx;
		const Scalar& fy = param_.fy;
		const Scalar& cx = param_.cx;
		const Scalar& cy = param_.cy;
		const Scalar& p1 = param_.p1;
		const Scalar& p2 = param_.p2;

		const Scalar& x = p[0];
		const Scalar& y = p[1];
		const Scalar& z = p[2];

		Vec2 res;

		const Scalar xPrime = x / z;
		const Scalar yPrime = y / z;
		const Scalar r2 = xPrime * xPrime + yPrime * yPrime;

		const Scalar radialDistortion = getRadialDistortion(r2);

		const Scalar xD = xPrime * radialDistortion +
						  getTangentialDistortion(p1, p2, xPrime, yPrime, r2);
		const Scalar yD = yPrime * radialDistortion +
						  getTangentialDistortion(p2, p1, yPrime, xPrime, r2);

		res << fx * xD + cx, fy * yD + cy;
		return res;
	}

	Vec3 unproject(const Vec2& p) const
	{
		const Scalar& x = p[0];
		const Scalar& y = p[1];

		const Scalar& fx = param_.fx;
		const Scalar& fy = param_.fy;
		const Scalar& cx = param_.cx;
		const Scalar& cy = param_.cy;
		const Scalar& p1 = param_.p1;
		const Scalar& p2 = param_.p2;

		const Scalar xDistort = (x - cx) / fx;
		const Scalar yDistort = (y - cy) / fy;

		Scalar xOpt = xDistort;
		Scalar yOpt = yDistort;

		for (size_t i = 0; i < 10; ++i)
		{
			const Scalar r2 = xOpt * xOpt + yOpt * yOpt;
			const Scalar radial = getRadialDistortion(r2);
			const Scalar deltaX = getTangentialDistortion(p1, p2, xOpt, yOpt, r2);
			const Scalar deltaY = getTangentialDistortion(p2, p1, yOpt, xOpt, r2);

	        xOpt = (xDistort - deltaX) / radial;
	        yOpt = (yDistort - deltaY) / radial;
		}

		const Scalar norm = sqrt(xOpt * xOpt + yOpt * yOpt + Scalar(1));
		const Scalar z = Scalar(1) / norm;

		Vec3 output;
		output << xOpt / norm, yOpt / norm, z;
		return output;
	}

	Scalar* getParams() { return reinterpret_cast<Scalar*>(&param_); }

	static std::shared_ptr<CameraModel<Scalar>> fromData(const Scalar* params)
	{
		return std::shared_ptr<CameraModel<Scalar>>(new CameraModel<Scalar>(
			*reinterpret_cast<const CameraModelParams<Scalar>*>(params)));
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   private:
	CameraModelParams<Scalar> param_;
};

}  // namespace common
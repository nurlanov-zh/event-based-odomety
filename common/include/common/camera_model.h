#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opencv2/opencv.hpp>

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
	double p1 = 0;
	double p2 = 0;
};

template <typename Scalar = double>
class CameraModel
{
   public:
	typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
	typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

	CameraModel(const CameraModelParams& p) : param_(p)
	{
		K_ = (cv::Mat_<float>(3, 3) << param_.fx, 0.0, param_.cx, 0.0,
			  param_.fy, param_.cy, 0.0, 0.0, 1.0);
		D_ = (cv::Mat_<float>(1, 5) << param_.k1, param_.k2, param_.p1,
			  param_.p2, param_.k3);
	}

	inline Scalar getTangentialDistortion(const Scalar& p1, const Scalar& p2,
										  const Scalar& x, const Scalar& y,
										  const Scalar& r2) const
	{
		return 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
	}

	inline Vec2 project(const Vec3& p) const
	{
		const Scalar& fx = param_.fx;
		const Scalar& fy = param_.fy;
		const Scalar& cx = param_.cx;
		const Scalar& cy = param_.cy;
		const Scalar& k1 = param_.k1;
		const Scalar& k2 = param_.k2;
		const Scalar& p1 = param_.p1;
		const Scalar& p2 = param_.p2;

		const Scalar& x = p[0];
		const Scalar& y = p[1];
		const Scalar& z = p[2];

		Vec2 res;

		const Scalar xPrime = x / z;
		const Scalar yPrime = y / z;
		const Scalar r2 = xPrime * xPrime + yPrime * yPrime;

		const Scalar radialDistortion = 1 + k1 * r2 + k2 * r2 * r2;

		const Scalar xD = xPrime * radialDistortion +
						  getTangentialDistortion(p1, p2, xPrime, yPrime, r2);
		const Scalar yD = yPrime * radialDistortion +
						  getTangentialDistortion(p2, p1, yPrime, xPrime, r2);

		res << fx * xD + cx, fy * yD + cy;
		return res;
	}

	Vec3 unproject(const Vec2& p) const
	{
		Vec3 res;
		const Scalar& x = p[0];
		const Scalar& y = p[1];

		cv::Point2f uv(x, y);
		cv::Point2f px;
		const cv::Mat src(1, 1, CV_32FC2, &uv.x);
		cv::Mat dst(1, 1, CV_32FC2, &px.x);
		cv::undistortPoints(src, dst, K_, D_);

		res << px.x, px.y, 1.0;

		return res.normalized();
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   private:
	CameraModelParams param_;
	cv::Mat D_;
	cv::Mat K_;
};

}  // namespace common
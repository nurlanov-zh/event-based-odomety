// Some type definitions:
#include "visual_odometry/aligner.h"

namespace visual_odometry
{
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using ArrX = Eigen::ArrayXd;

/// Compute Sim(3) transformation that aligns the 3D points in model to 3D
/// points in data in the least squares sense using Horn's algorithm. I.e. the
/// Sim(3) transformation T is computed that minimizes the sum over all i of
/// ||T*m_i - d_i||^2, m_i are the column's of model and d_i are the column's of
/// data. Both data and model need to be of the same dimension and have at least
/// 3 columns.
///
/// Optionally computes the translational rmse.
///
/// Note that for the orientation we don't actually use Horn's algorithm, but
/// the one published by Arun
/// (http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf) based on SVD with
/// later extension to cover degenerate cases.
///
/// See the illustrative comparison by Eggert:
/// http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
Sophus::Sim3d align_points_sim3(const Eigen::Ref<const Mat3X> &data,
								const Eigen::Ref<const Mat3X> &model,
								ErrorMetricValue *ate)
{
	//   CHECK_EQ(data.cols(), model.cols());
	//   CHECK_GE(data.cols(), 3);

	// 0. Centroids
	const Vec3 centroid_data = data.rowwise().mean();
	const Vec3 centroid_model = model.rowwise().mean();

	// center both clouds to 0 centroid
	const Mat3X data_centered = data.colwise() - centroid_data;
	const Mat3X model_centered = model.colwise() - centroid_model;

	// 1. Rotation

	// sum of outer products of columns
	const Mat3 W = data_centered * model_centered.transpose();

	const auto svd = W.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	// last entry to ensure we don't get a reflection, only rotations
	const Mat3 S = Eigen::DiagonalMatrix<double, 3, 3>(
		1, 1,
		svd.matrixU().determinant() * svd.matrixV().determinant() < 0 ? -1 : 1);

	const Mat3 R = svd.matrixU() * S * svd.matrixV().transpose();

	const Mat3X model_rotated = R * model_centered;

	// 2. Scale (regular, non-symmetric variant)

	// sum of column-wise dot products
	const double dots = (data_centered.cwiseProduct(model_rotated)).sum();

	// sum of column-wise norms
	const double norms = model_centered.colwise().squaredNorm().sum();

	// scale
	const double s = dots / norms;

	// 3. Translation
	const Vec3 t = centroid_data - s * R * centroid_model;

	// 4. Translational error
	if (ate)
	{
		static_assert(ArrX::ColsAtCompileTime == 1);

		const Mat3X diff = data - ((s * R * model).colwise() + t);
		const ArrX errors = diff.colwise().norm().transpose();
		auto &ref = *ate;
		ref.rmse = std::sqrt(errors.square().sum() / errors.rows());
		ref.mean = errors.mean();
		ref.min = errors.minCoeff();
		ref.max = errors.maxCoeff();
		ref.count = errors.rows();
	}

	return Sophus::Sim3d(Sophus::RxSO3d(s, R), t);
}

// Interface to construct the matrices of camera centers from a list of poses /
// cameras using calibration. Assumes calib_cam.T_i_c really contains the
// correct extrinsic calibration from camera to IMU frame, that
// reference_poses are transformations from IMU to world, and that cameras
// contain cam to world poses.
Sophus::Sim3d align_cameras_sim3(const std::vector<common::Pose3d> &reference_poses,
								 const std::list<Keyframe>& cameras,
								 ErrorMetricValue *ate)
{
	const Eigen::Index num_cameras = static_cast<int32_t>(cameras.size());

	Mat3X reference_centers(3, num_cameras);
	Mat3X camera_centers(3, num_cameras);

    size_t i = 0;
	for (const auto &kf : cameras)
	{
		const auto &T_w_i = reference_poses[i];
		reference_centers.col(i) = T_w_i.translation();
		camera_centers.col(i) = kf.pose.translation();
        ++i;
	}

	return align_points_sim3(reference_centers, camera_centers, ate);
}
}  // namespace visual_odometry

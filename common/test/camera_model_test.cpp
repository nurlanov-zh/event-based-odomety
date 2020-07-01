#include <common/camera_model.h>

#include <gtest/gtest.h>

TEST(CameraModelTest, simpleReprojectionCase)
{
	common::CameraModelParams params;
	params.fx = 199.092366542;
	params.fy = 198.82882047;
	params.cx = 132.192071378;
	params.cy = 110.712660011;
	params.k1 = -0.368436311798;
	params.k2 = 0.150947243557;
	params.p1 = -0.000296130534385;
	params.p2 = -0.000759431726241;
	params.k3 = 0.0;

	typedef typename common::CameraModel<double>::Vec2 Vec2;
	typedef typename common::CameraModel<double>::Vec3 Vec3;
	common::CameraModel<double> cam(params);

	for (int x = -9; x <= 9; x++)
	{
		for (int y = -9; y <= 9; y++)
		{
			Vec3 p(x, y, 10);

			Vec3 pNormalized = p.normalized();
			Vec2 res = cam.project(p);
			Vec3 pUproj = cam.unproject(res);
			EXPECT_TRUE(pNormalized.isApprox(pUproj, 0.01));
		}
	}
}
#pragma once

#include <pangolin/gl/gl.h>
#include <Eigen/Dense>

namespace tools
{
void renderCamera(const Eigen::Matrix4d& T_w_c, float lineWidth,
				  const u_int8_t* color, float sizeFactor)
{
	glPushMatrix();
	glMultMatrixd(T_w_c.data());
	glColor3ubv(color);
	glLineWidth(lineWidth);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	const float sz = sizeFactor;
	const float width = 640, height = 480, fx = 500, fy = 500, cx = 320,
				cy =
					240;  // choose an arbitrary intrinsics because we don't
						  // need the camera be exactly same as the original one
	glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	glVertex3f(0, 0, 0);
	glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(0, 0, 0);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(0, 0, 0);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	glEnd();
	glPopMatrix();
}
}  // namespace tools
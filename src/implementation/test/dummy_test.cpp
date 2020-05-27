#include <gtest/gtest.h>

TEST(DummyTest, DummyTest)
{
	ASSERT_TRUE(2 + 2 == 4);
	ASSERT_FALSE(2 + 2 == 5);
}
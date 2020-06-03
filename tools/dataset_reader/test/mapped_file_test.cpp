#include <dataset_reader/mapped_file.h>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "test/test_data/events.txt";

TEST(MappedFileTest, smokeTest)
{
	EXPECT_NO_THROW(tools::MappedFile file(TEST_DATA_PATH));

	EXPECT_THROW(tools::MappedFile file("fake_file.txt"), std::runtime_error);
}

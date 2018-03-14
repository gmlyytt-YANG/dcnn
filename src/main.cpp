#include "dcnn.h"

int main(void) {
    string config_file_path("../example/test_config_without_ground_truth.txt");
    //string output_file("../testImgs/result.txt");
    dcnn(config_file_path);

    return 0;
}

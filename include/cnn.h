#include "utils.h"
//#include "caffe/net.hpp"
//#include "caffe/common.hpp"
//#include <caffe/caffe.hpp>
//#include <caffe/caffe.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <algorithm>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>

using namespace caffe;
using namespace std;

class CNN {
public:
    static void get_instance(string& net, string& model) {
        if (!instance.get()) {
            instance.reset(new CNN(net, model));
        }
        //return instance;
    }
    static boost::shared_ptr<CNN> get_instance() {
        return instance;
    }
    
    //~CNN() {
    //    if (_cnn != NULL)
    //        delete _cnn;
    //}

    // 前向传播函数
    // vector<pair<float, float> > forward(Mat& img);
    cv::Mat_<double> forward(Mat& img);

private:
    //std::string _net; // prototxt文件路径
    //std::string _model; // caffemodel 文件路径
    boost::shared_ptr<Net<float> > _cnn; // 根据_net和_model 构建的网络
    CNN(const std::string& net, const std::string& model);
    static boost::shared_ptr<CNN> instance;
};


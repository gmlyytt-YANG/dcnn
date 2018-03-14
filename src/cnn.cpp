#include "cnn.h"

CNN::CNN(const string& net, const string& model){ 
    try {
        Caffe::set_mode(Caffe::CPU);
        
        _cnn.reset(new Net<float>(net, TEST));
        _cnn->CopyTrainedLayersFrom(model);
    }
    catch(std::exception& e) {
        std::cout << "Can not open " << net << ", " << model << std::endl;
    }
}

boost::shared_ptr<CNN> CNN::instance(NULL);

cv::Mat_<double> CNN::forward(Mat& img) {
    caffe::MemoryDataLayer<float>* m_layer = (caffe::MemoryDataLayer<float> *)_cnn->layers()[0].get();
    float label = 0.0;
    float* data_ptr = (float*)(img.data);
    float* label_ptr = &label;
    m_layer->Reset(data_ptr, label_ptr, img.channels());
    _cnn->Forward();
    boost::shared_ptr<Blob<float> > layer_data = _cnn->blob_by_name("fc2");
    const float* pstart = layer_data->cpu_data();

    //vector<pair<float, float> > result;
    cv::Mat_<double> shape(5,2);
    for (int i=0; i<5; i++) {
        shape(i,0) = *(pstart);
        shape(i,1) = *(pstart+1);
        //cout << shape(i,0);
        //cout << shape(i,1);
        //result.push_back(make_pair(*(pstart), *(pstart+1)));
        pstart += 2;
    }
    //delete m_layer;

    //return result;
    return shape;
}


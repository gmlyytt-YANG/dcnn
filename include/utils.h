
#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include "linear.h"
#include <stdio.h>
//opencv
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

//caffe
#ifndef USE_OPENCV
#define USE_OPENCV
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/net.hpp>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#endif // USE_OPENCV


using namespace cv;
using namespace std;
using namespace caffe;

// 人脸框
class BBox {
public:
    BBox(vector<int> bbox): _left(bbox[0]), _right(bbox[1]),
                            _top(bbox[2]), _bottom(bbox[3]){
        start_x = bbox[0];
        start_y = bbox[2];
        width = bbox[1] - bbox[0];
        height = bbox[3] - bbox[2];
    }
    BBox(){
        start_x = 0;
        start_y = 0;
        width = 0;
        height = 0;
        center_x = 0;
        center_y = 0;
    }

    inline int left() {return _left;}
    inline int right() {return _right;}
    inline int top() {return _top;}
    inline int bottom() {return _bottom;}

    BBox subBBox(float, float, float, float); // 在BBox的基础上进行制定方向的偏移得到的bounding box
    //void reproject(pair<float, float>&); // 把一个归一化的坐标点映射回原始坐标
    void reprojectLandmark(cv::Mat_<double>&); // 把整张脸的归一化五官点坐标映射回原始坐标
    string shape_string(); // 字符串形式的人脸框位置

public:
    int _left;
    int _right;
    int _top;
    int _bottom;
    int start_x;
    int start_y;
    int width; // 人脸框宽
    int height; // 人脸框高
    double center_x;
    double center_y;
};

// <人脸图片路径+人脸框>结构体
struct path_BBox {
    string _img_path; // 图片路径
    BBox _bbox; // 人脸框
    path_BBox(std::string path, BBox bbox): _img_path(path), _bbox(bbox) {}
};

class FeatureLocations
{
public:
	cv::Point2d start;
	cv::Point2d end;
	FeatureLocations(cv::Point2d a, cv::Point2d b){
		start = a;
		end = b;
	}
	FeatureLocations(){
		start = cv::Point2d(0.0, 0.0);
		end = cv::Point2d(0.0, 0.0);
	};
};

class Parameters {
	//private:
public:
	int local_features_num_;
	int landmarks_num_per_face_;
	int regressor_stages_;
	int tree_depth_;
	int trees_num_per_forest_;
	std::vector<double> local_radius_by_stage_;
	int initial_guess_;
	cv::Mat_<double> mean_shape_;
	double overlap_;

	Parameters() {

	}

	~Parameters() {
		local_radius_by_stage_.clear();
	}
	void output(){
        std::cout << "local_features_num_: " << local_features_num_ << std::endl;
        std::cout << "landmarks_num_per_face_: " << landmarks_num_per_face_ << std::endl;
        std::cout << "regressor_stages_: " << regressor_stages_ << std::endl;
        std::cout << "tree_depth_: " << tree_depth_ << std::endl;
        std::cout << "trees_num_per_forest_: " << trees_num_per_forest_ << std::endl;
		std::cout << "overlap_: " << overlap_ << std::endl;
        std::cout << "initial_guess_: " << initial_guess_ << std::endl;
        std::cout << "local_radius_by_stages_:";

        for (int i = 0; i < local_radius_by_stage_.size(); i++) {
            std::cout << " " << local_radius_by_stage_[i];
        }
        std::cout << std::endl;
    }

};

// 创建路径
void createDir(std::string&);

// 消除字符串的前后的空格
void erase_space(std::string&); 

// 把str用substr截断，返回截断后的子串数组
vector<string> split(const string& str, const string& substr);

// 读取文件的目录名(以'/'结尾)
string dirname(string);

// 读取文件的文件名
string basename(string);

// 从图片集的描述文档中读取每一张图片的信息(包括每张图片的路径以及人脸框位置)，
// 保存在path_BBox结构体中，并把所有的path_BBox结构体放入数组中
void getDataFromTxt(string&, vector<boost::shared_ptr<path_BBox> >&);

// 预处理图片，减均值除标准差
void processImage(Mat&);

// 根据提供的Mat型图片以及图片的人脸框位置以及人脸五官点坐标，把人脸框和五官点标注在图片上
Mat drawLandmark(string& , BBox&, cv::Mat_<double>&);

tuple<string, string, string> load_files(std::vector<string>& images_paths,
                std::vector<BBox>& bboxes,
                string& config_file_path);

BBox load_bbox_shape(const char* name, vector<int>& bbox);

void load_images_bboxes(std::vector<string>& images_paths,
                 std::vector<BBox>& bboxes,
                 std::vector<std::string>& image_path_prefix,
                 std::vector<std::string>& image_lists);

Mat get_face(BBox& bbox, Mat& img);

void record_pts(string& pts_path, cv::Mat_<double>& landmark);

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BBox& bbox);
cv::Mat_<double> ReProjection(const cv::Mat_<double>& shape, const BBox& bbox);
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
	const std::vector<BBox>& all_bboxes);
void getSimilarityTransform(const cv::Mat_<double>& shape_to,
	const cv::Mat_<double>& shape_from,
	cv::Mat_<double>& rotation, double& scale);

cv::Mat_<double> LoadGroundTruthShape(const char* name);

bool ShapeInRect(cv::Mat_<double>& ground_truth_shape, cv::Rect&);

double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape);

void DrawPredictImage(cv::Mat_<uchar>& image, cv::Mat_<double>& shapes);

BBox GetBoundingBox(cv::Mat_<double>& shape, int width, int height);

#endif

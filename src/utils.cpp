#include "utils.h"

BBox BBox::subBBox(float leftR, float rightR, float topR, float bottomR) {
    vector<int> bbox_vec;
    float leftDelta = width * leftR;
    float rightDelta = width * rightR;
    float topDelta = height * topR;
    float bottomDelta = height * bottomR;
    int left = _left + int(leftDelta); 
    int right = _left + int(rightDelta);
    int top = _top + int(topDelta);
    int bottom = _top + int(bottomDelta);
    bbox_vec.push_back(left);
    bbox_vec.push_back(right);
    bbox_vec.push_back(top);
    bbox_vec.push_back(bottom); 
    
    return BBox(bbox_vec);
}


void BBox::reprojectLandmark(cv::Mat_<double>& landmark) {
    for (int i=0; i<landmark.rows; i++) {
        landmark(i,0) = start_x + width * landmark(i,0);
        landmark(i,1) = start_y + height * landmark(i,1);
    }
}

string BBox::shape_string() {
    string item = "";
    item += to_string(_left) + " " + to_string(_right) + " " + to_string(_top) + " " + to_string(_bottom);
    
    return item;
}

void createDir(string& path) {
    if (access(path.c_str(), 0) == -1){
        string cmd("mkdir "+path);
        system(cmd.c_str());       
    }
}

void erase_space(string &s)  {  
    s.erase(s.find_last_not_of(" ") + 1);  
    s.erase(0, s.find_first_not_of(" "));  
}  

std::vector<std::string> split(const  std::string& s, const std::string& delim) {
    std::vector<std::string> elems;
    size_t pos = 0;
    size_t len = s.length();
    size_t delim_len = delim.length();
    if (delim_len == 0) return elems;
    while (pos < len) {
        int find_pos = s.find(delim, pos);
        if (find_pos < 0) {
            elems.push_back(s.substr(pos, len - pos));
            break;
        }
        elems.push_back(s.substr(pos, find_pos - pos));
        pos = find_pos + delim_len;
    }
    return elems;
}

string dirname(string file_path) {
    vector<string> component = split(file_path, "/");
    string file_name = component[component.size()-1];
    int position = file_path.find(file_name, 0);
    file_path.erase(position, file_name.size());
    return file_path;
}

string basename(string file_path) {
    vector<string> component = split(file_path, "/");
    return component[component.size()-1];
}

void getDataFromTxt(string& txt, vector<boost::shared_ptr<path_BBox> >& result) {
    ifstream file;
    ofstream fout;
    file.open(txt.c_str(), ios::in);
    //fout.close();
    if (!file.is_open()) exit(-1);
    
    //fout << "version: 1" << endl;
    //fout << "n_points: 5" << endl;
    //fout << "{" << endl;
    
    string dir = dirname(txt);  
    string str_line;
    while (getline(file, str_line)) {
        if (str_line.empty()) continue;
        erase_space(str_line);
        vector<string> components = split(str_line, " ");
        string img_path = dir + components[0];
        string bbox_file = split(img_path, ".jpg")[0] + ".bbox";
        fout.open(bbox_file, ios::out);
        if (!fout.is_open()) exit(-1);
        vector<int> bbox;
        for (int i=1; i<=4; i++) {
            bbox.push_back(atoi(components[i].c_str()));
            fout << components[i] << endl;
        }
        fout.close();
        //int arr[] = {atoi(components[1].c_str()), atoi(components[2].c_str()),
        //                    atoi(components[3].c_str()), atoi(components[4].c_str())};
        //vector<int> bbox(arr, arr+sizeof(arr)/sizeof(int));
        boost::shared_ptr<path_BBox> pb(new path_BBox(img_path, BBox(bbox)));
        result.push_back(pb);
    }
    file.close();
}

void processImage(Mat& img) {
    Scalar Mean, Std;
    meanStdDev(img, Mean, Std);
    double matMean = Mean.val[0];
    double stdDev = Std.val[0];
    double alpha = double(1/stdDev);
    double beta = -double(matMean/stdDev);
    Mat matrix = Mat::ones(img.size(), img.type());
    addWeighted(img, alpha, matrix, beta, 0.0, img, -1);
}

Mat drawLandmark(string& img_path, BBox& bbox, cv::Mat_<double>& landmark) {
    Mat img = cv::imread(img_path.c_str());
    rectangle(img, Point(bbox.left(), bbox.top()), Point(bbox.right(), bbox.bottom()), Scalar(0, 0, 255), 1, 8, 0);
    for (int i=0; i<landmark.rows; i++) {
        circle(img, Point(landmark(i,0), landmark(i,1)), 2, Scalar(0, 255, 0), 3);
    }
    return img;
}

tuple<string, string, string> load_files(std::vector<string>& images_paths,
                std::vector<BBox>& bboxes,
                string& config_file_path){
    cout << "parsing config_file: " << config_file_path << endl;
    ifstream fin;
    fin.open(config_file_path, ifstream::in);
    std::string dcnn_net, dcnn_param, cas_model_name;
    fin >> dcnn_net;
    fin >> dcnn_param;
    fin >> cas_model_name;
   
    bool images_has_ground_truth = false;
    fin >> images_has_ground_truth;
    if (images_has_ground_truth) {
        cout << "the image lists must have ground_truth_shapes!\n" << endl;
    }
    else{
        cout << "the image lists does not have ground_truth_shapes!!!\n" << endl;
    }

    int path_num;
    fin >> path_num;
    cout << "reading testing images paths: " << endl;
    std::vector<std::string> image_path_prefixes;
    std::vector<std::string> image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        image_lists.push_back(s);
    }
    cout << "parsing config file done" << endl;

    cout << "\nLoading test dataset..." << std::endl;
    load_images_bboxes(images_paths, bboxes, image_path_prefixes, image_lists);
    
    return make_tuple(dcnn_net, dcnn_param, cas_model_name);
}

BBox load_bbox_shape(const char* name, vector<int>& bbox) {
    ifstream fin;
    fin.open(name, fstream::in);
    for (int i=0; i<4; i++) {
        fin >> bbox[i];
    }    
    fin.close();
    return BBox(bbox);
}

void load_images_bboxes(std::vector<string>& images_paths,
                 std::vector<BBox>& bboxes,
                 std::vector<std::string>& image_path_prefix,
                 std::vector<std::string>& image_lists){
    cout << "loading images..." << std::endl;
    int count = 0;
    for (int i = 0; i < image_path_prefix.size(); i++) {
        int c = 0, count = 0;
        std::ifstream fin;
        fin.open((image_lists[i]).c_str(), std::ifstream::in);
        if (!fin.is_open()) exit(-1);
        std::string path_prefix = image_path_prefix[i];
        std::string image_file_name, image_pts_name, image_bbox_name;
        std::cout << "loading images in folder: " << path_prefix << std::endl;
        while (fin >> image_file_name >> image_bbox_name) {
            string image_path, bbox_path;
            if (path_prefix[path_prefix.size()-1] == '/') {
                image_path = path_prefix + image_file_name;
                bbox_path = path_prefix + image_bbox_name;
            }
            else {
                image_path = path_prefix + "/" + image_file_name;
                bbox_path = path_prefix + "/" + image_bbox_name;
            }
            images_paths.push_back(image_path);

            vector<int> box(4, 0);
            BBox bbox = load_bbox_shape(bbox_path.c_str(), box);
            bboxes.push_back(bbox);
            //if ((count++)%100 == 0) {std::cout << count << " images loaded\n";}
        }
        fin.close();
    }
    cout << "get " << bboxes.size() << " faces in total" << std::endl;
}

Mat get_face(BBox& bbox, Mat& img) {
    Mat face_row = img.rowRange(bbox.top(), bbox.bottom()+1);
    Mat face = face_row.colRange(bbox.left(), bbox.right()+1);
    resize(face, face, Size(39, 39));
    processImage(face);
    return face;
}

void record_pts(string& pts_path, cv::Mat_<double>& landmark) {
    ofstream fout;
    fout.open(pts_path, ios::out);
    if (!fout.is_open()) exit(-1);
    fout << "version: 1" << endl;
    fout << "n_points: 5" << endl;
    fout << "{" << endl;

    for (int j=0; j<landmark.rows; j++) {
        fout << to_string(landmark(j,0)) << " " << to_string(landmark(j,1)) << endl;
        //item += to_string(landmark[j].first) + " " + to_string(landmark[j].second) + " ";
    }
    fout << "}" << endl;
    fout.close();
}

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = (shape(i, 0) - bbox.center_x) / (bbox.width / 2.0);
		results(i, 1) = (shape(i, 1) - bbox.center_y) / (bbox.height / 2.0);
	}
	return results;
}

// reproject the shape to global coordinates
cv::Mat_<double> ReProjection(const cv::Mat_<double>& shape, const BBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = shape(i, 0)*bbox.width / 2.0 + bbox.center_x;
		results(i, 1) = shape(i, 1)*bbox.height / 2.0 + bbox.center_y;
	}
	return results;
}

// get the mean shape, [-1, 1]x[-1, 1]
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
	const std::vector<BBox>& all_bboxes) {

	cv::Mat_<double> mean_shape = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
	for (int i = 0; i < all_shapes.size(); i++)
	{
		mean_shape += ProjectShape(all_shapes[i], all_bboxes[i]);
	}
	mean_shape = 1.0 / all_shapes.size()*mean_shape;
	return mean_shape;
}

// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void getSimilarityTransform(const cv::Mat_<double>& shape_to,
	const cv::Mat_<double>& shape_from,
	cv::Mat_<double>& rotation, double& scale){
	rotation = cv::Mat(2, 2, 0.0);
	scale = 0;

	// center the data
	double center_x_1 = 0.0;
	double center_y_1 = 0.0;
	double center_x_2 = 0.0;
	double center_y_2 = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		center_x_1 += shape_to(i, 0);
		center_y_1 += shape_to(i, 1);
		center_x_2 += shape_from(i, 0);
		center_y_2 += shape_from(i, 1);
	}
	center_x_1 /= shape_to.rows;
	center_y_1 /= shape_to.rows;
	center_x_2 /= shape_from.rows;
	center_y_2 /= shape_from.rows;

	cv::Mat_<double> temp1 = shape_to.clone();
	cv::Mat_<double> temp2 = shape_from.clone();
	for (int i = 0; i < shape_to.rows; i++){
		temp1(i, 0) -= center_x_1;
		temp1(i, 1) -= center_y_1;
		temp2(i, 0) -= center_x_2;
		temp2(i, 1) -= center_y_2;
	}


	cv::Mat_<double> covariance1, covariance2;
	cv::Mat_<double> mean1, mean2;
	// calculate covariance matrix
  cv::calcCovarMatrix(temp1, covariance1, mean1, cv::COVAR_NORMAL | cv::COVAR_COLS, CV_64F); //CV_COVAR_COLS
  cv::calcCovarMatrix(temp2, covariance2, mean2, cv::COVAR_NORMAL | cv::COVAR_COLS, CV_64F);

	double s1 = sqrt(norm(covariance1));
	double s2 = sqrt(norm(covariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	double num = 0.0;
	double den = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}

	double norm = sqrt(num*num + den*den);
	double sin_theta = num / norm;
	double cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}

cv::Mat_<double> LoadGroundTruthShape(const char* name){
	int landmarks = 0;
	std::ifstream fin;
	std::string temp;
	fin.open(name, std::fstream::in);
	getline(fin, temp);// read first line
	fin >> temp >> landmarks;
	cv::Mat_<double> shape(landmarks, 2);
	getline(fin, temp); // read '\n' of the second line
	getline(fin, temp); // read third line
	for (int i = 0; i<landmarks; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
	fin.close();
	return shape;
}

cv::Mat_<double> LoadBBoxShape(const char* name) {
	std::ifstream fin;
	cv::Mat_<double> shape(4, 1);
	fin.open(name, std::fstream::in);
	for (int i = 0; i < 4; i++) {
		fin >> shape(i, 0);
	}
	fin.close();
	return shape;
}
    
cv::Mat_<double> LoadPtsShape(const char* name) {
    std::ifstream fin;
    string num_str;
    int num_pts;
    fin.open(name, std::fstream::in);
    fin >> num_str >> num_str 
        >> num_str >> num_str;
    
    //num_str = split(num_str, ":")[1];
    erase_space(num_str);
    num_pts = atoi(num_str.c_str());
    cv::Mat_<double> shape(num_pts, 2);
    
    fin >> num_str;
    for (int i=0; i<num_pts; i++) {
        fin >> shape(i, 0) >> shape(i, 1);
    }
    fin.close();
    return shape;
}

bool ShapeInRect(cv::Mat_<double>& shape, cv::Rect& ret){
	double sum_x = 0.0, sum_y = 0.0;
	double max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
    if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 2.0) return false;
    if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 2.0) return false;
	return true;
}


double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape){
    cv::Mat_<double> temp;
    temp = ground_truth_shape.rowRange(0, 1)-ground_truth_shape.rowRange(1, 2);
    double x =mean(temp.col(0))[0];
    double y = mean(temp.col(1))[0];
    double interocular_distance = sqrt(x*x+y*y);
    double sum = 0;
    for (int i=0;i<ground_truth_shape.rows;i++){
        sum += norm(ground_truth_shape.row(i)-predicted_shape.row(i));
    }
    return sum/(ground_truth_shape.rows*interocular_distance);
}

void DrawPredictImage(cv::Mat_<uchar> image, cv::Mat_<double>& shape){
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	cv::imshow("show image", image);
	cv::waitKey(0);
}

BBox GetBoundingBox(cv::Mat_<double>& shape, int width, int height){
	double min_x = 100000.0, min_y = 100000.0;
	double max_x = -1.0, max_y = -1.0;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);
	}
	BBox bbox;
	double scale = 0.6;
	bbox.start_x = min_x - (max_x - min_x) * (scale - 0.5);
	if (bbox.start_x < 0.0)
	{
		bbox.start_x = 0.0;
	}
	bbox.start_y = min_y - (max_y - min_y) * (scale - 0.5);
	if (bbox.start_y < 0.0)
	{
		bbox.start_y = 0.0;
	}
	bbox.width = (max_x - min_x) * scale * 2.0;
	if (bbox.width >= width){
		bbox.width = width - 1.0;
	}
	bbox.height = (max_y - min_y) * scale * 2.0;
	if (bbox.height >= height){
		bbox.height = height - 1.0;
	}
	bbox.center_x = bbox.start_x + bbox.width / 2.0;
	bbox.center_y = bbox.start_y + bbox.height / 2.0;
	return bbox;
}

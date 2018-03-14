#include "dcnn.h"
#include "regressor.h"

void dcnn(string& config_file_path) {
    string OUTPUT("../example/out_0/");
    float time_consume = 0.0;
    time_t t_start, t_end;
    t_start = time(NULL);
    ofstream fout;
    fout.open(dirname(config_file_path)+"test_images_list_with_ground_truth.txt", ios::out);
    createDir(OUTPUT);

    //vector<cv::Mat_<uchar> > images;
    vector<BBox> bboxes;
    vector<string> images_paths;
    
    tuple<string, string, string> model_param = load_files(images_paths, bboxes, config_file_path);
    
    string net = get<0>(model_param);
    string model = get<1>(model_param);
    CNN F = CNN(net, model);
    CascadeRegressor cas_load;
    cas_load.LoadCascadeRegressor(get<2>(model_param));
    cout << "load model done\n" << endl;
    
    int count = 0;
    for (int i=0; i<bboxes.size(); i++) {
        Mat img_gray=cv::imread(images_paths[i],0);
        img_gray.convertTo(img_gray, CV_32FC1);
        if (!img_gray.data) {
            cout << "error in open img "<< images_paths[i] << endl;
            continue;
        }
        string base = split(images_paths[i], ".jpg")[0];
        cout << "procesing" << " " << images_paths[i] << endl;
        BBox f_bbox = bboxes[i].subBBox(-0.05, 1.05, -0.05, 1.05);
        Mat f_face = get_face(f_bbox, img_gray);
        //cout << f_face << endl; 
        //vector<pair<float, float> > landmark = F.forward(f_face);
        cv::Mat_<double> current_shape = F.forward(f_face);
        //cout << current_shape(0,0) << " " << current_shape(0,1) << endl;
        bboxes[i].reprojectLandmark(current_shape);
        //cout << current_shape(0,0) << " " << current_shape(0,1) << endl;
        cv::Mat_<double> res = cas_load.Predict(images_paths[i], current_shape, bboxes[i]);   
        //cout << res(0,0) << res(0,1) << endl;     
        string pts_path = base + ".pts";
        string bbox_path = base + ".bbox";
        record_pts(pts_path, res);
        
        Mat img_draw = drawLandmark(images_paths[i], bboxes[i], res); 
        imwrite(OUTPUT + basename(images_paths[i]), img_draw);
        fout << images_paths[i] << " " << bbox_path << " " << pts_path << endl;
    }
    fout.close();

    t_end = time(NULL); 
    time_consume += difftime(t_end, t_start);
    cout << "the total time of processing " << bboxes.size() <<  " images is " << setprecision(3) << time_consume << " s." << endl;
}

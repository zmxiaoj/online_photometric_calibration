#include "StandardIncludes.h"

#include "Tracker.h"
#include "Database.h"
#include "GainRobustTracker.h"


int main(int argc, char** argv) 
{
    std::ifstream exposure_gt_file_handle("../data/times.txt");
    std::vector<double> gt_exp_times;   
    std::string line;

    auto split = [](const std::string &s, char delim) -> std::vector<std::string>{
        std::stringstream ss(s);
        std::string item;
        std::vector<std::string> tokens;
        while (std::getline(ss, item, delim)) {
            tokens.push_back(item);
        }
        return tokens;
    };
    double min_time = FLT_MAX;
    double max_time = FLT_MIN;

    while (getline(exposure_gt_file_handle, line)) {
        std::string delimiter = " ";
        std::vector<std::string> split_line = split(line,' ');
        double gt_exp_time = stod(split_line.at(split_line.size()-1));
        gt_exp_times.push_back(gt_exp_time);
        if(gt_exp_time < min_time)
            min_time = gt_exp_time;
        if(gt_exp_time > max_time)
            max_time = gt_exp_time;
    }    
    std::cout << "Ground truth exposure time file successfully load" << std::endl;

    std::string saved_path = "../results/";

    Database database(1280, 1024);
    Tracker tracker(3, 200, 2, &database);
    
    cv::Mat first_image = cv::imread("../data/00000.jpg", cv::IMREAD_GRAYSCALE);
    std::string orignal_saved_path = saved_path + "first_image.png";
    cv::imwrite(orignal_saved_path, first_image);

    double first_gt_exp_time = gt_exp_times[0];
    
    // Start Tracker::trackNewFrame Function
    cv::Mat first_gradient_image;
    std::cout << "Start computing gradient image and visualize it" << std::endl;
    tracker.computeGradientImageAndVisualize(first_image, first_gradient_image, saved_path);
    std::cout << "Finish computing gradient image and visualize it" << std::endl;

    std::vector<cv::Point2f> old_features;
    std::cout << "Start extracting features for first frame and visualize" << std::endl;
    tracker.initialFeatureExtractionAndVisualize(first_image, first_gradient_image, first_gt_exp_time, saved_path);
    std::cout << "Finish extracting features for first frame and visualize" << std::endl;

    int patch_size = 2, pyramid_levels = 3;
    GainRobustTracker gain_robust_klt_tracker(patch_size, pyramid_levels);
    cv::Mat last_image = database.fetchActiveImage();
    std::vector<cv::Point2f> last_frame_features = database.fetchActiveFeatureLocations();
    
    std::vector<cv::Point2f> tracked_points_new_frame;
    std::vector<int> tracked_point_status_int;

    cv::Mat new_frame = cv::imread("../data/00002.jpg", cv::IMREAD_GRAYSCALE);

    std::cout << "Start tracking new frame forward" << std::endl;
    std::string forward_saved_path = saved_path + "forward/";
    gain_robust_klt_tracker.trackImagePyramidsAndVisualize( last_image, new_frame, last_frame_features, 
                                                            tracked_points_new_frame, tracked_point_status_int,
                                                            forward_saved_path);
    std::cout << "Finish tracking new frame forward" << std::endl;

    GainRobustTracker gain_robust_klt_tracker_backward(patch_size, pyramid_levels);
    std::vector<cv::Point2f> tracked_points_backward;
    std::vector<int> tracked_point_status_int_backward;

    std::cout << "Start tracking old frame backward" << std::endl;
    std::string backward_saved_path = saved_path + "backward/";
    gain_robust_klt_tracker_backward.trackImagePyramidsAndVisualize(new_frame, last_image, tracked_points_new_frame, 
                                                                    tracked_points_backward, tracked_point_status_int_backward,
                                                                    backward_saved_path);
    std::cout << "Finish tracking old frame backward" << std::endl;

    // Finish Tracker::trackNewFrame Function
    // Ignore some part of Tracker::trackNewFrame, e.g., extract_new_features


    return 0;
}
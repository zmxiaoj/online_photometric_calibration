#include "Tracker.h"


int main(int argc, char** argv) 
{
    std::string saved_path = "../results/";
    Tracker tracker(3, 100, 3, nullptr);
    cv::Mat input_image = cv::imread("../data/00000.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat gradient_image;
    tracker.computeGradientImageAndVisualize(input_image, gradient_image, saved_path);

    return 0;
}
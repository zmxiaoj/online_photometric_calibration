//
//  GainRobustTracker.cpp
//  OnlinePhotometricCalibration
//
//  Created by Paul on 17.11.17.
//  Copyright (c) 2017-2018 Paul Bergmann and co-authors. All rights reserved.
//
//  See LICENSE.txt
//

#include "GainRobustTracker.h"

GainRobustTracker::GainRobustTracker(int patch_size,int pyramid_levels)
{
    // Initialize patch size and pyramid levels
    m_patch_size = patch_size;
    m_pyramid_levels = pyramid_levels;
}

/**
 * @brief 
 * 
 * @return average exposure ratio estimate 
 */
// Todo: change frame_1 frame 2 to ref (or const ref), pts_1 to ref
double GainRobustTracker::trackImagePyramids(cv::Mat frame_1,
                                             cv::Mat frame_2,
                                             std::vector<cv::Point2f> pts_1,
                                             std::vector<cv::Point2f>& pts_2,
                                             std::vector<int>& point_status)
{
    // All points valid in the beginning of tracking
    std::vector<int> point_validity;

    point_validity.reserve(pts_1.size());
    
    for(int i = 0;i < pts_1.size();i++)
    {
        point_validity.push_back(1);
    }
    
    // Calculate image pyramid of frame 1 and frame 2
    std::vector<cv::Mat> new_pyramid;
    cv::buildPyramid(frame_2, new_pyramid, m_pyramid_levels);
    
    std::vector<cv::Mat> old_pyramid;
    cv::buildPyramid(frame_1, old_pyramid, m_pyramid_levels);
    
    // Temporary vector to update tracking estiamtes over time
    std::vector<cv::Point2f> tracking_estimates = pts_1;
    
    double all_exp_estimates = 0.0;
    int nr_estimates = 0;
    
    // Iterate all pyramid levels and perform gain robust KLT on each level (coarse to fine)
    for(int level = (int)new_pyramid.size()-1;level >= 0;level--)
    {
        // Scale the input points and tracking estimates to the current pyramid level
        std::vector<cv::Point2f> scaled_tracked_points;
        std::vector<cv::Point2f> scaled_tracking_estimates;
        for(int i = 0;i < pts_1.size();i++)
        {
            cv::Point2f scaled_point;
            scaled_point.x = (float)(pts_1.at(i).x/pow(2,level));
            scaled_point.y = (float)(pts_1.at(i).y/pow(2,level));
            scaled_tracked_points.push_back(scaled_point);
            
            cv::Point2f scaled_estimate;
            scaled_estimate.x = (float)(tracking_estimates.at(i).x/pow(2,level));
            scaled_estimate.y = (float)(tracking_estimates.at(i).y/pow(2,level));
            scaled_tracking_estimates.push_back(scaled_estimate);
        }
        
        // Perform tracking on current level
        double exp_estimate = trackImageExposurePyr(old_pyramid.at(level),
                                                    new_pyramid.at(level),
                                                    scaled_tracked_points,
                                                    scaled_tracking_estimates,
                                                    point_validity);
        
        // Optional: Do something with the estimated exposure ratio
        // std::cout << "Estimated exposure ratio of current level: " << exp_estimate << std::endl;
        
        // Average estimates of each level later
        all_exp_estimates += exp_estimate;
        nr_estimates++;
        
        // Update the current tracking result by scaling down to pyramid level 0
        for(int i = 0;i < scaled_tracking_estimates.size();i++)
        {
            if(point_validity.at(i) == 0)
                continue;
            
            cv::Point2f scaled_point;
            scaled_point.x = (float)(scaled_tracking_estimates.at(i).x*pow(2,level));
            scaled_point.y = (float)(scaled_tracking_estimates.at(i).y*pow(2,level));
            
            tracking_estimates.at(i) = scaled_point;
        }
    }
    
    // Write result to output vectors passed by reference
    pts_2 = tracking_estimates;
    point_status = point_validity;
    
    // Average exposure ratio estimate
    double overall_exp_estimate = all_exp_estimates / nr_estimates;
    return overall_exp_estimate;
}

// track image by pyramids & visualize 
double GainRobustTracker::trackImagePyramidsAndVisualize(cv::Mat source_frame,
                                                         cv::Mat target_frame,
                                                         std::vector<cv::Point2f> pts_1,
                                                         std::vector<cv::Point2f>& pts_2,
                                                         std::vector<int>& point_status,
                                                         std::string saved_path)
{
    // All points valid in the beginning of tracking
    std::cout << "Initialize point validity vector" << std::endl;
    std::vector<int> point_validity;

    point_validity.reserve(pts_1.size());
    
    for(int i = 0;i < pts_1.size();i++)
    {
        point_validity.push_back(1);
    }
    

    // Calculate image pyramid of frame 1 and frame 2
    std::cout << "Start constructing image pyramids" << std::endl;
    std::vector<cv::Mat> target_frame_pyramid;
    cv::buildPyramid(target_frame, target_frame_pyramid, m_pyramid_levels);
    for (int i = 0; i < target_frame_pyramid.size(); i++) {
        std::string target_frame_pyramid_saved_path = saved_path + "target_frame_pyramid_" + std::to_string(i) + ".png";
        // cv::imshow("target_frame_pyramid", target_frame_pyramid.at(i));
        // cv::waitKey(0);
        cv::imwrite(target_frame_pyramid_saved_path, target_frame_pyramid.at(i));
        std::cout << "Save target frame pyramid images to " << target_frame_pyramid_saved_path << std::endl;
    }


    std::vector<cv::Mat> source_frame_pyramid;
    cv::buildPyramid(source_frame, source_frame_pyramid, m_pyramid_levels);
    for (int i = 0; i < source_frame_pyramid.size(); i++) {
        std::string source_frame_pyramid_saved_path = saved_path + "source_frame_pyramid_" + std::to_string(i) + ".png";
        // cv::imshow("source_frame_pyramid", source_frame_pyramid.at(i));
        // cv::waitKey(0);
        cv::imwrite(source_frame_pyramid_saved_path, source_frame_pyramid.at(i));
        std::cout << "Save source frame pyramid images to " << source_frame_pyramid_saved_path << std::endl;
    }
    
    std::cout << "Finish constructing image pyramids" << std::endl;

    // Temporary vector to update tracking estiamtes over time
    // Intialize tracking point estimates with input points
    std::cout << "Start tracking & estimating exposure time" << std::endl;
    std::vector<cv::Point2f> tracking_estimates = pts_1;
    
    double all_exp_estimates = 0.0;
    int nr_estimates = 0;
    
    // Iterate all pyramid levels and perform gain robust KLT on each level
    // from top to bottom (coarse to fine)
    std::cout << "Start iterating all pyramid levels" << std::endl;
    for(int level = (int)target_frame_pyramid.size()-1;level >= 0;level--)
    {
        // Scale the input points and tracking estimates to the current pyramid level
        std::cout << "Start processing " << level << " pyramid level" << std::endl;
        std::vector<cv::Point2f> scaled_tracked_points;
        std::vector<cv::Point2f> scaled_tracking_estimates;
        for(int i = 0;i < pts_1.size();i++)
        {
            cv::Point2f scaled_point;
            scaled_point.x = (float)(pts_1.at(i).x/pow(2,level));
            scaled_point.y = (float)(pts_1.at(i).y/pow(2,level));
            scaled_tracked_points.push_back(scaled_point);
            
            cv::Point2f scaled_estimate;
            scaled_estimate.x = (float)(tracking_estimates.at(i).x/pow(2,level));
            scaled_estimate.y = (float)(tracking_estimates.at(i).y/pow(2,level));
            scaled_tracking_estimates.push_back(scaled_estimate);
        }
        
        // Perform tracking on current level
        std::cout << "Start tracking " << level << " pyramid level" << std::endl;
        // Gain Robust KLT to find tracking points in frame 2
        double exp_estimate = trackImageExposurePyrAndVisualize(source_frame_pyramid.at(level),
                                                                target_frame_pyramid.at(level),
                                                                scaled_tracked_points,
                                                                scaled_tracking_estimates,
                                                                point_validity,
                                                                saved_path);

        // Optional: Do something with the estimated exposure ratio
        // std::cout << "Estimated exposure ratio of current level: " << exp_estimate << std::endl;
        
        // Average estimates of each level later
        all_exp_estimates += exp_estimate;
        nr_estimates++;
        
        // Update the current tracking result by scaling down to pyramid level 0 from current level
        for(int i = 0;i < scaled_tracking_estimates.size();i++)
        {
            if(point_validity.at(i) == 0)
                continue;
            
            cv::Point2f scaled_point;
            scaled_point.x = (float)(scaled_tracking_estimates.at(i).x*pow(2,level));
            scaled_point.y = (float)(scaled_tracking_estimates.at(i).y*pow(2,level));
            
            tracking_estimates.at(i) = scaled_point;
        }
    }
    
    std::cout << "Finish iterating all pyramid levels" << std::endl;

    // Write result to output vectors passed by reference
    pts_2 = tracking_estimates;
    point_status = point_validity;

    // TODO: Visualize the tracking results, and compare results with the vanilla KLT
    cv::Mat target_frame_tracking_results;
    cv::cvtColor(target_frame, target_frame_tracking_results, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < point_validity.size(); i++) {
        if (point_validity.at(i) == 0)
            continue;
        cv::circle(target_frame_tracking_results, pts_1.at(i), 2, cv::Scalar(128, 0, 128), -1);
        cv::circle(target_frame_tracking_results, pts_2.at(i), 2, cv::Scalar(255, 165, 0), -1);
        cv::line(target_frame_tracking_results, pts_1.at(i), pts_2.at(i), cv::Scalar(0, 255, 255), 1);
    }
    std::string target_frame_tracking_results_saved_path = saved_path + "target_frame_tracking_results.png";
    // cv::imshow("target_frame_tracking_results", target_frame_tracking_results);
    // cv::waitKey(0);
    cv::imwrite(target_frame_tracking_results_saved_path, target_frame_tracking_results);
    std::cout << "Save target frame tracking results to " << target_frame_tracking_results_saved_path << std::endl;

    // Average exposure ratio estimate
    double overall_exp_estimate = all_exp_estimates / nr_estimates;
    return overall_exp_estimate;
}

/**
 * @brief a reference on the meaning of the optimization variables and the overall concept of this function
 *        refer to the photometric calibration paper 
 *        introducing gain robust KLT tracking by Kim et al.
 */
 // Todo: change Mat and vector to ref
double GainRobustTracker::trackImageExposurePyr(cv::Mat old_image,
                                                cv::Mat new_image,
                                                std::vector<cv::Point2f> input_points,
                                                std::vector<cv::Point2f>& output_points,
                                                std::vector<int>& point_validity)
{
    // Number of points to track
    int nr_points = static_cast<int>(input_points.size());
    
    // Updated point locations which are updated throughout the iterations
    if(output_points.size() == 0)
    {
        output_points = input_points;
    }
    else if(output_points.size() != input_points.size())
    {
        std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
        return -1;
    }
    
    // Input image dimensions
    int image_rows = new_image.rows;
    int image_cols = new_image.cols;
    
    // Final exposure time estimate
    double K_total = 0.0;
    
    for(int round = 0;round < 1;round++)
    {
        // Get the currently valid points
        int nr_valid_points = getNrValidPoints(point_validity);
        
        // Allocate space for W,V matrices
        cv::Mat W(2*nr_valid_points,1,CV_64F,0.0);
        cv::Mat V(2*nr_valid_points,1,CV_64F,0.0);
        
        // Allocate space for U_INV and the original Us
        cv::Mat U_INV(2*nr_valid_points,2*nr_valid_points,CV_64F,0.0);
        std::vector<cv::Mat> Us;
        
        double lambda = 0;
        double m = 0;

        int absolute_point_index = -1;
        
        for(int p = 0;p < input_points.size();p++)
        {
            if(point_validity.at(p) == 0)
            {
                continue;
            }
            
            absolute_point_index++;
            
            // Build U matrix
            cv::Mat U(2,2, CV_64F, 0.0);
            
            // Bilinear image interpolation
            cv::Mat patch_intensities_1;
            cv::Mat patch_intensities_2;
            int absolute_patch_size = ((m_patch_size+1)*2+1);  // Todo: why m_patch_size+1?
            cv::getRectSubPix(new_image, cv::Size(absolute_patch_size,absolute_patch_size), output_points.at(p), patch_intensities_2,CV_32F);
            cv::getRectSubPix(old_image, cv::Size(absolute_patch_size,absolute_patch_size), input_points.at(p), patch_intensities_1,CV_32F);
            
            // Go through image patch around this point
            for(int r = 0; r < 2*m_patch_size+1;r++)
            {
                for(int c = 0; c < 2*m_patch_size+1;c++)
                {
                    // Fetch patch intensity values
                    double i_frame_1 = patch_intensities_1.at<float>(1+r,1+c);
                    double i_frame_2 = patch_intensities_2.at<float>(1+r,1+c);
                    
                    if(i_frame_1 < 1)
                        i_frame_1 = 1;
                    if(i_frame_2 < 1)
                        i_frame_2 = 1;
                    
                    // Estimate patch gradient values
                    double grad_1_x = (patch_intensities_1.at<float>(1+r,1+c+1) - patch_intensities_1.at<float>(1+r,1+c-1))/2;
                    double grad_1_y = (patch_intensities_1.at<float>(1+r+1,1+c) - patch_intensities_1.at<float>(1+r-1,1+c))/2;
                    
                    double grad_2_x = (patch_intensities_2.at<float>(1+r,1+c+1) - patch_intensities_2.at<float>(1+r,1+c-1))/2;
                    double grad_2_y = (patch_intensities_2.at<float>(1+r+1,1+c) - patch_intensities_2.at<float>(1+r-1,1+c))/2;
                    
                    double a = (1.0/i_frame_2)*grad_2_x + (1.0/i_frame_1)*grad_1_x;
                    double b = (1.0/i_frame_2)*grad_2_y + (1.0/i_frame_1)*grad_1_y;
                    double beta = log(i_frame_2/255.0) - log(i_frame_1/255.0);
                    
                    U.at<double>(0,0) += 0.5*a*a;
                    U.at<double>(1,0) += 0.5*a*b;
                    U.at<double>(0,1) += 0.5*a*b;
                    U.at<double>(1,1) += 0.5*b*b;
                    
                    W.at<double>(2*absolute_point_index,0)   -= a;
                    W.at<double>(2*absolute_point_index+1,0) -= b;
                    
                    V.at<double>(2*absolute_point_index,0)   -= beta*a;
                    V.at<double>(2*absolute_point_index+1,0) -= beta*b;
                    
                    lambda += 2;
                    m += 2*beta;
                }
            }
            
            //Back up U for re-substitution
            Us.push_back(U);
            
            //Invert matrix U for this point and write it to diagonal of overall U_INV matrix
            cv::Mat U_INV_p = U.inv();
            //std::cout << cv::determinant(U_INV_p) << std::endl;
            //std::cout << U_INV_p << std::endl;
            //std::cout << U << std::endl;
            
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index) = U_INV_p.at<double>(0,0);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index) = U_INV_p.at<double>(1,0);
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index+1) = U_INV_p.at<double>(0,1);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index+1) = U_INV_p.at<double>(1,1);
        }

        // Todo: check if opencv utilizes the sparsity of U
        //solve for the exposure
        cv::Mat K_MAT;
        cv::solve(-W.t()*U_INV*W+lambda, -W.t()*U_INV*V+m, K_MAT);
        double K = K_MAT.at<double>(0,0);
        
        //std::cout << -W.t()*U_INV*W+lambda << std::endl;
        //std::cout << -W.t()*U_INV*V+m << std::endl;
        //std::cout << K_MAT << std::endl;
        
        // Solve for the displacements
        absolute_point_index = -1;
        for(int p = 0;p < nr_points;p++)
        {
            if(point_validity.at(p) == 0)
                continue;
            
            absolute_point_index++;
            
            cv::Mat U_p = Us.at(absolute_point_index);
            cv::Mat V_p = V(cv::Rect(0,2*absolute_point_index,1,2));
            cv::Mat W_p = W(cv::Rect(0,2*absolute_point_index,1,2));
            
            cv::Mat displacement;
            cv::solve(U_p, V_p - K*W_p, displacement);
            
            //std::cout << displacement << std::endl;
            
            output_points.at(p).x += displacement.at<double>(0,0);
            output_points.at(p).y += displacement.at<double>(1,0);
            
            // Filter out this point if too close at the boundaries
            int filter_margin = 2;
            double x = output_points.at(p).x;
            double y = output_points.at(p).y;
            // Todo: the latter two should be ">=" ?
            if(x < filter_margin || y < filter_margin || x > image_cols-filter_margin || y > image_rows-filter_margin)
            {
                point_validity.at(p) = 0;
            }
        }
        
        K_total += K;
    }
    
    return exp(K_total);
}

/**
 * @brief Gain Robust KLT Tracking with exposure estimation and image pyramids
 *        This func can get the position of tracking points in the second image 
 *        And output the exposure time difference K between the two images
 *        Reference paper: Joint Radiometric Calibration and Feature Tracking for an Adaptive Stereo System, Kim et al.
 * 
 * @param [in ] old_image 
 * @param [in ] new_image 
 * @param [in ] input_points 
 * @param [out] output_points 
 * @param [out] point_validity 
 * @param [in ] saved_path 
 * @return double 
 */
double GainRobustTracker::trackImageExposurePyrAndVisualize(cv::Mat old_image,
                                                            cv::Mat new_image,
                                                            std::vector<cv::Point2f> input_points,
                                                            std::vector<cv::Point2f>& output_points,
                                                            std::vector<int>& point_validity,
                                                            std::string saved_path)
{
    // Number of points to track
    int nr_points = static_cast<int>(input_points.size());
    // std::cout << "Number of points to track: " << nr_points << std::endl;
    
    // Updated point locations which are updated throughout the iterations
    if(output_points.size() == 0)
    {
        output_points = input_points;
    }
    else if(output_points.size() != input_points.size())
    {
        std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
        return -1;
    }
    
    // Input image dimensions
    int image_rows = new_image.rows;
    int image_cols = new_image.cols;
    
    // Final exposure time estimate
    double K_total = 0.0;
    
    // TODO: Try to replace Opencv matrix calculation with Eigen, and compare the precision
    for(int round = 0;round < 1;round++)
    {
        // Get the currently valid points
        int nr_valid_points = getNrValidPoints(point_validity);
        // std::cout << "Number of valid points: " << nr_valid_points << std::endl;
        
        // Allocate space for W,V matrices
        // W_2nx1
        cv::Mat W(2*nr_valid_points,1,CV_64F,0.0);
        // V_2nx1
        cv::Mat V(2*nr_valid_points,1,CV_64F,0.0);
        
        // Allocate space for U_INV and the original Us
        // U_INV_2nx2n
        cv::Mat U_INV(2*nr_valid_points,2*nr_valid_points,CV_64F,0.0);
        std::vector<cv::Mat> Us;
        
        double lambda = 0;
        double m = 0;

        int absolute_point_index = -1;
        
        for(int p = 0;p < input_points.size();p++)
        {
            if(point_validity.at(p) == 0)
            {
                continue;
            }
            
            absolute_point_index++;
            
            // Build U matrix
            // U_2x2
            cv::Mat U(2,2, CV_64F, 0.0);
            
            // Bilinear image interpolation
            cv::Mat patch_intensities_1;
            cv::Mat patch_intensities_2;
            // m_patch_size = 2
            int absolute_patch_size = ((m_patch_size+1)*2+1);  // Todo: why m_patch_size+1?
            // get 7x7 patch from new_image&old_image
            cv::getRectSubPix(new_image, cv::Size(absolute_patch_size,absolute_patch_size), output_points.at(p), patch_intensities_2,CV_32F);
            cv::getRectSubPix(old_image, cv::Size(absolute_patch_size,absolute_patch_size), input_points.at(p), patch_intensities_1,CV_32F);
            
            // std::cout << "Size of patch_intensities_1: " << patch_intensities_1.size() << std::endl;
            // std::cout << "Size of patch_intensities_2: " << patch_intensities_2.size() << std::endl;
            // // svae patch_intensities_1&patch_intensities_2
            // std::string patch_intensities_1_saved_path = saved_path + "patch_intensities_1_p" + std::to_string(p) + ".png";
            // // cv::imshow("patch_intensities_1", patch_intensities_1);
            // cv::waitKey(0);
            // cv::imwrite(patch_intensities_1_saved_path, patch_intensities_1);
            // std::cout << "Save patch_intensities_1 to " << patch_intensities_1_saved_path << std::endl;
            // std::string patch_intensities_2_saved_path = saved_path + "patch_intensities_2_p" + std::to_string(p) + ".png";
            // // cv::imshow("patch_intensities_2", patch_intensities_2);
            // cv::waitKey(0);
            // cv::imwrite(patch_intensities_2_saved_path, patch_intensities_2);
            // std::cout << "Save patch_intensities_2 to " << patch_intensities_2_saved_path << std::endl;

            int point_patch_size = 2*m_patch_size+1;
            // std::cout << "Go through patch around this point, size is " << point_patch_size << "*" << point_patch_size << std::endl;
            
            // Go through image patch around this point
            //* I = f(e*V(r)*L)
            //* g(I) = ln(f_inv(I)) 
            //* g(I) = lne + lnV(r) + lnL
            //* g(J) - g(I) = K, K = ln(e_2/e_1) 
            
            //* [ U       W  ] * [ dx ] = [ v ]
            //* [ W^T  lambda]   [  K ]   [ m ]
            for(int r = 0; r < 2*m_patch_size+1;r++)
            {
                for(int c = 0; c < 2*m_patch_size+1;c++)
                {
                    // Fetch patch intensity values
                    // Patch intensity removed response function "f()"& vignette function
                    // f_inv(I)/V(r) 
                    double i_frame_1 = patch_intensities_1.at<float>(1+r,1+c);
                    double i_frame_2 = patch_intensities_2.at<float>(1+r,1+c);
                    
                    // clamp
                    if(i_frame_1 < 1)
                        i_frame_1 = 1;
                    if(i_frame_2 < 1)
                        i_frame_2 = 1;
                    
                    // Estimate patch gradient values
                    // central difference, calculate gradient in x&y direction of patch_intensities_1&patch_intensities_2
                    double grad_1_x = (patch_intensities_1.at<float>(1+r,1+c+1) - patch_intensities_1.at<float>(1+r,1+c-1))/2;
                    double grad_1_y = (patch_intensities_1.at<float>(1+r+1,1+c) - patch_intensities_1.at<float>(1+r-1,1+c))/2;
                    
                    double grad_2_x = (patch_intensities_2.at<float>(1+r,1+c+1) - patch_intensities_2.at<float>(1+r,1+c-1))/2;
                    double grad_2_y = (patch_intensities_2.at<float>(1+r+1,1+c) - patch_intensities_2.at<float>(1+r-1,1+c))/2;
                    
                    double a = (1.0/i_frame_2)*grad_2_x + (1.0/i_frame_1)*grad_1_x;
                    double b = (1.0/i_frame_2)*grad_2_y + (1.0/i_frame_1)*grad_1_y;
                    // Normalizing the intensity value to [0,1] has the same result 
                    double beta = log(i_frame_2/255.0) - log(i_frame_1/255.0);
                    
                    // U_2x2
                    U.at<double>(0,0) += 0.5*a*a;
                    U.at<double>(1,0) += 0.5*a*b;
                    U.at<double>(0,1) += 0.5*a*b;
                    U.at<double>(1,1) += 0.5*b*b;
                    
                    W.at<double>(2*absolute_point_index,0)   -= a;
                    W.at<double>(2*absolute_point_index+1,0) -= b;
                    
                    V.at<double>(2*absolute_point_index,0)   -= beta*a;
                    V.at<double>(2*absolute_point_index+1,0) -= beta*b;
                    
                    lambda += 2;
                    m += 2*beta;
                }
            }
            
            //Back up U for re-substitution
            Us.push_back(U);
            
            //Invert matrix U for this point and write it to diagonal of overall U_INV matrix
            cv::Mat U_INV_p = U.inv();
            //std::cout << cv::determinant(U_INV_p) << std::endl;
            //std::cout << U_INV_p << std::endl;
            //std::cout << U << std::endl;
            
            // Save U_inv for schur complement
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index) = U_INV_p.at<double>(0,0);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index) = U_INV_p.at<double>(1,0);
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index+1) = U_INV_p.at<double>(0,1);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index+1) = U_INV_p.at<double>(1,1);
        }

        // Todo: check if opencv utilizes the sparsity of U
        //solve for the exposure
        cv::Mat K_MAT;
        cv::solve(-W.t()*U_INV*W+lambda, -W.t()*U_INV*V+m, K_MAT);
        double K = K_MAT.at<double>(0,0);
        
        //std::cout << -W.t()*U_INV*W+lambda << std::endl;
        //std::cout << -W.t()*U_INV*V+m << std::endl;
        //std::cout << K_MAT << std::endl;
        
        // Solve for the displacements
        absolute_point_index = -1;
        for(int p = 0;p < nr_points;p++)
        {
            if(point_validity.at(p) == 0)
                continue;
            
            absolute_point_index++;
            
            cv::Mat U_p = Us.at(absolute_point_index);
            cv::Mat V_p = V(cv::Rect(0,2*absolute_point_index,1,2));
            cv::Mat W_p = W(cv::Rect(0,2*absolute_point_index,1,2));
            
            cv::Mat displacement;
            cv::solve(U_p, V_p - K*W_p, displacement);
            
            //std::cout << displacement << std::endl;
            
            output_points.at(p).x += displacement.at<double>(0,0);
            output_points.at(p).y += displacement.at<double>(1,0);
            
            // Filter out this point if too close at the boundaries
            int filter_margin = 2;
            double x = output_points.at(p).x;
            double y = output_points.at(p).y;
            // Todo: the latter two should be ">=" ?
            if(x < filter_margin || y < filter_margin || x > image_cols-filter_margin || y > image_rows-filter_margin)
            {
                point_validity.at(p) = 0;
            }
        }
        
        K_total += K;
    }
    
    return exp(K_total);
}

/**
 * @brief 
 * 
 * @param validity_vector 
 * @return int 
 */
int GainRobustTracker::getNrValidPoints(std::vector<int> validity_vector)
{
    // Simply sum up the validity vector
    int result = 0;
    for(int i = 0;i < validity_vector.size();i++)
    {
        result += validity_vector.at(i);
    }
    return result;
}

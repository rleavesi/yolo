#ifndef MY_DETECTOR_H
#define MY_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <omp.h>

class Yolo {
public:
    Yolo();

    ~Yolo();

    typedef struct {
        float prob;
        int id;
        cv::Rect rect;
    } Object;

    bool Process(const cv::Mat& src,std::vector<Object> &objects);
    
    static void Draw(cv::Mat& src, std::vector<Object> detected_objects);


private:

    void Init();

    double Sigmoid(double x);
    
    cv::Rect Recover(const cv::Rect& rect);

    std::vector<int> GetAnchors(int net_grid);
    
    void Parse(const InferenceEngine::Blob::Ptr blob,int net_grid,float cof_threshold,std::vector<cv::Rect>& o_rect,std::vector<float>& o_rect_cof, std::vector<int>& o_res_id);    
    
    InferenceEngine::ExecutableNetwork _network;
    
    InferenceEngine::OutputsDataMap _outputinfo;
    
    std::string _input_name;

    float _ratio_rows;

    float _ratio_cols;
    
    // set param    
    std::string _input_model; 			// 训练好的模型位置 
    size_t _n_class;					// 分的类num，装甲板是36
    size_t _imgsz;						// 输入网络图像的大小，这里是512x512
    double _cof_threshold;              // 置信度阈值,计算方法是框置信度乘以物品种类置信度，小于该值会被过滤，这里设0.4
    double _nms_area_threshold;         // nms最小重叠面积阈值(框和框之间的重叠面积)，这里设0.1
    bool _use_gpu;                      // 是否使用CPU核显
};


#endif
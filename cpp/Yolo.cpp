#include "Yolo.hpp"

static omp_lock_t lock_use;
static omp_lock_t s_lock;

Yolo::Yolo():   _n_class(36),
                _imgsz(512),
                _input_model("../last.xml"),
                _cof_threshold(0.4),
                _nms_area_threshold(0.1),
                _use_gpu(true)
                { Init(); }

Yolo::~Yolo() {}

void Yolo::Init() {
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(_input_model); 
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(InferenceEngine::Precision::FP32);
    input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    _outputinfo = InferenceEngine::OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32); }
    if(_use_gpu) _network =  ie.LoadNetwork(cnnNetwork, "GPU");
    else _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    omp_init_lock(&lock_use); // 初始化互斥锁
    omp_init_lock(&s_lock); // 初始化互斥锁
}


bool Yolo::Process(const cv::Mat& dst,std::vector<Object> &objects) {
    if(dst.empty()) return false;

    // 计算图像缩小率方便画图复原
    _ratio_rows = (float)(_imgsz) / (float)(dst.rows);
    _ratio_cols = (float)(_imgsz) / (float)(dst.cols);

    // 图像预处理
    cv::Mat src;
    cv::resize(dst,src,cv::Size(_imgsz,_imgsz));
    cv::cvtColor(src,src,cv::COLOR_BGR2RGB);

    // 将图像数据转化为openvino加速需要的tensor(blob)
    size_t img_size = _imgsz * _imgsz;
    InferenceEngine::InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    InferenceEngine::Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    
    // nchw 
    omp_set_num_threads(4); // 线程加速1
    #pragma omp parallel for
    for(size_t row = 0; row < _imgsz; row++){
        for(size_t col = 0; col < _imgsz; col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*_imgsz + col] = float(src.at<cv::Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }

    // 推理
    infer_request->Infer();

    // 推理结束后需要获取的数据
    std::vector<cv::Rect> origin_rect;
    std::vector<float> origin_rect_cof;
    std::vector<int> origin_res_id;

    // 获取数据时大规模计算之前先收集指针
    std::vector<InferenceEngine::Blob::Ptr> blobs;
    for (auto &output : _outputinfo) {
        if(output.first == "output")    break;
        InferenceEngine::Blob::Ptr blob = infer_request->GetBlob(output.first);
        blobs.push_back(blob);
    }

    // 这里的三个数对应网络三个输出的三个锚框的大小(Tips:使用netron可视化网络，查看output)
    int s[3] = {64,32,16};

    omp_set_num_threads(3); // 线程加速2
    #pragma omp parallel for
    for(int i = 0; i < blobs.size(); i++){
        float th = 0.5;
        // 小目标置信度阈值
        if(i == 0)
            th = 0.55;
        // 大目标置信度阈值
        else if(i == 1)
            th = 0.45;
        else if(i == 2)
            th = 0.40;

        std::vector<cv::Rect> origin_rect_temp;
        std::vector<float> origin_rect_cof_temp;
        std::vector<int> origin_res_id_temp;

        // 获取所有信息
        Parse(blobs[i],s[i],th,origin_rect_temp,origin_rect_cof_temp,origin_res_id_temp);
        
        // 加锁
        omp_set_lock(&lock_use); // 互斥器
        origin_rect.insert(origin_rect.end(),origin_rect_temp.begin(),origin_rect_temp.end());
        origin_rect_cof.insert(origin_rect_cof.end(),origin_rect_cof_temp.begin(),origin_rect_cof_temp.end());
        origin_res_id.insert(origin_res_id.end(),origin_res_id_temp.begin(),origin_res_id_temp.end());
        omp_unset_lock(&lock_use); // 释放
    }

    // NMS后处理
    std::vector<int> final_id;
    cv::dnn::NMSBoxes(origin_rect,origin_rect_cof,_cof_threshold,_nms_area_threshold,final_id);

    // 获取最终结果
    for(int i = 0; i < final_id.size(); i++) {
        objects.push_back(Object{
            .prob = origin_rect_cof[final_id[i]],
            .id = origin_res_id[final_id[i]],
            .rect = Recover(origin_rect[final_id[i]])
        });
    }
    return true;
}

double Yolo::Sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

std::vector<int> Yolo::GetAnchors(int net_grid) {
    std::vector<int> anchors(6);

    // 这里数据对应Python训练时的model/yolo.yaml
    int a64[6] = {10,15, 13,24, 18,19};
    int a32[6] = {20,44, 29,26, 42,43};
    int a16[6] = {64,115, 139,111, 309,379}; 
    if(net_grid == 64){
        anchors.insert(anchors.begin(),a64,a64 + 6);
    }
    else if(net_grid == 32){
        anchors.insert(anchors.begin(),a32,a32 + 6);
    }
    else if(net_grid == 16){
        anchors.insert(anchors.begin(),a16,a16 + 6);
    }
    return anchors;
}

void Yolo::Parse(const InferenceEngine::Blob::Ptr blob,int net_grid,float cof_threshold,std::vector<cv::Rect>& o_rect,std::vector<float>& o_rect_cof, std::vector<int>& o_res_id) {
    std::vector<int> anchors = GetAnchors(net_grid);
    InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    size_t item_size = _n_class + 5;
    size_t anchor_n = 3;
    for(int n = 0; n < anchor_n; n++)
        for(int i = 0;i < net_grid; i++)
            for(int j = 0;j < net_grid; j++) {
                double box_prob = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size+ 4];
                box_prob = Sigmoid(box_prob);
                
                // 框置信度不满足则整体置信度不满足
                if(box_prob < cof_threshold)
                    continue;
                
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 0];
                double y = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 1];
                double w = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 2];
                double h = output_blob[n * net_grid * net_grid * item_size + i * net_grid * item_size + j * item_size + 3];
               
                double max_prob = 0;
                int idx = 0;
                for(int t = 5;t < item_size;t++){
                    double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                    tp = Sigmoid(tp);
                    if(tp > max_prob){
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = box_prob * max_prob;                
                
                // 对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if(cof < cof_threshold)
                    continue;

                x = (Sigmoid(x) * 2 - 0.5 + j) * 512.0f / net_grid;
                y = (Sigmoid(y) * 2 - 0.5 + i) * 512.0f / net_grid;
                w = pow(Sigmoid(w) * 2, 2) * anchors[n * 2];
                h = pow(Sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

                double r_x = x - w / 2;
                double r_y = y - h / 2;
                cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                o_res_id.push_back(idx - 5);
            }
}

cv::Rect Yolo::Recover(const cv::Rect& rect) {
    float x = rect.x / _ratio_cols;
    float y = rect.y / _ratio_rows;
    float w = rect.width / _ratio_cols;
    float h = rect.height / _ratio_rows;
    return cv::Rect(x,y,w,h);
}


void Yolo::Draw(cv::Mat& src, std::vector<Object> detected_objects) {
    for(int i = 0;i < detected_objects.size(); i++){
        int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        cv::Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(src, rect, cv::Scalar(0, 0, 255),1, cv::LINE_8,0);
    }
}


int main() {
    Yolo yolo;
    cv::Mat src = cv::imread("../../images/707.jpg");
    std::vector<Yolo::Object> detected_objects;
    for(int i = 0; i < 50; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        yolo.Process(src,detected_objects);
        // std::cout << i << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = (end - start);
        std::cout<<"use "<< diff.count()<<" s" << std::endl;

    }
    Yolo::Draw(src, detected_objects);
    cv::imshow("src",src);
    cv::waitKey(0);


}
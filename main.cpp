#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "darknet.h"
#include <espeak/speak_lib.h>

image mat_to_image(cv::Mat m)
{
    int h = m.rows;
    int w = m.cols;
    int c = m.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)(m.data);

    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            for(int k = 0; k < c; ++k){
                im.data[k*w*h + i*w + j] = data[i*w*c + j*c + k]/255.0;
            }
        }
    }
    return im;
}

int main()
{
    // Загрузка модели
    std::string cfg = "/home/sie/darknet/cfg/yolov4-tiny.cfg";
    std::string weights = "/home/sie/darknet/yolov4-tiny.weights";
    std::string names_file = "/home/sie/darknet/data/coco.names";
    network *net = load_network((char*)cfg.c_str(), (char*)weights.c_str(), 0);
    set_batch_network(net, 1);

    // Загрузка меток
    char **names = get_labels((char*)names_file.c_str());
    
    // Инициализация espeak
    espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, NULL, 0);

    // Открытие камеры
    cv::VideoCapture capture(0); // 0 - номер камеры
    if(!capture.isOpened()){
        std::cout << "Не удалось открыть камеру\n";
        return 0;
    }

    cv::Mat frame;
    while(true){
        capture >> frame;
        image im = mat_to_image(frame);
        image sized = letterbox_image(im, net->w, net->h);

        // Предсказание
        float *X = sized.data;
        network_predict(net, X);

        // Получение результатов
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, 0.5, 0.5, 0, 1, &nboxes);

        // Отрисовка результатов   
        for(int i = 0; i < nboxes; i++){
            int best_class = max_index(dets[i].prob, 80);
            float prob = dets[i].prob[best_class];
            if(prob > 0.5){
                std::string label = names[best_class];
                std::string text = "Detected " + label;
                espeak_Synth(text.c_str(), text.size() + 1, 0, POS_CHARACTER, 0, espeakCHARS_AUTO, NULL, NULL); 
                draw_detections(im, dets, nboxes, 0.5, names, NULL, 80);
            }
        }
        free_detections(dets, nboxes);

        // Отображение кадра
        cv::imshow("frame", frame);

        free_image(im);
        free_image(sized);

        // Выход по нажатию 'q'
        if(cv::waitKey(1) == 'q') break;
    }

    // Освобождение ресурсов
    capture.release();
    cv::destroyAllWindows();

    return 0;
}


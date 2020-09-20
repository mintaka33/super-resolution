#include<stdio.h>
#include<iostream>
#include<string>
#include<map>

#include <inference_engine.hpp>
#include <ie_version.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace InferenceEngine;

string msr = "/home/fresh/data/model/single-image-super-resolution-1032.xml";
string inputImgFile = "/home/fresh/data/work/decode_sr_encode/build/sr/test.png";

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    T *blob_data = mblobHolder.as<T *>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] =
                        resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void infoIE(Core& ie, const string d)
{
    std::map<std::string, Version> vm =  ie.GetVersions(d);
    for(auto p : vm) {
        int x = p.second.apiVersion.major;
        int y = p.second.apiVersion.minor;
        string bn = p.second.buildNumber;
        string desc = p.second.description;
        string vs = std::to_string(x) + "." + std::to_string(y);
        cout << "INFO: device: " << p.first << ", version: " << vs << endl;
        cout << "INFO: build number: " << bn << endl;
        cout << "INFO: description: " << desc << endl;
    }
}

int main(int argc, char* argv[])
{
    const string device = "CPU";
    const string lrinputBlobName = "0";
    const string bicInputBlobName = "1";

    Core ie;

    infoIE(ie, device);

    auto network = ie.ReadNetwork(msr);
    network.setBatchSize(1);

    InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() == 1 || inputInfo.size() == 2) {
        printf("INFO: network requires %d input\n", inputInfo.size());
    } else {
        printf("ERROR: The network topologies with 1 or 2 inputs only\n");
        return -1;
    }

    auto lrInputInfoItem = inputInfo[lrinputBlobName];
    int w = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[3]);
    int h = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[2]);
    int c = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[1]);
    printf("INFO: input1 buffer dim: w = %d, h = %d, c = %d\n", w, h, c);

    auto bicInputInfoItem = inputInfo[bicInputBlobName];
    int w2 = static_cast<int>(bicInputInfoItem->getTensorDesc().getDims()[3]);
    int h2 = static_cast<int>(bicInputInfoItem->getTensorDesc().getDims()[2]);
    int c2 = static_cast<int>(bicInputInfoItem->getTensorDesc().getDims()[1]);
    printf("INFO: input2 buffer dim: w = %d, h = %d, c = %d\n", w2, h2, c2);

    OutputsDataMap outputInfo(network.getOutputsInfo());
    std::string firstOutputName;
    for (auto &item : outputInfo) {
        if (firstOutputName.empty()) {
            firstOutputName = item.first;
        }
        DataPtr outputData = item.second;
        if (!outputData) {
            printf("ERROR: output data pointer is not valid\n");
            return -1;
        }
        item.second->setPrecision(Precision::FP32);
    }
    auto outputInfoItem = outputInfo[firstOutputName];
    int w3 = static_cast<int>(outputInfoItem->getTensorDesc().getDims()[3]);
    int h3 = static_cast<int>(outputInfoItem->getTensorDesc().getDims()[2]);
    int c3 = static_cast<int>(outputInfoItem->getTensorDesc().getDims()[1]);
    printf("INFO: Output buffer dim: w = %d, h = %d, c = %d\n", w3, h3, c3);

    ExecutableNetwork executableNetwork = ie.LoadNetwork(network, device);
    InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // low resoution input
    Blob::Ptr lrInputBlob = inferRequest.GetBlob(lrinputBlobName);
    cv::Mat inputImg = cv::imread(inputImgFile, cv::IMREAD_COLOR);
    if (inputImg.empty()) {
        printf("ERROR: failed to load input impage file!\n");
        return -1;
    }
    matU8ToBlob<float_t>(inputImg, lrInputBlob, 0);

    // high resolution input from bicubic up-scaling
    cv::Mat resizedImg;
    Blob::Ptr bicInputBlob = inferRequest.GetBlob(bicInputBlobName);
    cv::resize(inputImg, resizedImg, cv::Size(w2, h2), 0, 0, cv::INTER_CUBIC);
    matU8ToBlob<float_t>(resizedImg, bicInputBlob, 0);

    // do inference
    inferRequest.Infer();

    // process output
    const Blob::Ptr outputBlob = inferRequest.GetBlob(firstOutputName);
    LockedMemory<const void> outputBlobMapped = as<MemoryBlob>(outputBlob)->rmap();
    const auto outputData = outputBlobMapped.as<float*>();
    size_t numOfImages = outputBlob->getTensorDesc().getDims()[0];
    size_t numOfChannels = outputBlob->getTensorDesc().getDims()[1];
    size_t nunOfPixels = w3 * h3;

    printf("INFO: Output size [N,C,H,W]: %d, %d, %d, %d\n", numOfImages, numOfChannels, h3, w3);

    for (size_t i = 0; i < numOfImages; ++i) {
        std::vector<cv::Mat> imgPlanes;
        if (numOfChannels == 3) {
            imgPlanes = std::vector<cv::Mat>{
                  cv::Mat(h3, w3, CV_32FC1, &(outputData[i * nunOfPixels * numOfChannels])),
                  cv::Mat(h3, w3, CV_32FC1, &(outputData[i * nunOfPixels * numOfChannels + nunOfPixels])),
                  cv::Mat(h3, w3, CV_32FC1, &(outputData[i * nunOfPixels * numOfChannels + nunOfPixels * 2]))};
        } else {
            imgPlanes = std::vector<cv::Mat>{cv::Mat(h3, w3, CV_32FC1, &(outputData[i * nunOfPixels * numOfChannels]))};
            // Post-processing for text-image-super-resolution models
            cv::threshold(imgPlanes[0], imgPlanes[0], 0.5f, 1.0f, cv::THRESH_BINARY);
        };

        for (auto & img : imgPlanes)
            img.convertTo(img, CV_8UC1, 255);
    
        cv::Mat resultImg;
        cv::merge(imgPlanes, resultImg);
        std::string outImgName = std::string("sr_" + std::to_string(i + 1) + ".png");
        cv::imwrite(outImgName, resultImg);
    }

    printf("\nExecution done!\n");
    return 0;
}
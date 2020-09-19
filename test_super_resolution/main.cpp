#include<stdio.h>
#include<iostream>
#include<string>
#include<map>

#include <inference_engine.hpp>
#include <ie_version.hpp>

using namespace std;
using namespace InferenceEngine;

string msr = "/home/fresh/data/model/single-image-super-resolution-1032.xml";

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
    const string inputBlobName = "0";

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

    auto lrInputInfoItem = inputInfo[inputBlobName];
    int w = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[3]);
    int h = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[2]);
    int c = static_cast<int>(lrInputInfoItem->getTensorDesc().getDims()[1]);
    printf("INFO: input buffer dim: w = %d, h = %d, c = %d\n", w, h, c);

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

    ExecutableNetwork executableNetwork = ie.LoadNetwork(network, device);
    InferRequest inferRequest = executableNetwork.CreateInferRequest();
    Blob::Ptr lrInputBlob = inferRequest.GetBlob(inputBlobName);

    printf("\nExecution done!\n");
    return 0;
}
#include<stdio.h>
#include<iostream>
#include<string>
#include<map>

#include <inference_engine.hpp>
#include <ie_version.hpp>

using namespace std;
using namespace InferenceEngine;

string msr = "/home/fresh/data/model/single-image-super-resolution-1032.xml";

void infoIE(Core& ie)
{
    std::map<std::string, Version> vm =  ie.GetVersions("CPU");
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
    Core ie;

    infoIE(ie);

    auto network = ie.ReadNetwork(msr);

    InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1 && inputInfo.size() != 2) {
        cout << "ERROR: The demo supports topologies with 1 or 2 inputs only" << endl;
        return -1;
    }


    printf("\nExecution done!\n");
    return 0;
}
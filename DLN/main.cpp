#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

std::unique_ptr<Ort::Session> onnx_session = nullptr;
std::unique_ptr<Ort::Env> onnx_env = nullptr;

void loadONNX(std::wstring model_name)
{
    const wchar_t* model_path = model_name.c_str();

    Ort::SessionOptions session_options{ nullptr };
    auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Low-light enhancement");
    onnx_env = std::move(envLocal);
    auto sessionLocal = std::make_unique<Ort::Session>(*onnx_env, model_path, session_options);
    onnx_session = std::move(sessionLocal);

    std::cout << "DLN model loaded." << std::endl;
}
 
cv::Mat postprocess(std::vector<float> rawData, unsigned int width, unsigned int height)
{
    for (int i = 0; i < rawData.size(); i++)
    {
        rawData[i] *= 255.0;
        if (rawData[i] > 255) rawData[i] = 255;
        else if (rawData[i] < 0) rawData[i] = 0;
    }

    size_t numOfPixels = width * height;
    cv::Mat1f r_channel = cv::Mat1f(height, width, &rawData[0]);
    cv::Mat1f g_channel = cv::Mat1f(height, width, &rawData[numOfPixels]);
    cv::Mat1f b_channel = cv::Mat1f(height, width, &rawData[2 * numOfPixels]);

    std::vector<cv::Mat> channels{ b_channel, g_channel, r_channel };

    cv::Mat output;

    merge(channels, output);

    output.convertTo(output, CV_8UC3);

    return output;
}

int main()
{
    std::wstring model_path_str(L"model.onnx");
    auto imageFilepath = "Input.png";

    loadONNX(model_path_str);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = onnx_session->GetInputCount();
    size_t numOutputNodes = onnx_session->GetOutputCount();

    const char* inputName = onnx_session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = onnx_session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    const char* outputName = onnx_session->GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = onnx_session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    cv::Mat frame = cv::imread(imageFilepath);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    int channel = inputDims[1];
    int height = inputDims[2];
    int width = inputDims[3];

    cv::Mat blob;

    // ONNX: (N x 3 x H x W)
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(width, height)); 

    size_t TensorSize = height * width * channel;
    std::vector<float> inputTensorValues(TensorSize);
    inputTensorValues.assign(blob.begin<float>(), blob.end<float>());

    std::vector<float> outputTensorValues(TensorSize);

    std::vector<const char*> inputNames{ inputName };
    std::vector<const char*> outputNames{ outputName };

    // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value inputTensors = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), TensorSize, inputDims.data(), inputDims.size());
    Ort::Value outputTensors = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), TensorSize, outputDims.data(), outputDims.size());

    onnx_session->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &inputTensors, 1, outputNames.data(), &outputTensors, 1);
    
    cv::Mat output = postprocess(outputTensorValues, width, height);
    
    // cv::imwrite("result.png", output);

    cv::imshow("Result", output);
    cv::waitKey(0);

    return 0;
}

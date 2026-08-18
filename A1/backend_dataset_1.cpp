#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <future>

namespace py = pybind11;
namespace fs = std::filesystem;

class Tensor {
public:
    std::vector<float> vec_data;
    std::vector<float> vec_grad;
    std::vector<float> vec_vel;
    std::vector<int> shape;

    Tensor(std::vector<int> s) : shape(s) {
        size_t sz = 1;
        for (int d : s) sz *= d;
        vec_data.assign(sz, 0.0f);
        vec_grad.assign(sz, 0.0f);
        vec_vel.assign(sz, 0.0f);
    }

    void zero_grad() {
        std::fill(vec_grad.begin(), vec_grad.end(), 0.0f);
    }

    float grad_abs_sum() {
        float s = 0;
        for (float g : vec_grad) s += std::abs(g);
        return s;
    }
};

class Ops {
public:
    // DATASET 
    static py::tuple load_dataset_1(std::string base_path) {
        std::vector<std::pair<std::string, int>> files;
        for (int label = 0; label < 10; ++label) {
            fs::path p = fs::path(base_path) / std::to_string(label);
            if (!fs::exists(p)) continue;
            for (const auto& entry : fs::directory_iterator(p))
                files.push_back({entry.path().string(), label});
        }

        std::vector<Tensor*> tensors(files.size());
        std::vector<int> labels(files.size());

        int T = std::thread::hardware_concurrency();
        int chunk = (files.size() + T - 1) / T;
        std::vector<std::future<void>> futures;

        for (int t = 0; t < T; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                for (int i = t * chunk; i < std::min((int)files.size(), (t + 1) * chunk); ++i) {
                    cv::Mat img = cv::imread(files[i].first);
                    if (img.empty()) continue;
                    cv::resize(img, img, cv::Size(32, 32));

                    Tensor* tens = new Tensor({1, 3, 32, 32});
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < 32; ++y)
                            for (int x = 0; x < 32; ++x)
                                tens->vec_data[c*1024 + y*32 + x] =
                                    ((float)img.at<cv::Vec3b>(y, x)[c] / 255.0f - 0.5f) * 2.0f;

                    tensors[i] = tens;
                    labels[i] = files[i].second;
                }
            }));
        }
        for (auto& f : futures) f.get();
        return py::make_tuple(tensors, labels);
    }

    // FLATTEN
    static void flatten(const Tensor& in, Tensor& out) {
        std::copy(in.vec_data.begin(), in.vec_data.end(), out.vec_data.begin());
    }

    //  LOSS 
    static float softmax_cross_entropy_grad(Tensor& logits, int target) {
        int n = logits.vec_data.size();
        std::vector<float> probs(n);
        float max_l = *std::max_element(logits.vec_data.begin(), logits.vec_data.end());

        float sum = 0;
        for (int i = 0; i < n; ++i) {
            probs[i] = std::exp(logits.vec_data[i] - max_l);
            sum += probs[i];
        }
        for (int i = 0; i < n; ++i) {
            probs[i] /= sum;
            logits.vec_grad[i] = probs[i] - (i == target ? 1.0f : 0.0f);
        }
        return -std::log(std::max(probs[target], 1e-12f));
    }

    // CONV 
    static void conv2d_fwd(const Tensor& in, const Tensor& w, const Tensor& b,
                           Tensor& out, int s, int p) {
        int IC = in.shape[1], IH = in.shape[2], IW = in.shape[3];
        int OC = w.shape[0], KH = w.shape[2], KW = w.shape[3];
        int OH = out.shape[2], OW = out.shape[3];

        std::fill(out.vec_data.begin(), out.vec_data.end(), 0.0f);

        for (int oc = 0; oc < OC; ++oc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = b.vec_data[oc];
                    for (int ic = 0; ic < IC; ++ic)
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * s - p + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * s - p + kw;
                                if (iw < 0 || iw >= IW) continue;
                                sum += in.vec_data[(ic*IH+ih)*IW+iw] *
                                       w.vec_data[((oc*IC+ic)*KH+kh)*KW+kw];
                            }
                        }
                    out.vec_data[(oc*OH+oh)*OW+ow] = sum;
                }
    }

    static void conv2d_bwd(Tensor& in, Tensor& w, Tensor& b,
                           const Tensor& grad_out, int s, int p) {
        int IC = in.shape[1], IH = in.shape[2], IW = in.shape[3];
        int OC = w.shape[0], KH = w.shape[2], KW = w.shape[3];
        int OH = grad_out.shape[2], OW = grad_out.shape[3];

        for (int oc = 0; oc < OC; ++oc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float go = grad_out.vec_grad[(oc*OH+oh)*OW+ow];
                    if (go == 0) continue;
                    b.vec_grad[oc] += go;
                    for (int ic = 0; ic < IC; ++ic)
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * s - p + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * s - p + kw;
                                if (iw < 0 || iw >= IW) continue;
                                w.vec_grad[((oc*IC+ic)*KH+kh)*KW+kw] +=
                                    in.vec_data[(ic*IH+ih)*IW+iw] * go;
                                in.vec_grad[(ic*IH+ih)*IW+iw] +=
                                    w.vec_data[((oc*IC+ic)*KH+kh)*KW+kw] * go;
                            }
                        }
                }
    }

    // LINEAR 
    static void linear_fwd(const Tensor& in, const Tensor& w, const Tensor& b, Tensor& out) {
        int InF = in.vec_data.size();
        int OutF = w.shape[0];
        for (int j = 0; j < OutF; ++j) {
            float sum = b.vec_data[j];
            for (int k = 0; k < InF; ++k)
                sum += in.vec_data[k] * w.vec_data[j*InF + k];
            out.vec_data[j] = sum;
        }
    }

    static void linear_bwd(Tensor& in, Tensor& w, Tensor& b, const Tensor& grad_out) {
        int InF = in.vec_data.size();
        int OutF = w.shape[0];
        for (int j = 0; j < OutF; ++j) {
            float g = grad_out.vec_grad[j];
            if (g == 0) continue;
            b.vec_grad[j] += g;
            for (int k = 0; k < InF; ++k) {
                w.vec_grad[j*InF + k] += in.vec_data[k] * g;
                in.vec_grad[k] += w.vec_data[j*InF + k] * g;
            }
        }
    }

    // MAXPOOL
    static void maxpool_fwd(const Tensor& in, Tensor& out, Tensor& mask, int k, int s) {
        int C = in.shape[1], IH = in.shape[2], IW = in.shape[3];
        int OH = out.shape[2], OW = out.shape[3];

        std::fill(mask.vec_data.begin(), mask.vec_data.end(), 0.0f);

        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float maxv = -1e30;
                    int maxidx = -1;
                    for (int kh = 0; kh < k; ++kh)
                        for (int kw = 0; kw < k; ++kw) {
                            int ih = oh*s + kh;
                            int iw = ow*s + kw;
                            int idx = (c*IH + ih)*IW + iw;
                            if (in.vec_data[idx] > maxv) {
                                maxv = in.vec_data[idx];
                                maxidx = idx;
                            }
                        }
                    out.vec_data[(c*OH+oh)*OW+ow] = maxv;
                    mask.vec_data[maxidx] = 1.0f;
                }
    }

    static void maxpool_bwd(const Tensor& grad_out, Tensor& grad_in, const Tensor& mask) {
        for (size_t i = 0; i < grad_in.vec_grad.size(); ++i)
            if (mask.vec_data[i] > 0)
                grad_in.vec_grad[i] += grad_out.vec_grad[i];
    }

    // RELU 
    static void relu_fwd(Tensor& t) {
        for (float& v : t.vec_data) if (v < 0) v = 0;
    }

    static void relu_bwd(const Tensor& out, Tensor& grad) {
        for (size_t i = 0; i < out.vec_data.size(); ++i)
            if (out.vec_data[i] <= 0) grad.vec_grad[i] = 0;
    }

    // OPTIMIZER
    static void sgd_momentum_step(Tensor& p, float lr, float m, float wd) {
        for (size_t i = 0; i < p.vec_data.size(); ++i) {
            float g = p.vec_grad[i] + wd * p.vec_data[i];
            p.vec_vel[i] = m * p.vec_vel[i] + g;
            p.vec_data[i] -= lr * p.vec_vel[i];
        }
    }
};

PYBIND11_MODULE(custom_core_dataset_1, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def_readwrite("data", &Tensor::vec_data)
        .def_readwrite("grad", &Tensor::vec_grad)
        .def_readwrite("shape", &Tensor::shape)
        .def("zero_grad", &Tensor::zero_grad)
        .def("grad_abs_sum", &Tensor::grad_abs_sum);

    m.def("load_dataset_1", &Ops::load_dataset_1);
    m.def("flatten", &Ops::flatten);
    m.def("softmax_cross_entropy_grad", &Ops::softmax_cross_entropy_grad);
    m.def("conv2d_fwd", &Ops::conv2d_fwd);
    m.def("conv2d_bwd", &Ops::conv2d_bwd);
    m.def("linear_fwd", &Ops::linear_fwd);
    m.def("linear_bwd", &Ops::linear_bwd);
    m.def("maxpool_fwd", &Ops::maxpool_fwd);
    m.def("maxpool_bwd", &Ops::maxpool_bwd);
    m.def("relu_fwd", &Ops::relu_fwd);
    m.def("relu_bwd", &Ops::relu_bwd);
    m.def("sgd_momentum_step", &Ops::sgd_momentum_step);
}


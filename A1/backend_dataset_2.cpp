#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <future>
#include <unordered_map>

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
};

class Ops {
public:
 
static py::tuple load_dataset_1(const std::string& base_path) {
    std::vector<std::pair<std::string,int>> files;

    for (int label = 0; label < 10; ++label) {
        fs::path p = fs::path(base_path) / std::to_string(label);
        if (!fs::exists(p)) continue;
        for (auto& e : fs::directory_iterator(p))
            files.push_back({e.path().string(), label});
    }

    return load_images(files);
}

static py::tuple load_dataset_2(const std::string& base_path) {
    std::vector<std::string> class_names;

    for (auto& e : fs::directory_iterator(base_path))
        if (e.is_directory())
            class_names.push_back(e.path().filename().string());

    std::sort(class_names.begin(), class_names.end());

    std::unordered_map<std::string,int> cls2id;
    for (int i = 0; i < (int)class_names.size(); ++i)
        cls2id[class_names[i]] = i;

    std::vector<std::pair<std::string,int>> files;
    for (auto& cls : class_names) {
        fs::path p = fs::path(base_path) / cls;
        for (auto& e : fs::directory_iterator(p))
            files.push_back({e.path().string(), cls2id[cls]});
    }

    return load_images(files);
}

static py::tuple load_images(const std::vector<std::pair<std::string,int>>& files) {
    std::vector<Tensor*> tensors(files.size());
    std::vector<int> labels(files.size());

    int T = std::max(1u, std::thread::hardware_concurrency());
    int chunk = (files.size() + T - 1) / T;

    std::vector<std::future<void>> futs;

    for (int t = 0; t < T; ++t) {
        futs.push_back(std::async(std::launch::async, [&, t]() {
            for (int i = t * chunk; i < std::min((int)files.size(), (t + 1) * chunk); ++i) {
                cv::Mat img = cv::imread(files[i].first);
                if (img.empty()) continue;

                cv::resize(img, img, cv::Size(32,32));
                Tensor* ten = new Tensor({1,3,32,32});

                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < 32; ++y)
                        for (int x = 0; x < 32; ++x)
                            ten->vec_data[c*1024 + y*32 + x] =
                                (img.at<cv::Vec3b>(y,x)[c]/255.0f - 0.5f) * 2.0f;

                tensors[i] = ten;
                labels[i] = files[i].second;
            }
        }));
    }
    for (auto& f : futs) f.get();
    return py::make_tuple(tensors, labels);
}

static void flatten(const Tensor& in, Tensor& out) {
    std::copy(in.vec_data.begin(), in.vec_data.end(), out.vec_data.begin());
}

static float softmax_cross_entropy_grad(Tensor& logits, int target) {
    int n = logits.vec_data.size();
    float maxv = *std::max_element(logits.vec_data.begin(), logits.vec_data.end());

    float sum = 0;
    for (int i = 0; i < n; ++i) {
        logits.vec_data[i] = std::exp(logits.vec_data[i] - maxv);
        sum += logits.vec_data[i];
    }

    for (int i = 0; i < n; ++i) {
        float p = logits.vec_data[i] / sum;
        logits.vec_grad[i] = p - (i == target);
        logits.vec_data[i] = p;
    }
    return -std::log(std::max(logits.vec_data[target], 1e-12f));
}

static void conv2d_fwd(const Tensor& in, const Tensor& w, const Tensor& b,
                       Tensor& out, int s, int p) {
    int IC = in.shape[1], IH = in.shape[2], IW = in.shape[3];
    int OC = w.shape[0], KH = w.shape[2], KW = w.shape[3];
    int OH = out.shape[2], OW = out.shape[3];

    std::fill(out.vec_data.begin(), out.vec_data.end(), 0);

    for (int oc=0; oc<OC; ++oc)
        for (int oh=0; oh<OH; ++oh)
            for (int ow=0; ow<OW; ++ow) {
                float sum = b.vec_data[oc];
                for (int ic=0; ic<IC; ++ic)
                    for (int kh=0; kh<KH; ++kh) {
                        int ih = oh*s - p + kh;
                        if (ih<0||ih>=IH) continue;
                        for (int kw=0; kw<KW; ++kw) {
                            int iw = ow*s - p + kw;
                            if (iw<0||iw>=IW) continue;
                            sum += in.vec_data[(ic*IH+ih)*IW+iw] *
                                   w.vec_data[((oc*IC+ic)*KH+kh)*KW+kw];
                        }
                    }
                out.vec_data[(oc*OH+oh)*OW+ow] = sum;
            }
}

static void conv2d_bwd(Tensor& in, Tensor& w, Tensor& b,
                       const Tensor& gout, int s, int p) {
    int IC=in.shape[1], IH=in.shape[2], IW=in.shape[3];
    int OC=w.shape[0], KH=w.shape[2], KW=w.shape[3];
    int OH=gout.shape[2], OW=gout.shape[3];

    for (int oc=0; oc<OC; ++oc)
        for (int oh=0; oh<OH; ++oh)
            for (int ow=0; ow<OW; ++ow) {
                float g = gout.vec_grad[(oc*OH+oh)*OW+ow];
                if (!g) continue;
                b.vec_grad[oc] += g;
                for (int ic=0; ic<IC; ++ic)
                    for (int kh=0; kh<KH; ++kh) {
                        int ih=oh*s-p+kh;
                        if (ih<0||ih>=IH) continue;
                        for (int kw=0; kw<KW; ++kw) {
                            int iw=ow*s-p+kw;
                            if (iw<0||iw>=IW) continue;
                            int wi=((oc*IC+ic)*KH+kh)*KW+kw;
                            int ii=(ic*IH+ih)*IW+iw;
                            w.vec_grad[wi]+=in.vec_data[ii]*g;
                            in.vec_grad[ii]+=w.vec_data[wi]*g;
                        }
                    }
            }
}

static void linear_fwd(const Tensor& in, const Tensor& w,
                       const Tensor& b, Tensor& out) {
    int InF=in.vec_data.size(), OutF=w.shape[0];
    for (int j=0;j<OutF;++j){
        float sum=b.vec_data[j];
        for (int k=0;k<InF;++k)
            sum+=in.vec_data[k]*w.vec_data[j*InF+k];
        out.vec_data[j]=sum;
    }
}

static void linear_bwd(Tensor& in, Tensor& w, Tensor& b,
                       const Tensor& gout) {
    int InF=in.vec_data.size(), OutF=w.shape[0];
    for (int j=0;j<OutF;++j){
        float g=gout.vec_grad[j];
        if (!g) continue;
        b.vec_grad[j]+=g;
        for (int k=0;k<InF;++k){
            w.vec_grad[j*InF+k]+=in.vec_data[k]*g;
            in.vec_grad[k]+=w.vec_data[j*InF+k]*g;
        }
    }
}

static void maxpool_fwd(const Tensor& in, Tensor& out,
                        Tensor& mask, int k, int s) {
    int C=in.shape[1], IH=in.shape[2], IW=in.shape[3];
    int OH=out.shape[2], OW=out.shape[3];

    std::fill(mask.vec_data.begin(), mask.vec_data.end(), 0);

    for (int c=0;c<C;++c)
        for (int oh=0;oh<OH;++oh)
            for (int ow=0;ow<OW;++ow){
                float mx=-1e30f; int idx=-1;
                for (int kh=0;kh<k;++kh)
                    for (int kw=0;kw<k;++kw){
                        int ih=oh*s+kh, iw=ow*s+kw;
                        int i=(c*IH+ih)*IW+iw;
                        if (in.vec_data[i]>mx){
                            mx=in.vec_data[i];
                            idx=i;
                        }
                    }
                out.vec_data[(c*OH+oh)*OW+ow]=mx;
                mask.vec_data[idx]=1;
            }
}

static void maxpool_bwd(const Tensor& gout, Tensor& gin,
                        const Tensor& mask) {
    // gin and mask are size: C * IH * IW
    // gout is size: C * OH * OW

    int C  = gin.shape[1];
    int IH = gin.shape[2];
    int IW = gin.shape[3];
    int OH = gout.shape[2];
    int OW = gout.shape[3];

    for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                float g = gout.vec_grad[(c*OH + oh)*OW + ow];

                // route gradient only to max location
                for (int kh = 0; kh < 2; ++kh)
                    for (int kw = 0; kw < 2; ++kw) {
                        int ih = oh*2 + kh;
                        int iw = ow*2 + kw;
                        int idx = (c*IH + ih)*IW + iw;
                        if (mask.vec_data[idx])
                            gin.vec_grad[idx] += g;
                    }
            }
}

static void relu_fwd(Tensor& t) {
    for (float& v:t.vec_data) if(v<0)v=0;
}

static void relu_bwd(const Tensor& out, Tensor& grad) {
    for (size_t i=0;i<out.vec_data.size();++i)
        if(out.vec_data[i]<=0) grad.vec_grad[i]=0;
}

static void sgd_momentum_step(Tensor& p,float lr,float m,float wd){
    for(size_t i=0;i<p.vec_data.size();++i){
        float g=p.vec_grad[i]+wd*p.vec_data[i];
        p.vec_vel[i]=m*p.vec_vel[i]+g;
        p.vec_data[i]-=lr*p.vec_vel[i];
    }
}
};

PYBIND11_MODULE(custom_core_dataset_2, m) {
    py::class_<Tensor>(m,"Tensor")
        .def(py::init<std::vector<int>>())
        .def_readwrite("data",&Tensor::vec_data)
        .def_readwrite("grad",&Tensor::vec_grad)
        .def_readwrite("shape",&Tensor::shape)
        .def("zero_grad",&Tensor::zero_grad);

    m.def("load_dataset_1",&Ops::load_dataset_1);
    m.def("load_dataset_2",&Ops::load_dataset_2);
    m.def("flatten",&Ops::flatten);
    m.def("softmax_cross_entropy_grad",&Ops::softmax_cross_entropy_grad);
    m.def("conv2d_fwd",&Ops::conv2d_fwd);
    m.def("conv2d_bwd",&Ops::conv2d_bwd);
    m.def("linear_fwd",&Ops::linear_fwd);
    m.def("linear_bwd",&Ops::linear_bwd);
    m.def("maxpool_fwd",&Ops::maxpool_fwd);
    m.def("maxpool_bwd",&Ops::maxpool_bwd);
    m.def("relu_fwd",&Ops::relu_fwd);
    m.def("relu_bwd",&Ops::relu_bwd);
    m.def("sgd_momentum_step",&Ops::sgd_momentum_step);
}


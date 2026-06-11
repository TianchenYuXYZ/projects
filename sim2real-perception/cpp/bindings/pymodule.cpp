// pybind11 绑定: Python 评测直接复用 C++ 部署组件, 保证训练/部署一致性。
//
//   import s2r_cpp
//   rt = s2r_cpp.DeployRuntime("deploy_bundle")
//   action, accepted, score, t_us = rt.step(rgb_u8_hwc, proprio_f32)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "s2r/image_pipeline.hpp"
#include "s2r/runtime.hpp"

namespace py = pybind11;

PYBIND11_MODULE(s2r_cpp, m) {
    m.doc() = "Sim2Real C++ deployment runtime bindings";

    py::class_<s2r::ImagePipeline>(m, "ImagePipeline")
        .def(py::init<int>(), py::arg("target_size"))
        .def("preprocess_rgb",
             [](const s2r::ImagePipeline& self,
                py::array_t<unsigned char, py::array::c_style | py::array::forcecast> rgb) {
                 if (rgb.ndim() != 3 || rgb.shape(2) != 3) {
                     throw std::invalid_argument("expect HxWx3 uint8");
                 }
                 auto chw = self.preprocess_rgb(
                     rgb.data(), static_cast<int>(rgb.shape(0)),
                     static_cast<int>(rgb.shape(1)));
                 const int s = self.size();
                 py::array_t<float> out({3, s, s});
                 std::copy(chw.begin(), chw.end(), out.mutable_data());
                 return out;
             },
             py::arg("rgb"), "HxWx3 uint8 RGB -> 3xSxS float32 [0,1]");

    py::class_<s2r::DeployRuntime>(m, "DeployRuntime")
        .def(py::init<const std::filesystem::path&>(), py::arg("bundle_dir"))
        .def("step",
             [](s2r::DeployRuntime& self,
                py::array_t<unsigned char, py::array::c_style | py::array::forcecast> rgb,
                py::array_t<float, py::array::c_style | py::array::forcecast> proprio) {
                 if (rgb.ndim() != 3 || rgb.shape(2) != 3) {
                     throw std::invalid_argument("expect HxWx3 uint8 RGB");
                 }
                 std::span<const float> prop(proprio.data(),
                                             static_cast<size_t>(proprio.size()));
                 auto r = self.step_rgb(rgb.data(),
                                        static_cast<int>(rgb.shape(0)),
                                        static_cast<int>(rgb.shape(1)), prop);
                 py::array_t<float> action(static_cast<py::ssize_t>(r.action.size()));
                 std::copy(r.action.begin(), r.action.end(), action.mutable_data());
                 return py::make_tuple(action, r.accepted, r.gate_score, r.t_total_us);
             },
             py::arg("rgb"), py::arg("proprio"),
             "returns (action, accepted, gate_score, total_us)");
}

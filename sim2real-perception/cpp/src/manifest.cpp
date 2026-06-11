#include "s2r/manifest.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace s2r {

Manifest Manifest::load(const std::filesystem::path& bundle_dir) {
    const auto manifest_path = bundle_dir / "manifest.json";
    std::ifstream f(manifest_path);
    if (!f) {
        throw std::runtime_error("cannot open " + manifest_path.string());
    }
    nlohmann::json j = nlohmann::json::parse(f);

    Manifest m;
    m.version = j.at("version").get<int>();
    m.image_size = j.at("image").at("size").get<int>();
    m.feature_dim = j.at("feature_dim").get<int>();
    m.proprio_dim = j.at("proprio_dim").get<int>();
    m.action_dim = j.at("action_dim").get<int>();
    m.backbone = j.at("backbone").get<std::string>();

    const auto& files = j.at("files");
    m.perception_onnx = bundle_dir / files.at("perception").get<std::string>();
    m.policy_onnx = bundle_dir / files.at("policy").get<std::string>();

    const auto& gate = j.at("gate");
    m.n_anchors = gate.at("n_anchors").get<int>();
    if (m.n_anchors > 0) {
        m.tau = gate.at("tau").get<float>();
        m.anchors_bin = bundle_dir / files.at("anchors").get<std::string>();
    }

    for (const auto& p : {m.perception_onnx, m.policy_onnx}) {
        if (!std::filesystem::exists(p)) {
            throw std::runtime_error("bundle file missing: " + p.string());
        }
    }
    return m;
}

}  // namespace s2r
